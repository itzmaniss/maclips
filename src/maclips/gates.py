"""Startup environment gates.

Every check here stops the process with a reason instead of degrading. That is
the project rule from PLAN.md: hard gates, not logs. Nothing in this module
writes to a logger; a failed check raises and the caller exits non-zero.

Run them once, before any stage does work:

    from maclips.gates import check_environment
    check_environment()
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from . import config

# Filters the render pipeline cannot do without. `ass` burns the word-highlight
# captions (PLAN.md §2.3) and `drawtext` draws the hook and commentary
# overlays; both come from libass/freetype, which Homebrew's slim `ffmpeg`
# formula omits. `scdet` finds shot boundaries for S6.
REQUIRED_FFMPEG_FILTERS = ("ass", "subtitles", "drawtext", "scdet", "crop", "scale", "loudnorm")

# h264_videotoolbox renders previews (S8), libx264 the final encode (S12).
REQUIRED_FFMPEG_ENCODERS = ("h264_videotoolbox", "libx264")

MIN_MACOS_MAJOR = 14  # Vision landmark APIs this project relies on


class EnvironmentGate(RuntimeError):
    """A startup precondition failed. The run must not continue."""

    def __init__(self, check: str, reason: str, fix: str = "") -> None:
        self.check = check
        self.reason = reason
        self.fix = fix
        message = f"[gate:{check}] {reason}"
        if fix:
            message += f"\n  fix: {fix}"
        super().__init__(message)


@dataclass
class GateReport:
    """What passed, so a failure message can say what was already fine."""

    passed: list[str] = field(default_factory=list)

    def ok(self, check: str, detail: str) -> None:
        self.passed.append(f"{check}: {detail}")


def _ffmpeg_list(kind: str) -> str:
    """Return `ffmpeg -filters` or `-encoders` output, or raise EnvironmentGate.

    Probes the *configured* binary (config.FFMPEG), never PATH, so the gate
    cannot pass against a build the pipeline will not use.
    """
    binary = config.FFMPEG
    if not binary.is_file():
        raise EnvironmentGate(
            "ffmpeg-present",
            f"no ffmpeg at the configured path: {binary}",
            "brew install ffmpeg-full, or point MACLIPS_FFMPEG at a build that "
            "has libass (the slim `ffmpeg` formula does not).",
        )
    try:
        proc = subprocess.run(
            [str(binary), "-hide_banner", f"-{kind}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentGate("ffmpeg-present", f"could not run `ffmpeg -{kind}`: {exc}") from exc
    return proc.stdout


def check_platform(report: GateReport) -> None:
    """Apple Silicon macOS only. PLAN.md §1.3 rules out every other target."""
    if sys.platform != "darwin":
        raise EnvironmentGate(
            "platform",
            f"this pipeline is macOS-only; got sys.platform={sys.platform!r}.",
            "run it on the M4 Pro. There is no cross-platform path by design.",
        )
    machine = platform.machine()
    if machine != "arm64":
        raise EnvironmentGate(
            "platform",
            f"Apple Silicon required; got machine={machine!r}.",
            "MLX and VideoToolbox both need arm64. Rosetta will not do.",
        )
    release = platform.mac_ver()[0]
    major = int(release.split(".")[0]) if release else 0
    if major < MIN_MACOS_MAJOR:
        raise EnvironmentGate(
            "platform",
            f"macOS {MIN_MACOS_MAJOR}+ required for the Vision APIs; got {release}.",
        )
    report.ok("platform", f"darwin/{machine} macOS {release}")


def check_python(report: GateReport) -> None:
    """The venv must be the uv-managed 3.12 one, not a system interpreter."""
    if sys.version_info[:2] != (3, 12):
        got = ".".join(str(p) for p in sys.version_info[:3])
        raise EnvironmentGate(
            "python",
            f"Python 3.12 required; got {got}.",
            "uv run maclips  (whispermlx caps at <3.14; torchcodec 0.7 at <=3.13)",
        )
    report.ok("python", ".".join(str(p) for p in sys.version_info[:3]))


def check_ffmpeg(report: GateReport) -> None:
    """ffmpeg must be able to burn captions and use both encoders."""
    binary = config.FFMPEG
    filters = _ffmpeg_list("filters")
    missing = [f for f in REQUIRED_FFMPEG_FILTERS if f" {f} " not in filters]
    if missing:
        raise EnvironmentGate(
            "ffmpeg-filters",
            f"ffmpeg is missing required filter(s): {', '.join(missing)}.",
            f"{binary} lacks them. Homebrew's default `ffmpeg` formula is built "
            "without libass, freetype or fontconfig, so `ass`, `subtitles` and "
            "`drawtext` are absent. Install ffmpeg-full (keg-only, no PATH change "
            "needed) or point MACLIPS_FFMPEG at a build that has them.",
        )

    binary = config.FFMPEG
    encoders = _ffmpeg_list("encoders")
    missing_enc = [e for e in REQUIRED_FFMPEG_ENCODERS if e not in encoders]
    if missing_enc:
        raise EnvironmentGate(
            "ffmpeg-encoders",
            f"ffmpeg is missing required encoder(s): {', '.join(missing_enc)}.",
            "brew install ffmpeg-full",
        )

    if not config.FFPROBE.is_file():
        raise EnvironmentGate(
            "ffprobe-present",
            f"no ffprobe at the configured path: {config.FFPROBE}; "
            "S0 cannot probe sources without it.",
            "brew install ffmpeg-full, or set MACLIPS_FFPROBE.",
        )
    report.ok("ffmpeg", f"{binary} — ass, subtitles, drawtext, scdet, loudnorm "
                        "+ libx264/h264_videotoolbox")


def check_mlx(report: GateReport) -> None:
    """MLX must see the GPU; transcription (S3) is the main MLX consumer."""
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise EnvironmentGate("mlx", f"mlx is not importable: {exc}", "uv sync") from exc
    device = mx.default_device()
    if "gpu" not in str(device):
        raise EnvironmentGate(
            "mlx",
            f"MLX is not on the GPU (default device is {device}).",
            "transcription on CPU would blow the 15-minute target.",
        )
    report.ok("mlx", f"default device {device}")


def check_torch_mps(report: GateReport) -> None:
    """Alignment and diarization stay on PyTorch; they need the MPS backend."""
    try:
        import torch
    except ImportError as exc:
        raise EnvironmentGate("torch", f"torch is not importable: {exc}", "uv sync") from exc
    if not torch.backends.mps.is_available():
        raise EnvironmentGate(
            "torch-mps",
            "PyTorch cannot reach the MPS backend.",
            "diarization on CPU is the documented slow path (PLAN.md §10 risk 2).",
        )
    report.ok("torch-mps", f"torch {torch.__version__} with MPS")


def check_vision(report: GateReport) -> None:
    """Apple Vision, via pyobjc, is the only face detector this project uses."""
    try:
        import Vision  # noqa: F401  (pyobjc-framework-Vision)
    except ImportError as exc:
        raise EnvironmentGate(
            "vision",
            f"pyobjc-framework-Vision is not importable: {exc}",
            "uv sync",
        ) from exc
    report.ok("vision", "pyobjc Vision framework importable")


def check_deno(report: GateReport) -> None:
    """yt-dlp's JS challenge solver runs on Deno.

    YouTube serves player challenges that must be executed, not parsed.
    `yt-dlp[default]` ships `yt_dlp_ejs` to drive them and needs a JS runtime;
    without one, extraction degrades to formats that may not include the
    resolutions S0 asks for, or fails outright.
    """
    binary = shutil.which("deno")
    if binary is None:
        raise EnvironmentGate(
            "deno",
            "deno is not on PATH; yt-dlp cannot run YouTube's JS challenges.",
            "brew install deno",
        )
    try:
        proc = subprocess.run(
            [binary, "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EnvironmentGate("deno", f"could not run `deno --version`: {exc}") from exc
    version = (proc.stdout or "").splitlines()[0].strip() if proc.stdout else "unknown"
    report.ok("deno", f"{version} at {binary}")


def check_yt_dlp_ejs(report: GateReport) -> None:
    """The `yt-dlp[default]` extra must actually be installed, not bare yt-dlp."""
    try:
        import yt_dlp_ejs  # noqa: F401
    except ImportError as exc:
        raise EnvironmentGate(
            "yt-dlp-ejs",
            f"yt_dlp_ejs is not importable: {exc}",
            'the plain `yt-dlp` dependency is not enough — it must be '
            '`yt-dlp[default]`. Run: uv add "yt-dlp[default]"',
        ) from exc
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("yt-dlp-ejs")
    except PackageNotFoundError:
        installed = "unknown"
    report.ok("yt-dlp-ejs", f"{installed} importable")


CHECKS = (
    check_platform,
    check_python,
    check_ffmpeg,
    check_deno,
    check_yt_dlp_ejs,
    check_mlx,
    check_torch_mps,
    check_vision,
)


def check_environment() -> GateReport:
    """Run every startup gate in order. Raises EnvironmentGate on the first failure."""
    report = GateReport()
    for check in CHECKS:
        check(report)
    return report
