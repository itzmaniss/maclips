"""ffmpeg via subprocess. No Python video wrapper, ever.

The subclip call is the fork's, kept because it was the one piece of its
clipping path worth keeping (NOTICE). Everything around it is gone: the fork
cut with libx264 and then re-encoded through an OpenCV `mp4v` VideoWriter,
which is the double encode PLAN.md §2.2 item 5 removes. S12 is the only
full-quality encode and it reads the original source once.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config


class FFmpegError(RuntimeError):
    """ffmpeg or ffprobe exited non-zero. Carries stderr, which is the useful part."""


def _run(cmd: list[str], timeout: float = 3600.0) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"{Path(cmd[0]).name} timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise FFmpegError(f"could not execute {cmd[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise FFmpegError(f"{Path(cmd[0]).name} exited {proc.returncode}:\n" + "\n".join(tail))
    return proc


def parse_ratio(aspect_ratio: str) -> float:
    """'9:16' -> 0.5625. From the fork."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def probe(path: Path) -> dict:
    """ffprobe a source into the dict S0 gates on."""
    proc = _run(
        [str(config.FFPROBE), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        timeout=120.0,
    )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"ffprobe returned unparseable JSON for {path}") from exc

    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    return {
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "duration_s": float(data.get("format", {}).get("duration", 0.0) or 0.0),
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


def cut_subclip(source: Path, start: float, end: float, out_path: Path) -> Path:
    """Cut [start, end] from `source`. Kept from the fork, with input seeking.

    The fork placed `-ss` after `-i`, which decodes from the file's start and
    throws frames away. Seeking before `-i` skips that work; the keyframe-
    accuracy cost does not matter because the final render re-encodes anyway.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        str(config.FFMPEG), "-y", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(source),
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        str(out_path),
    ])
    return out_path


def extract_audio(source: Path, out_path: Path, sample_rate: int = 16000) -> Path:
    """S2: decode to 16 kHz mono 16-bit PCM WAV.

    That format is not arbitrary — it is what whispermlx and pyannote both
    consume, so producing it once here means neither library ever has to decode
    media itself. This is what keeps torchcodec off the path entirely (§2.4).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        str(config.FFMPEG), "-y", "-nostdin", "-loglevel", "error",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(out_path),
    ])
    return out_path
