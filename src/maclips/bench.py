"""Stage timing and memory sampling.

**Why not "free memory":** macOS fills otherwise-idle RAM with file cache, so
free pages are not a measure of headroom — a machine reporting little "free"
memory may be perfectly comfortable. The numbers that mean something here are:

- **process RSS**, sampled at peak, for what this pipeline itself costs;
- **swap used**, because growth during a stage means the timing measured
  paging rather than the work;
- **memory pressure**, which is the signal macOS itself acts on.

Sampling runs on a background thread because peak RSS between two endpoint
readings is invisible — a model load that spikes and frees would not show up.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

SAMPLE_INTERVAL_S = 0.25

# macOS pressure levels from `kern.memorystatus_vm_pressure_level`.
PRESSURE_LEVELS = {1: "normal", 2: "warn", 4: "critical"}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def swap_used_mb() -> float:
    """Swap in use, from `sysctl vm.swapusage`."""
    out = _run(["sysctl", "-n", "vm.swapusage"])
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
    if not m:
        return 0.0
    value = float(m.group(1))
    return value * 1024 if m.group(2) == "G" else value


def compressed_gib() -> float:
    """Pages held by the memory compressor — pressure before swap becomes visible.

    The page size is read from `vm_stat` rather than assumed: Apple Silicon
    uses 16 KiB pages, and hardcoding 4096 understates this by 4x.
    """
    vm = _run(["vm_stat"])
    size_match = re.search(r"page size of (\d+) bytes", vm)
    page = int(size_match.group(1)) if size_match else 16384
    comp = re.search(r"Pages occupied by compressor:\s+(\d+)", vm)
    return int(comp.group(1)) * page / 2**30 if comp else 0.0


def memory_pressure() -> str:
    """macOS's own pressure verdict, from the sysctl it actually acts on.

    `memory_pressure -Q` prints a free percentage but no level, and free
    percentage is the misleading number (macOS fills free RAM with file
    cache). `kern.memorystatus_vm_pressure_level` is the verdict itself.
    """
    raw = _run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    try:
        level = PRESSURE_LEVELS.get(int(raw), f"level {raw}")
    except ValueError:
        level = "unknown"
    return f"{level}, compressed {compressed_gib():.1f}GiB"


def rss_mb() -> float:
    """This process's **current** resident size, in MiB.

    `resource.getrusage().ru_maxrss` is deliberately not used: it is a
    high-water mark that never falls, so every stage after the first would
    inherit the largest earlier peak and per-stage figures would be
    meaningless. `ps` reports the live value (in KiB).
    """
    out = _run(["ps", "-o", "rss=", "-p", str(os.getpid())]).strip()
    try:
        return int(out) / 1024
    except ValueError:
        return 0.0


@dataclass
class StageMetrics:
    name: str
    wall_s: float = 0.0
    peak_rss_mb: float = 0.0
    rss_before_mb: float = 0.0
    swap_before_mb: float = 0.0
    swap_after_mb: float = 0.0
    pressure_at_peak: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def swap_delta_mb(self) -> float:
        return self.swap_after_mb - self.swap_before_mb

    @property
    def swapped(self) -> bool:
        """Swap growth above noise means the timing is measuring paging."""
        return self.swap_delta_mb > 32.0


class measure:
    """Context manager: time a stage and sample its peak memory.

        with measure("S3 transcribe") as m:
            ...
        print(m.wall_s, m.peak_rss_mb)
    """

    def __init__(self, name: str) -> None:
        self.metrics = StageMetrics(name=name)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak = 0.0
        self._peak_pressure = ""

    def _sample(self) -> None:
        while not self._stop.wait(SAMPLE_INTERVAL_S):
            current = rss_mb()
            if current > self._peak:
                self._peak = current
                self._peak_pressure = memory_pressure()

    def __enter__(self) -> StageMetrics:
        m = self.metrics
        m.rss_before_mb = rss_mb()
        m.swap_before_mb = swap_used_mb()
        self._peak = m.rss_before_mb
        self._peak_pressure = memory_pressure()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._start = time.perf_counter()
        return m

    def __exit__(self, *exc: object) -> None:
        self.metrics.wall_s = time.perf_counter() - self._start
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.metrics.peak_rss_mb = max(self._peak, rss_mb())
        self.metrics.pressure_at_peak = self._peak_pressure
        self.metrics.swap_after_mb = swap_used_mb()


def render_table(rows: list[StageMetrics], audio_duration_s: float = 0.0) -> str:
    """The benchmark table, with a realtime factor when the duration is known."""
    head = (
        f"{'stage':<20} {'wall':>9} {'xRT':>7} {'peak RSS':>10} "
        f"{'swap Δ':>9}  pressure at peak"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        xrt = f"{audio_duration_s / r.wall_s:.1f}x" if audio_duration_s and r.wall_s else "-"
        flag = "  ** SWAPPED **" if r.swapped else ""
        lines.append(
            f"{r.name:<20} {r.wall_s:>8.1f}s {xrt:>7} {r.peak_rss_mb:>9.0f}M "
            f"{r.swap_delta_mb:>+8.0f}M  {r.pressure_at_peak}{flag}"
        )
    total = sum(r.wall_s for r in rows)
    lines.append("-" * len(head))
    total_xrt = f"{audio_duration_s / total:.1f}x" if audio_duration_s and total else "-"
    lines.append(f"{'TOTAL':<20} {total:>8.1f}s {total_xrt:>7}")
    return "\n".join(lines)
