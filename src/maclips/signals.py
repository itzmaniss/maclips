"""Deterministic hint lines for ranking prompt v3 (PLAN.md §5.2i).

No model is involved. Every line names a transcript line by its [n], the
global index of its first word, exactly as `ranking.build_transcript` prints
it, and never by timestamp (§2.2 item 3). The hints are shown to the model as
hints only; nothing here filters or reorders candidates.

S5 runs before diarization (§2.2 item 1), so there are no speaker signals.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .ranking import SENTENCE_END

MAX_LINES = 12
PAUSE_BEFORE_S = 1.0
PEAK_SPACING_S = 4.0

# Words that tend to mark a claim, a surprise or a turn in the argument.
TRIGGERS = re.compile(
    r"\b(actually|honestly|truth|secret|mistake|wrong|never|nobody|everyone|"
    r"biggest|worst|crazy|insane|surprising|shocked|problem|hate|love|why)\b",
    re.IGNORECASE)
# First-person moments where a story turns.
STORY_TURNS = re.compile(
    r"\b(i|we)\s+(realised|realized|learned|learnt|lost|failed|quit|decided|"
    r"discovered|almost|nearly)\b|\bturns? out\b|\bthat's when\b|\bthe moment\b",
    re.IGNORECASE)


def _text(w: Any) -> str:
    return w.word if hasattr(w, "word") else w["word"]


def _time(w: Any, key: str) -> float | None:
    return getattr(w, key) if hasattr(w, key) else w.get(key)


def transcript_lines(words: Sequence[Any]) -> list[tuple[int, int]]:
    """(first, last) word index of each line `build_transcript` prints."""
    lines, first = [], 0
    for i, w in enumerate(words):
        if SENTENCE_END.search(_text(w)):
            lines.append((first, i))
            first = i + 1
    if first < len(words):
        lines.append((first, len(words) - 1))
    return lines


def loud_seconds(wav: Path) -> list[tuple[float, float]]:
    """(second, RMS dBFS) for each whole second of the WAV, via ffmpeg astats."""
    proc = subprocess.run(
        [str(config.FFMPEG), "-nostdin", "-hide_banner", "-i", str(wav), "-af",
         "aresample=16000,asetnsamples=n=16000:p=0,astats=metadata=1:reset=1,"
         "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg loudness scan failed: {proc.stderr[-300:]}")
    out, t = [], None
    for line in proc.stderr.splitlines():
        if m := re.search(r"pts_time:([0-9.]+)", line):
            t = float(m.group(1))
        elif (m := re.search(r"RMS_level=(-?[0-9.]+|-inf)", line)) and t is not None:
            if m.group(1) != "-inf":
                out.append((t, float(m.group(1))))
            t = None
    return out


def loud_peaks(seconds: list[tuple[float, float]], count: int = MAX_LINES) -> list[float]:
    """Start times of the loudest seconds, at least PEAK_SPACING_S apart."""
    peaks: list[float] = []
    for t, _ in sorted(seconds, key=lambda s: s[1], reverse=True):
        if all(abs(t - p) >= PEAK_SPACING_S for p in peaks):
            peaks.append(t)
            if len(peaks) == count:
                break
    return sorted(peaks)


def build(words: Sequence[Any], peaks: Sequence[float] = ()) -> str:
    """The hint block: up to MAX_LINES lines, strongest first by a fixed tally.

    The tally only picks which lines are shown; the block states no score.
    """
    rows = []
    lines = transcript_lines(words)
    for n, (first, last) in enumerate(lines):
        text = " ".join(_text(words[i]) for i in range(first, last + 1))
        reasons, weight = [], 0.0
        if TRIGGERS.search(text):
            reasons.append("trigger word"); weight += 2
        if "?" in text:
            reasons.append("question"); weight += 1.5
        if "!" in text:
            reasons.append("exclamation"); weight += 1
        if STORY_TURNS.search(text):
            reasons.append("story turn"); weight += 1.5
        start, end = _time(words[first], "start"), _time(words[last], "end")
        if first > 0 and start is not None:
            before = _time(words[first - 1], "end")
            if before is not None and start - before >= PAUSE_BEFORE_S:
                reasons.append(f"pause of {start - before:.1f}s before it"); weight += 1
        if start is not None and end is not None and any(start <= p < end for p in peaks):
            reasons.append("one of the loudest moments"); weight += 1.5
        if reasons:
            rows.append((weight, n, first, reasons))
    chosen = sorted(rows, key=lambda r: (-r[0], r[1]))[:MAX_LINES]
    if not chosen:
        return ""
    head = ("Hints only, computed without a model from the transcript text and the "
            "audio loudness. No speaker labels exist at this stage, so there are no "
            "speaker signals. Use them to notice moments; judge every clip on the "
            "transcript itself.")
    body = [f"[{first}] {', '.join(reasons)}" for _, _, first, reasons in sorted(chosen, key=lambda r: r[2])]
    return "\n".join([head, *body])
