"""Dead-air removal and punch-in zoom on one edited timeline (PLAN.md §4.4 item 8).

Pure timeline arithmetic; render.py turns the result into one filter graph.
"""
from __future__ import annotations

import bisect
import math

from . import config


ZOOMABLE = {"face-centred", "centre", "centre-crop"}  # split and letterbox never zoom [I]


def silences(words: list[dict], start: float, end: float, longer_than: float) -> list[tuple[float, float]]:
    """(speech end, next word start) for each inter-word silence above the threshold.

    Speech end is the latest end of any earlier word, so a long or overlapping
    word is never treated as silence.
    """
    spoken = []
    for word in words:
        if word.get("start") is None or word.get("end") is None:
            continue
        a, b = float(word["start"]), float(word["end"])
        if math.isfinite(a) and math.isfinite(b) and b > a and b > start and a < end:
            spoken.append((a, b))
    gaps, speech_end = [], None
    for a, b in sorted(spoken):
        if speech_end is not None and a - speech_end > longer_than:
            gaps.append((speech_end, a))
        speech_end = b if speech_end is None else max(speech_end, b)
    return gaps


def dead_air_keeps(words: list[dict], start: float, end: float) -> list[tuple[float, float]]:
    """Source intervals kept after cutting silences longer than DEAD_AIR_GAP_S.

    Each cut leaves DEAD_AIR_PAD_S of silence on both sides, so no word is clipped.
    """
    pad = config.DEAD_AIR_PAD_S
    keeps, cursor = [], start
    for silent_from, silent_to in silences(words, start, end, config.DEAD_AIR_GAP_S):
        cut_from, cut_to = max(cursor, silent_from + pad), min(end, silent_to - pad)
        if cut_to > cut_from:
            keeps.append((cursor, cut_from))
            cursor = cut_to
    keeps.append((cursor, end))
    return [(a, b) for a, b in keeps if b > a]


def half_frame(t: float, fps: float, origin: float) -> float:
    return origin + (math.floor((t - origin) * fps) + .5) / fps


def snap_keeps(keeps: list[tuple[float, float]], fps: float, origin: float = 0.0) -> list[tuple[float, float]]:
    """Move every boundary to the nearest half-frame point; merge keeps that touch.

    A keep then holds a whole number of frames, so its video and audio lengths
    are equal and joins add no A/V drift.
    """
    snapped: list[tuple[float, float]] = []
    for a, b in keeps:
        a, b = half_frame(a, fps, origin), half_frame(b, fps, origin)
        if snapped and a <= snapped[-1][1] + 1e-9:
            snapped[-1] = (snapped[-1][0], max(b, snapped[-1][1]))
        elif b > a:
            snapped.append((a, b))
    return snapped


def edited_time(t: float, keeps: list[tuple[float, float]]) -> float:
    """Source time to edited time. A time inside a cut maps to the next keep's start."""
    offset = 0.0
    for a, b in keeps:
        if t < a:
            return offset
        if t < b:
            return offset + t - a
        offset += b - a
    return offset


def zoom_switches(triggers: list[float], duration: float) -> list[float]:
    """Accepted switch times: every framing, first and last included, holds at
    least ZOOM_MIN_HOLD_S, so there is at most one change per hold [I]."""
    hold = config.ZOOM_MIN_HOLD_S
    switches, last = [], 0.0
    for t in sorted(triggers):
        if t - last >= hold - 1e-9 and duration - t >= hold - 1e-9:
            switches.append(t)
            last = t
    return switches


def plan_pacing(candidate: dict, words: list[dict], plan: dict, start: float, end: float,
                grid=None) -> dict:
    """Keeps, zoom triggers and accepted switches (edited seconds) for one clip.

    `grid()` returns the source video's (fps, origin). It is called only when a
    cut exists, so a clip with nothing to cut needs no extra probe.
    """
    dead_air = candidate.get("dead_air", True) is not False
    zoom = candidate.get("zoom", True) is not False
    keeps = dead_air_keeps(words, start, end) if dead_air else [(start, end)]
    fps = origin = None
    if len(keeps) > 1:
        fps, origin = grid()
        keeps = snap_keeps(keeps, fps, origin)
    duration = sum(b - a for a, b in keeps)
    triggers, offset = [], 0.0
    for a, b in keeps[:-1]:
        offset += b - a
        triggers.append(offset)  # every join
    triggers += [edited_time(float(t), keeps) for t in plan.get("speaker_changes") or [] if start < float(t) < end]
    if not dead_air:
        triggers += [edited_time(resume - config.DEAD_AIR_PAD_S, keeps)
                     for _, resume in silences(words, start, end, config.ZOOM_PAUSE_S)]
    kinds = {s.get("kind") for s in plan["segments"]} if plan.get("segments") else {plan.get("kind", plan.get("layout"))}
    switches = zoom_switches(triggers, duration) if zoom and kinds & ZOOMABLE else []
    return {"dead_air": dead_air, "zoom": zoom, "keeps": keeps, "duration": duration,
            "triggers": sorted(triggers), "switches": switches, "fps": fps, "origin": origin}


def video_pieces(keeps, segments, switches, snap) -> list[list]:
    """Video runs in edited order: [source start, source end, segment, zoomed]."""
    pieces: list[list] = []
    offset = 0.0
    for a, b in keeps:
        bounds = {a, b}
        bounds.update(snap(t) for s in segments for t in (s["start"], s["end"]) if a < t < b)
        bounds.update(snap(a + t - offset) for t in switches if offset < t < offset + b - a)
        ordered = sorted(t for t in bounds if a <= t <= b)
        for x, y in zip(ordered, ordered[1:]):
            if y - x < 1e-6:
                continue
            mid = (x + y) / 2
            segment = next(s for s in segments if s["start"] <= mid < s["end"])
            zoomed = segment["kind"] in ZOOMABLE and segment.get("zoom_ok", True) and bisect.bisect_right(switches, offset + mid - a) % 2 == 1
            last = pieces[-1] if pieces else None
            if last and abs(last[1] - x) < 1e-9 and last[2] is segment and last[3] == zoomed:
                last[1] = y
            else:
                pieces.append([x, y, segment, zoomed])
        offset += b - a
    return pieces


