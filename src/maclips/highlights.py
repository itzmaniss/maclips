"""Highlight ranking — the surviving seam from the fork.

Carried over from SamurAIGPT/AI-Youtube-Shorts-Generator (MIT; see NOTICE),
with the MuAPI backend removed. `llm_fn` is now a required argument rather
than defaulting to a hosted provider, so there is no hidden network call: the
caller passes `maclips.llm.rank_fn`.

**This is transitional.** PLAN.md §5.3-§5.4 replaces the contract at build
step 4:

  * the model returns `start_word` / `end_word` word indices, not timestamps,
    and boundaries are snapped to sentence starts using transcript punctuation
    (§2.2 item 3) — which is what makes hallucinated timestamps impossible;
  * `score` disappears. The model's rank order *is* the rank (§5.4);
  * chunking disappears. A 2-hour source is ~25-35k tokens and fits one call
    (§5.1), so `chunk_transcript` and the offset arithmetic below go with it.

What survives past step 4 is `_parse_json_loose` (models still fence JSON) and
`dedupe_highlights` (overlap suppression is still a deterministic filter).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

LLMFn = Callable[[str], str]

CHUNK_SIZE_SECONDS = 1200
LONG_VIDEO_THRESHOLD = 1800
CHUNK_OVERLAP_SECONDS = 60
MAX_HIGHLIGHT_ATTEMPTS = 3


def _parse_json_loose(raw: str) -> dict[str, Any]:
    """Strip markdown fences and parse, falling back to the outermost braces."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _sanitize_highlights(raw_highlights: object, duration: float) -> list[dict]:
    """Normalise model output, dropping entries that cannot be trusted."""
    if not isinstance(raw_highlights, list):
        return []

    max_end = duration if duration > 0 else float("inf")
    cleaned: list[dict] = []
    for item in raw_highlights:
        if not isinstance(item, dict):
            continue
        start = _coerce_float(item.get("start_time"), default=-1.0)
        end = _coerce_float(item.get("end_time"), default=-1.0)
        if start < 0 or end <= start:
            continue
        if max_end != float("inf"):
            start, end = min(start, max_end), min(end, max_end)
            if end <= start:
                continue
        cleaned.append(
            {
                "title": str(item.get("title") or "Untitled Highlight").strip(),
                "start_time": start,
                "end_time": end,
                "score": max(0, min(100, _coerce_int(item.get("score"), default=0))),
                "hook_sentence": str(item.get("hook_sentence") or "").strip(),
                "virality_reason": str(item.get("virality_reason") or "").strip(),
            }
        )
    return cleaned


def build_transcript_text(transcript: dict) -> str:
    segments = transcript.get("segments", [])
    return "\n".join(f"[{s['start']:.1f}s] {s['text'].strip()}" for s in segments)


def chunk_transcript(transcript: dict) -> list[dict]:
    """Split a long source into overlapping windows. Removed at step 4 (§5.1)."""
    segments = transcript.get("segments", [])
    duration = transcript.get("duration", segments[-1]["end"] if segments else 0)
    chunks: list[dict] = []
    start = 0.0
    while start < duration:
        end = min(start + CHUNK_SIZE_SECONDS, duration)
        chunk_segs = [
            s for s in segments
            if s["start"] >= start and s["end"] <= end + CHUNK_OVERLAP_SECONDS
        ]
        if chunk_segs:
            chunk = dict(transcript)
            chunk["segments"] = chunk_segs
            chunk["duration"] = end - start
            chunk["_offset"] = start
            chunks.append(chunk)
        start += CHUNK_SIZE_SECONDS - CHUNK_OVERLAP_SECONDS
    return chunks


def call_highlight_api(
    transcript_text: str,
    duration: float,
    num_clips: int,
    llm_fn: LLMFn,
    prompt_template: str,
    is_chunk: bool = False,
) -> dict:
    """One ranking call with a bounded retry, then a hard stop.

    PLAN.md §5.4 tightens this to a single retry before the S5 gate fires.
    """
    target = max(num_clips * 2, 5)
    natural_max = max(2 if is_chunk else 3, int(duration / 90))
    min_clips = min(target, natural_max, 8)
    base_prompt = prompt_template.format(
        num_clips_instruction=f"Generate at least {min_clips} highlights",
        transcript=transcript_text,
    )
    prompt = base_prompt
    last_error = "unknown"

    for attempt in range(1, MAX_HIGHLIGHT_ATTEMPTS + 1):
        raw = llm_fn(prompt)
        try:
            parsed = _parse_json_loose(raw)
            highlights = _sanitize_highlights(parsed.get("highlights"), duration=duration)
            if highlights:
                return {"highlights": highlights}
            last_error = "no valid highlights in response"
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            last_error = str(exc)

        if attempt < MAX_HIGHLIGHT_ATTEMPTS:
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Return ONLY valid JSON with a top-level "
                "'highlights' array. No markdown fences, no commentary."
            )

    raise RuntimeError(
        f"Ranking produced invalid output after {MAX_HIGHLIGHT_ATTEMPTS} attempts: {last_error}"
    )


def dedupe_highlights(highlights: list[dict]) -> list[dict]:
    """Drop a highlight overlapping >50% with a higher-scoring kept one."""
    highlights = sorted(highlights, key=lambda x: int(x.get("score", 0)), reverse=True)
    kept: list[dict] = []
    for h in highlights:
        h_start, h_end = float(h["start_time"]), float(h["end_time"])
        h_dur = h_end - h_start
        overlapping = False
        for k in kept:
            overlap = min(h_end, float(k["end_time"])) - max(h_start, float(k["start_time"]))
            if overlap > 0 and overlap > 0.5 * h_dur:
                overlapping = True
                break
        if not overlapping:
            kept.append(h)
    return kept


def get_highlights(
    transcript: dict,
    llm_fn: LLMFn,
    prompt_template: str,
    num_clips: int = 12,
) -> dict:
    """Rank candidates. `llm_fn` is the seam; nothing here picks a provider."""
    duration = transcript.get("duration", 0)

    if duration >= LONG_VIDEO_THRESHOLD:
        all_highlights: list[dict] = []
        for chunk in chunk_transcript(transcript):
            offset = chunk.get("_offset", 0)
            result = call_highlight_api(
                build_transcript_text(chunk), chunk["duration"], num_clips,
                llm_fn, prompt_template, is_chunk=True,
            )
            for h in result.get("highlights", []):
                h["start_time"] = float(h["start_time"]) + offset
                h["end_time"] = float(h["end_time"]) + offset
                all_highlights.append(h)
        highlights = dedupe_highlights(all_highlights)
    else:
        result = call_highlight_api(
            build_transcript_text(transcript), duration, num_clips, llm_fn, prompt_template
        )
        highlights = dedupe_highlights(result.get("highlights", []))

    return {"highlights": highlights}
