"""S5: candidate ranking (PLAN.md §5.3-§5.4).

The model returns **word indices, not timestamps** (§2.2 item 3). Timestamps
are then looked up from the alignment data, which makes a hallucinated
timestamp structurally impossible: an out-of-range index is caught, and an
in-range one resolves to a time that was measured rather than generated.

Post-processing is deterministic filtering, never scoring. There is no weighted
blend: the model's order is the rank, and the human is the second judge (§5.4).
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

SENTENCE_END = re.compile(r"[.!?]['\"’”)]*$")

MIN_SURVIVING_CANDIDATES = 5
DEFAULT_MIN_DURATION_S = 30.0
DEFAULT_MAX_DURATION_S = 60.0


class RankingError(RuntimeError):
    """The model's output could not be used, after the one permitted retry."""


@dataclass
class Candidate:
    """One ranked moment. `rank` is the model's order, preserved throughout."""

    rank: int
    start_word: int
    end_word: int
    hook_text: str = ""
    why: str = ""
    topic: str = ""
    brief_flags: list[str] = field(default_factory=list)
    start: float | None = None
    end: float | None = None

    @property
    def duration(self) -> float:
        if self.start is None or self.end is None:
            return 0.0
        return self.end - self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank, "start_word": self.start_word, "end_word": self.end_word,
            "hook_text": self.hook_text, "why": self.why, "topic": self.topic,
            "brief_flags": self.brief_flags, "start": self.start, "end": self.end,
            "duration": self.duration,
        }


RANKING_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_word": {"type": "integer"},
                    "end_word": {"type": "integer"},
                    "hook_text": {"type": "string"},
                    "why": {"type": "string"},
                    "topic": {"type": "string"},
                    "brief_flags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_word", "end_word", "hook_text", "why", "topic"],
            },
        }
    },
    "required": ["candidates"],
}

PROMPT = """You are selecting standalone short-form clips from a long transcript.

The transcript below is one line per sentence. Each line begins with the word \
index of its first word, in square brackets. **Refer to moments only by word \
index.** Never output a timestamp; timestamps are looked up from alignment data.

Return JSON only: {{"candidates": [...]}}, ordered best first. Produce exactly \
{count} candidates.

Each candidate needs:
- `start_word`, `end_word`: inclusive word indices bounding the clip.
- `hook_text`: at most 8 words, the on-screen opener. Not a summary — the line \
that stops a scroll.
- `why`: one line. What makes it standalone, and what the payoff is.
- `topic`: two or three words.
- `brief_flags`: brief rules this clip might touch. Empty list if none.

Requirements:
- **Standalone comprehensibility.** A viewer with no context must follow it. \
No clip may depend on something said earlier in the source.
- **Hook in the first 3 seconds.** The opening sentence must earn attention.
- **Payoff before the end.** The clip must resolve, not trail off.
- **No overlapping candidates.**
- **Cover the whole source.** Spread candidates across the transcript rather \
than clustering at the start; late material is less likely to be picked by \
other clippers working from the same source.
- Target {min_duration:.0f}-{max_duration:.0f} seconds. Roughly \
{min_words}-{max_words} words at this speaker's pace.
{brief_block}
Transcript:
{transcript}"""


def build_transcript(words: Sequence[Any], with_speakers: bool = False) -> str:
    """One line per sentence, prefixed with the first word's index.

    `with_speakers` prefixes the speaker label too, which is what the step-4
    experiment varies. Diarization normally runs *after* ranking (§2.2 item 1),
    so the labelled form exists only via `--full-diarization`.
    """
    lines: list[str] = []
    current: list[str] = []
    start_idx = 0
    speaker = None

    for i, w in enumerate(words):
        text = w.word if hasattr(w, "word") else w["word"]
        spk = w.speaker if hasattr(w, "speaker") else w.get("speaker")
        if not current:
            start_idx = i
            speaker = spk
        current.append(text)
        if SENTENCE_END.search(text):
            prefix = f"[{start_idx}]"
            if with_speakers and speaker:
                prefix += f" {speaker}:"
            lines.append(f"{prefix} {' '.join(current)}")
            current = []

    if current:
        prefix = f"[{start_idx}]"
        if with_speakers and speaker:
            prefix += f" {speaker}:"
        lines.append(f"{prefix} {' '.join(current)}")
    return "\n".join(lines)


def sentence_starts(words: Sequence[Any]) -> list[int]:
    """Indices where a sentence begins. A clip may only start here."""
    starts = [0]
    for i, w in enumerate(words[:-1]):
        text = w.word if hasattr(w, "word") else w["word"]
        if SENTENCE_END.search(text):
            starts.append(i + 1)
    return starts


def sentence_ends(words: Sequence[Any]) -> list[int]:
    """Indices where a sentence ends. A clip may only end here."""
    ends = [
        i for i, w in enumerate(words)
        if SENTENCE_END.search(w.word if hasattr(w, "word") else w["word"])
    ]
    if not ends or ends[-1] != len(words) - 1:
        ends.append(len(words) - 1)
    return ends


def snap(candidate: Candidate, starts: list[int], ends: list[int], total: int) -> Candidate:
    """Move a span outward to the nearest sentence boundary (§2.2 item 3).

    Outward, not nearest: snapping inward could cut the very sentence that made
    the moment worth clipping.
    """
    lo = max(0, min(candidate.start_word, total - 1))
    hi = max(0, min(candidate.end_word, total - 1))
    if hi < lo:
        lo, hi = hi, lo

    start = max((s for s in starts if s <= lo), default=starts[0])
    end = min((e for e in ends if e >= hi), default=ends[-1])
    candidate.start_word, candidate.end_word = start, end
    return candidate


def resolve_times(candidate: Candidate, words: Sequence[Any]) -> Candidate:
    """Look timestamps up from alignment data. Never generated (§2.2 item 3)."""
    def get(w, key):
        return getattr(w, key) if hasattr(w, key) else w.get(key)

    head = [get(words[i], "start") for i in range(candidate.start_word, candidate.end_word + 1)]
    tail = [get(words[i], "end") for i in range(candidate.start_word, candidate.end_word + 1)]
    starts = [v for v in head if v is not None]
    ends = [v for v in tail if v is not None]
    candidate.start = min(starts) if starts else None
    candidate.end = max(ends) if ends else None
    return candidate


def remove_overlaps(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the higher-ranked candidate of any overlapping pair (§5.4 item 4)."""
    kept: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: x.rank):
        if c.start is None or c.end is None:
            continue
        clash = any(
            c.start < k.end and k.start < c.end
            for k in kept if k.start is not None and k.end is not None
        )
        if not clash:
            kept.append(c)
    return kept


def parse_candidates(raw: str) -> list[Candidate]:
    """Validate the model's JSON against the schema's essentials."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RankingError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise RankingError("missing a top-level 'candidates' array")

    out: list[Candidate] = []
    for rank, item in enumerate(data["candidates"], 1):
        if not isinstance(item, dict):
            raise RankingError(f"candidate {rank} is not an object")
        try:
            start = int(item["start_word"])
            end = int(item["end_word"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RankingError(f"candidate {rank} has no usable word indices: {exc}") from exc
        out.append(Candidate(
            rank=rank, start_word=start, end_word=end,
            hook_text=str(item.get("hook_text", "")).strip(),
            why=str(item.get("why", "")).strip(),
            topic=str(item.get("topic", "")).strip(),
            brief_flags=[str(f) for f in (item.get("brief_flags") or [])],
        ))
    if not out:
        raise RankingError("the candidates array is empty")
    return out


def post_process(
    candidates: list[Candidate],
    words: Sequence[Any],
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> tuple[list[Candidate], dict[str, int]]:
    """§5.4 steps 2-4. Returns survivors and a count of what each filter dropped."""
    starts, ends = sentence_starts(words), sentence_ends(words)
    total = len(words)
    dropped = {"out_of_range": 0, "no_timing": 0, "duration": 0, "overlap": 0}

    staged: list[Candidate] = []
    for c in candidates:
        if not (0 <= c.start_word < total) or not (0 <= c.end_word < total):
            dropped["out_of_range"] += 1
            continue
        snap(c, starts, ends, total)
        resolve_times(c, words)
        if c.start is None or c.end is None:
            dropped["no_timing"] += 1
            continue
        if not (min_duration_s <= c.duration <= max_duration_s):
            dropped["duration"] += 1
            continue
        staged.append(c)

    before = len(staged)
    kept = remove_overlaps(staged)
    dropped["overlap"] = before - len(kept)
    return kept, dropped


def rank(
    words: Sequence[Any],
    llm_fn: Callable[[str], str],
    count: int = 12,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
    with_speakers: bool = False,
    brief_block: str = "",
) -> tuple[list[Candidate], dict[str, Any]]:
    """One ranking pass with a single retry, then a hard stop (§5.4 item 1)."""
    transcript = build_transcript(words, with_speakers=with_speakers)
    prompt = PROMPT.format(
        count=count, transcript=transcript,
        min_duration=min_duration_s, max_duration=max_duration_s,
        min_words=int(min_duration_s * 2.5), max_words=int(max_duration_s * 2.5),
        brief_block=brief_block,
    )

    last_error = ""
    for attempt in (1, 2):
        raw = llm_fn(prompt if attempt == 1 else prompt + _RETRY_SUFFIX)
        try:
            parsed = parse_candidates(raw)
        except RankingError as exc:
            last_error = str(exc)
            continue
        kept, dropped = post_process(parsed, words, min_duration_s, max_duration_s)
        if len(kept) < MIN_SURVIVING_CANDIDATES:
            return kept, {
                "attempts": attempt, "returned": len(parsed),
                "dropped": dropped, "insufficient": True,
            }
        return kept, {"attempts": attempt, "returned": len(parsed), "dropped": dropped}

    raise RankingError(f"schema-invalid output after one retry: {last_error}")


_RETRY_SUFFIX = (
    "\n\nIMPORTANT: your previous reply was not valid. Return ONLY a JSON object "
    'with a top-level "candidates" array. Every candidate needs integer '
    "start_word and end_word. No markdown fences, no commentary."
)
