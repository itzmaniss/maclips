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
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

SENTENCE_END = re.compile(r"[.!?]['\"’”)]*$")
# A cold-open line may be a phrase: it may also end at a clause mark.
PHRASE_END = re.compile(r"([.!?,;:]|--|[—–])['\"’”)]*$")
# Whisper writes "..." for a hesitation, usually mid-sentence (§6.7): a clip or
# line never ends there. The transcript shown to the model is unchanged.
TRAILING_OFF = re.compile(r"(\.\.\.|…)['\"’”)]*$")
# A sentence end followed this closely by the same speaker is a run-on (§6.7).
RUN_ON_GAP_S = 0.3
# At most this many sentences are added to reach a pause. On video 2's fast
# speaker the needed extensions were 1-2 sentences (0.5-6.6 s) or 3-11
# (11.8-43 s), material the model never picked (§6.7) [V].
RUN_ON_MAX_SENTENCES = 2

MIN_SURVIVING_CANDIDATES = 5
# Used only when the brief states no range. Any clip postable on Reels, TikTok
# and Shorts is fine (user rule): 180 s is the smallest platform maximum
# (Shorts), 10 s the floor four captured briefs state (§6.1).
DEFAULT_MIN_DURATION_S = 10.0
DEFAULT_MAX_DURATION_S = 180.0
# A cold-open line (tease or payoff) must play for 1.5-4.0 s after snapping.
COLD_OPEN_MIN_S = 1.5
COLD_OPEN_MAX_S = 4.0
COLD_OPEN_MODES = ("tease", "payoff")

# Run-level S5 duration presets, used only when the brief states no duration.
# `shorts-dense` comes from the user's retention research, which cites a vendor
# source [U]; `default` is today's range. Only the numbers change, never PROMPT.
LENGTH_PRESETS = {
    "default": (DEFAULT_MIN_DURATION_S, DEFAULT_MAX_DURATION_S),
    "shorts-dense": (22.0, 45.0),
}


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
    # Display and log only (§5.7); never sorted, filtered or blended (§5.4).
    hook_strength: float | None = None
    # Model indices for the cold-open lines, checked in cold_open_spans.
    tease_words: tuple[int | None, int | None] = (None, None)
    payoff_words: tuple[int | None, int | None] = (None, None)
    tease: dict[str, Any] | None = None
    payoff: dict[str, Any] | None = None
    # Prompt v3 only (§5.2i): 0-25 subscores and a hook type. Display and log
    # only, like hook_strength; never sorted, filtered or blended (§5.4).
    subscores: dict[str, int | None] | None = None
    hook_type: str | None = None
    # The model's raw span, kept for audit; start_word/end_word are snapped.
    model_start_word: int | None = None
    model_end_word: int | None = None
    # Boundary decisions a reviewer should see (§6.7); never a filter.
    boundary_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model_start_word is None:
            self.model_start_word = self.start_word
        if self.model_end_word is None:
            self.model_end_word = self.end_word

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
            "duration": self.duration, "hook_strength": self.hook_strength,
            "tease": self.tease, "payoff": self.payoff,
            "model_start_word": self.model_start_word, "model_end_word": self.model_end_word,
            "boundary_notes": self.boundary_notes,
            "subscores": self.subscores, "hook_type": self.hook_type,
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
                    "hook_strength": {"type": "number"},
                    "brief_flags": {"type": "array", "items": {"type": "string"}},
                    "tease_start_word": {"type": "integer"},
                    "tease_end_word": {"type": "integer"},
                    "payoff_start_word": {"type": "integer"},
                    "payoff_end_word": {"type": "integer"},
                },
                "required": ["start_word", "end_word", "hook_text", "why", "topic",
                             "hook_strength", "brief_flags", "tease_start_word",
                             "tease_end_word", "payoff_start_word", "payoff_end_word"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}
"""Sent as the S5 structured-output format (§5.3). Structured outputs require
`additionalProperties: false` on every object and reject numeric and length
constraints, so index ranges, durations, overlap and the 0-1 range of
`hook_strength` stay with post_process."""

PROMPT = """You are selecting standalone short-form clips from a long transcript.

TRANSCRIPT FORMAT
One sentence per line. Each line begins with [n], the global word index of its
first word (indexing starts at 0 and runs across the whole transcript).
Refer to moments only by word index. Never output a timestamp; timestamps are
looked up from alignment data.

TASK
Select up to {count} clips, ordered best first.

FIELDS
- start_word, end_word: inclusive global word indices. Aim for
  {min_words}-{max_words} words (~{min_duration:.0f}-{max_duration:.0f}s at
  this speaker's pace).
- hook_text: at most 8 words, the on-screen opener shown in the first seconds.
  The line that stops a scroll, never a description or summary of the clip.
- why: at most 20 words. What makes it standalone, and the payoff a viewer gets.
- topic: 2-3 lowercase words.
- hook_strength: 0.0-1.0, how strongly the opening earns attention.
- brief_flags: brief rules this clip might touch, else [].
- tease_start_word, tease_end_word: inclusive global word indices of the single
  most gripping line in the clip that raises the question or tension WITHOUT
  giving away the answer. One sentence or phrase of 4-12 words, inside
  [start_word, end_word]. It may be played before the clip as a cold open.
- payoff_start_word, payoff_end_word: inclusive global word indices of the line
  where the tension resolves and the viewer gets the reward. One sentence or
  phrase of 4-12 words, inside [start_word, end_word], not overlapping the tease.

REQUIREMENTS
1. Standalone: a viewer with zero prior context can follow the clip. Reject
   clips that lean on earlier context (pronouns with unresolved antecedents,
   "as I said", callbacks).
2. Hook early: the opening words earn attention within about 3 seconds.
3. Payoff: the clip resolves on its own, with no trailing off mid-thought.
4. No overlap: no two candidates share a word.
5. Spread: distribute candidates across the whole transcript. Late material is
   under-picked by other clippers on this source, so prefer it when quality is
   close.
6. Quantity: return {count} clips only if the source has {count} that meet
   every requirement. Otherwise return fewer; never pad with weak clips.
{brief_block}
Transcript:
{transcript}"""


SUBSCORES = ("hook_score", "engagement_score", "value_score", "shareability_score")
HOOK_TYPES = ("question", "statement", "statistic", "story", "contrast", "none")

RANKING_SCHEMA_V3 = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        **RANKING_SCHEMA["properties"]["candidates"]["items"],
        "properties": {
            **RANKING_SCHEMA["properties"]["candidates"]["items"]["properties"],
            **{name: {"type": "integer"} for name in SUBSCORES},
            "hook_type": {"type": "string", "enum": list(HOOK_TYPES)},
        },
        "required": [*RANKING_SCHEMA["properties"]["candidates"]["items"]["required"],
                     *SUBSCORES, "hook_type"],
    }}},
    "required": ["candidates"],
    "additionalProperties": False,
}
"""Prompt v3's schema (§5.2i): RANKING_SCHEMA plus the four subscores and
hook_type. `enum` is structured-output-safe; the 0-25 range is checked in
parse_candidates, like hook_strength's 0-1."""

# Prompt v3 (§5.2i), an experiment beside the approved PROMPT, which stays the
# default. Ideas borrowed from supoclip (AGPL-3.0) as ideas only: the wording
# here is our own and no text or code was copied.
_V3_FIELDS = """- hook_score, engagement_score, value_score, shareability_score: integers
  0-25, described under SCORES.
- hook_type: one of question, statement, statistic, story, contrast, none.
"""
_V3_GUIDANCE = """
WHAT A CLIP NEEDS
A good clip moves through four beats: a setup, then a tension or a claim, then
a specific detail that makes it concrete, then a payoff. When the strongest
moment is a single line, widen the span to take in the nearest setup before it
and the nearest payoff after it. Stop widening as soon as the topic shifts or
the speaker starts repeating themselves.

PREFER
- claims that cut against what most people believe
- mistakes, and the lesson the speaker took from them
- concrete examples: numbers, names, specifics
- before-and-after contrasts
- results that surprised the speaker
- a complete answer to a question worth asking

AVOID
- introductions, sponsor reads and calls to action
- openings that stay vague about what is coming
- quotes that only make sense with earlier context
- points the speaker has already made
- long, wandering background

SCORES
Each 0-25. They describe the clip; your order alone is the ranking.
- hook_score: 0-6 the opening gives no reason to stay; 7-13 a mildly
  interesting opening; 14-19 a clear question, claim or number within the
  first seconds; 20-25 an opening a scrolling viewer cannot ignore.
- engagement_score: 0-6 flat or hard to follow; 7-13 holds attention in
  places; 14-19 keeps building to the payoff; 20-25 tension or emotion
  throughout.
- value_score: 0-6 nothing to take away; 7-13 a general point; 14-19 a
  specific insight or method; 20-25 something the viewer can use or will
  remember.
- shareability_score: 0-6 nobody would pass it on; 7-13 of niche interest;
  14-19 a viewer would send it to someone they know; 20-25 people will argue
  about it or quote it.
"""
_V3_SIGNALS = """
SIGNALS
{signals}
"""


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


def _ends_with(pattern: re.Pattern, text: str) -> bool:
    return bool(pattern.search(text)) and not TRAILING_OFF.search(text)


def sentence_starts(words: Sequence[Any]) -> list[int]:
    """Indices where a sentence begins. A clip may only start here."""
    return [0] + [i + 1 for i, w in enumerate(words[:-1]) if _ends_with(SENTENCE_END, _word(w))]


def sentence_ends(words: Sequence[Any]) -> list[int]:
    """Indices where a sentence ends. A clip may only end here."""
    ends = [i for i, w in enumerate(words) if _ends_with(SENTENCE_END, _word(w))]
    if not ends or ends[-1] != len(words) - 1:
        ends.append(len(words) - 1)
    return ends


def exclusive_end(lo: int, hi: int, starts: set[int]) -> int:
    """The model's end index, read as exclusive when it opens a sentence.

    17 of 22 real end indices were the first word of the next sentence (§6.7):
    the model sees an index only for each line's first word and gives the next
    line's. Outward snapping then appended that whole sentence.
    """
    return hi - 1 if hi > lo and hi in starts else hi


def snap(candidate: Candidate, starts: list[int], ends: list[int], total: int) -> Candidate:
    """Move a span outward to the nearest sentence boundary (§2.2 item 3).

    Outward, not nearest: snapping inward could cut the very sentence that made
    the moment worth clipping. An end index that opens a sentence is first read
    as exclusive (`exclusive_end`).
    """
    lo = max(0, min(candidate.start_word, total - 1))
    hi = max(0, min(candidate.end_word, total - 1))
    if hi < lo:
        lo, hi = hi, lo
    moved = exclusive_end(lo, hi, set(starts))
    if moved != hi:
        candidate.boundary_notes.append("model end opened the next sentence; ended on the one before")
        hi = moved

    start = max((s for s in starts if s <= lo), default=starts[0])
    end = min((e for e in ends if e >= hi), default=ends[-1])
    candidate.start_word, candidate.end_word = start, end
    return candidate


def _speaker(w: Any) -> Any:
    return w.speaker if hasattr(w, "speaker") else w.get("speaker")


def _time(w: Any, key: str) -> float | None:
    return getattr(w, key) if hasattr(w, key) else w.get(key)


def pause_end(end_word: int, words: Sequence[Any], ends: list[int]) -> int:
    """The first sentence end at or after `end_word` that the speaker pauses on.

    A pause is at least RUN_ON_GAP_S of silence before the next word, a
    speaker change, or the transcript's end. S5 ranks an unlabelled transcript
    (§2.2 item 1), so normally only the gap decides.
    """
    for e in ends:
        if e < end_word:
            continue
        if e + 1 >= len(words):
            return e
        here, after = words[e], words[e + 1]
        a, b = _time(here, "end"), _time(after, "start")
        spk_a, spk_b = _speaker(here), _speaker(after)
        if (a is not None and b is not None and b - a >= RUN_ON_GAP_S - 1e-9) or \
                (spk_a is not None and spk_b is not None and spk_a != spk_b):
            return e
    return len(words) - 1


def extend_run_on(candidate: Candidate, words: Sequence[Any], ends: list[int],
                  max_duration_s: float) -> Candidate:
    """Extend a timed clip whose speaker runs straight on to the next pause.

    The extension must stay within `max_duration_s` and RUN_ON_MAX_SENTENCES;
    if it cannot, the end is kept and the clip carries a note for Review.
    """
    target = pause_end(candidate.end_word, words, ends)
    if target == candidate.end_word:
        return candidate
    added = sum(1 for e in ends if candidate.end_word < e <= target)
    trial = resolve_times(Candidate(0, candidate.start_word, target), words)
    if added > RUN_ON_MAX_SENTENCES:
        candidate.boundary_notes.append(
            f"speaker runs on past the end; the next pause is {added} sentences later")
        return candidate
    if trial.end is not None and trial.start is not None and trial.duration <= max_duration_s:
        candidate.boundary_notes.append(
            f"speaker ran on; extended {target - candidate.end_word} words to the next pause")
        candidate.end_word = target
        return resolve_times(candidate, words)
    candidate.boundary_notes.append(
        "speaker runs on past the end; the next pause would exceed the duration range")
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

    # An empty array is legal: the prompt says to return fewer rather than pad,
    # and the >=5-survivor gate decides (§5.4).
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
            hook_strength=_unit(item.get("hook_strength")),
            tease_words=(_index(item.get("tease_start_word")), _index(item.get("tease_end_word"))),
            payoff_words=(_index(item.get("payoff_start_word")), _index(item.get("payoff_end_word"))),
            subscores=({name: _subscore(item.get(name)) for name in SUBSCORES}
                       if any(name in item for name in SUBSCORES) else None),
            hook_type=item.get("hook_type") if item.get("hook_type") in HOOK_TYPES else None,
        ))
    return out


def _unit(value: Any) -> float | None:
    """A 0-1 self-rating, or None when it is missing or out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _subscore(value: Any) -> int | None:
    """A 0-25 prompt-v3 subscore, or None when it is missing or out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return int(value) if value == int(value) and 0 <= value <= 25 else None


def _index(value: Any) -> int | None:
    """A cold-open word index. A bad one disables only that mode, never the clip."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def phrase_starts(words: Sequence[Any]) -> list[int]:
    """Indices where a sentence or phrase begins."""
    return [0] + [i + 1 for i, w in enumerate(words[:-1]) if _ends_with(PHRASE_END, _word(w))]


def phrase_ends(words: Sequence[Any]) -> list[int]:
    """Indices where a sentence or phrase ends."""
    ends = [i for i, w in enumerate(words) if _ends_with(PHRASE_END, _word(w))]
    if not ends or ends[-1] != len(words) - 1:
        ends.append(len(words) - 1)
    return ends


def span_problem(span: dict, start_word: int, end_word: int) -> str | None:
    """Why a checked cold-open span cannot play in a clip spanning these words.

    Also run at Review and render time, because a trim can move the clip
    boundary past a line that was valid when S5 checked it.
    """
    if not span:
        return "no line"
    if span.get("problem"):
        return span["problem"]
    if not start_word <= span["start_word"] <= span["end_word"] <= end_word:
        return "outside the clip"
    return None


def cold_open_modes(clip: dict) -> list[str]:
    """The cold-open modes this clip may offer: its line passed S5's checks and
    still lies inside the clip's current (possibly trimmed) word span."""
    return [m for m in COLD_OPEN_MODES
            if span_problem(clip.get(m), clip["start_word"], clip["end_word"]) is None]


def cold_open_choice(clip: dict) -> tuple[str, dict] | None:
    """(mode, span) for a clip whose cold open is on; None when it is off.

    Raises ValueError for an unknown mode or a line that cannot play, so the
    renderer and S11 never trust a stored mode the checks would refuse.
    """
    mode = clip.get("cold_open") or "off"
    if mode == "off":
        return None
    if mode not in COLD_OPEN_MODES:
        raise ValueError(f"unknown cold open mode {mode!r}")
    problem = span_problem(clip.get(mode), clip["start_word"], clip["end_word"])
    if problem:
        raise ValueError(f"cold open {mode} line cannot play: {problem}")
    return mode, clip[mode]


def cold_open_spans(candidate: Candidate, words: Sequence[Any],
                    starts: list[int], ends: list[int],
                    sentence_start_set: set[int] | None = None) -> Candidate:
    """Check, snap and time the tease and payoff lines of a snapped, timed clip.

    An invalid line only disables that cold-open mode for this clip; it never
    drops the candidate or gates the run. When the two overlap, the payoff is
    disabled: tease is the mode the user prefers [I].
    """
    def check(pair: tuple[int | None, int | None]) -> dict[str, Any]:
        lo, hi = pair
        span: dict[str, Any] = {"model_start_word": lo, "model_end_word": hi, "problem": None}
        if lo is None or hi is None or lo > hi:
            span["problem"] = "missing or reversed word indices"
            return span
        if not candidate.start_word <= lo <= hi <= candidate.end_word:
            span["problem"] = "outside the clip"
            return span
        # Same exclusive-end reading as the clip (§6.7).
        hi = exclusive_end(lo, hi, sentence_start_set or set())
        # Outward, like clip spans. Clip ends are sentence ends, which are also
        # phrase ends, so the snapped line stays inside the clip.
        a = max(s for s in starts if s <= lo)
        b = min(e for e in ends if e >= hi)
        timed = resolve_times(Candidate(0, a, b), words)
        span.update(start_word=a, end_word=b, start=timed.start, end=timed.end,
                    duration=timed.duration,
                    text=" ".join(_word(words[i]) for i in range(a, b + 1)))
        if timed.start is None or timed.end is None:
            span["problem"] = "no aligned timing"
        elif not COLD_OPEN_MIN_S <= timed.duration <= COLD_OPEN_MAX_S:
            span["problem"] = (f"{timed.duration:.2f}s is outside "
                               f"{COLD_OPEN_MIN_S:g}-{COLD_OPEN_MAX_S:g}s")
        return span

    tease, payoff = check(candidate.tease_words), check(candidate.payoff_words)
    if ("start_word" in tease and "start_word" in payoff
            and tease["start_word"] <= payoff["end_word"]
            and payoff["start_word"] <= tease["end_word"]):
        payoff["problem"] = payoff["problem"] or "overlaps the tease"
    candidate.tease, candidate.payoff = tease, payoff
    return candidate


def _word(w: Any) -> str:
    return w.word if hasattr(w, "word") else w["word"]


def post_process(
    candidates: list[Candidate],
    words: Sequence[Any],
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
    max_duration_s: float = DEFAULT_MAX_DURATION_S,
) -> tuple[list[Candidate], dict[str, int]]:
    """§5.4 steps 2-4. Returns survivors and a count of what each filter dropped."""
    starts, ends = sentence_starts(words), sentence_ends(words)
    start_set = set(starts)
    pstarts, pends = phrase_starts(words), phrase_ends(words)
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
        extend_run_on(c, words, ends, max_duration_s)
        staged.append(cold_open_spans(c, words, pstarts, pends, start_set))

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
    template: str = PROMPT,
    signals: str = "",
) -> tuple[list[Candidate], dict[str, Any]]:
    """One ranking pass with a single retry, then a hard stop (§5.4 item 1).

    `template` is PROMPT or PROMPT_V3; only v3 has a place for `signals`.
    """
    transcript = build_transcript(words, with_speakers=with_speakers)
    prompt = template.format(
        count=count, transcript=transcript,
        min_duration=min_duration_s, max_duration=max_duration_s,
        min_words=int(min_duration_s * 2.5), max_words=int(max_duration_s * 2.5),
        brief_block=brief_block, signals=signals or "(none found)",
    )

    last_error = ""
    for attempt in (1, 2):
        raw = llm_fn(prompt if attempt == 1 else prompt + _RETRY_SUFFIX.format(error=last_error))
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


PROMPT_V3 = (PROMPT
             .replace("- brief_flags: brief rules this clip might touch, else [].\n",
                      "- brief_flags: brief rules this clip might touch, else [].\n" + _V3_FIELDS)
             .replace("{brief_block}\nTranscript:", _V3_GUIDANCE + "{brief_block}" + _V3_SIGNALS + "\nTranscript:"))
assert PROMPT_V3.count(_V3_FIELDS) == 1 and PROMPT_V3.count("SIGNALS") == 1
PROMPT_VERSIONS = {"current": (PROMPT, RANKING_SCHEMA), "v3": (PROMPT_V3, RANKING_SCHEMA_V3)}
V3_DEFAULT_COUNT = 5

_RETRY_SUFFIX = ("\n\nYour previous reply could not be used ({error}). "
                 "Reply with just the JSON object described above.")
