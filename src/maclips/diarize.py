"""S4: speaker diarization with pyannote community-1.

**Authentication** comes from the Hugging Face cached token (`hf auth login`).
No token is read from `.env`, passed explicitly, or printed. `huggingface_hub`
resolves the cache itself when `token=None`, which is what whispermlx's wrapper
passes through.

**Audio** is always the in-memory array S2 produced (§2.4). Verified in build
step 2: `whispermlx.diarize.DiarizationPipeline` wraps whatever it is given
into `{"waveform": ..., "sample_rate": ...}` and never hands pyannote a path,
so its wrapper is safe to use and no bypass is needed. pyannote itself prints
the same advice when torchcodec is unloadable.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import numpy as np

from .audio import SAMPLE_RATE
from .resources import release
from .transcribe import Word

COMMUNITY_MODEL = "pyannote/speaker-diarization-community-1"
DEVICE = "mps"

WINDOW_PAD_S = 10.0
"""Context either side of a candidate span. Diarization needs a little audio to
separate voices; a bare 60-second cut starting mid-sentence gives it less to
work with, and the padding is discarded after labelling."""

MIN_SPEAKING_SHARE = 0.05
"""A label holding under 5% of a window's speaking time is not treated as a
speaker for gating. Diarization routinely emits a sliver label for a
backchannel or a breath, and counting raw labels would fail the gate on a
clean two-person window."""


class DiarizationAccessError(RuntimeError):
    """The gated model is not reachable with the cached credentials."""


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[Turn] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)

    def speaking_time(self) -> dict[str, float]:
        """Seconds of speech per label."""
        totals: dict[str, float] = {}
        for turn in self.turns:
            totals[turn.speaker] = totals.get(turn.speaker, 0.0) + (turn.end - turn.start)
        return totals

    def significant_speakers(self, min_share: float = MIN_SPEAKING_SHARE) -> list[str]:
        """Labels holding at least `min_share` of total speaking time.

        This, not `len(speakers)`, is what the count gate compares against:
        raw label count treats a half-second sliver as a person.
        """
        totals = self.speaking_time()
        total = sum(totals.values())
        if total <= 0:
            return []
        return sorted(s for s, secs in totals.items() if secs / total >= min_share)

    def shares(self) -> dict[str, float]:
        totals = self.speaking_time()
        total = sum(totals.values()) or 1.0
        return {s: secs / total for s, secs in totals.items()}

    def as_dict(self) -> dict[str, Any]:
        return {
            "speakers": self.speakers,
            "speaker_count": len(self.speakers),
            "significant_speakers": self.significant_speakers(),
            "shares": {s: round(v, 4) for s, v in self.shares().items()},
            "turn_count": len(self.turns),
            "turns": [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in self.turns],
        }


def check_access(model: str = COMMUNITY_MODEL) -> str:
    """Hard gate: the gated repo's **files** must be fetchable before any work starts.

    **`model_info()` is not sufficient and must not be used alone.** These repos
    are `gated: auto`, which makes their *metadata* world-readable while file
    downloads still require accepted terms. Build step 2 hit exactly that:
    `model_info()` returned happily, the gate passed, and the run then died
    eight minutes later inside `Pipeline.from_pretrained` with a 403 on
    `config.yaml`. So the gate fetches a real file — the same `config.yaml`
    pyannote loads first — which is the only thing that proves authorization.

    Nothing about the token is included in the message.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    hint = (
        f"gated model not accepted or token lacks access: {model}\n"
        f"  Visit https://huggingface.co/{model} and accept the user conditions,\n"
        "  then `hf auth login`. Metadata being readable is not enough: these\n"
        "  repos are gated:auto, so model_info() succeeds while files 403."
    )

    # Already cached? Then the gate has nothing to add and must not require a
    # network round trip — PLAN.md §8 wants the weights usable offline, and a
    # gate that only exists to catch a misconfiguration should not be the
    # thing that breaks an offline run.
    try:
        hf_hub_download(model, "config.yaml", local_files_only=True)
        return ""
    except Exception:  # noqa: BLE001 - not cached; fall through to the live check
        pass

    try:
        hf_hub_download(model, "config.yaml")
    except (GatedRepoError, RepositoryNotFoundError) as exc:
        raise DiarizationAccessError(hint) from exc
    except HfHubHTTPError as exc:
        raise DiarizationAccessError(f"{hint}\n  ({exc.__class__.__name__})") from exc

    from huggingface_hub import model_info

    try:
        return getattr(model_info(model), "sha", "") or ""
    except Exception:  # noqa: BLE001 - the access check already passed
        return ""


def load_pipeline(device: str = DEVICE, model: str = COMMUNITY_MODEL):
    """Load the diarization pipeline once, for reuse across many windows.

    Loading costs seconds and several GB; doing it per window would dominate
    the cost of window-only diarization and defeat the point of the change.
    """
    from whispermlx.diarize import DiarizationPipeline

    return DiarizationPipeline(model_name=model, token=None, device=device)


def _run_pipeline(
    pipeline,
    waveform: np.ndarray,
    max_speakers: int | None,
    min_speakers: int | None,
    offset: float = 0.0,
) -> DiarizationResult:
    """One pipeline call. `offset` shifts turns back onto the source timeline."""
    frame = pipeline(waveform, min_speakers=min_speakers, max_speakers=max_speakers)
    turns = [
        Turn(start=float(r.start) + offset, end=float(r.end) + offset, speaker=str(r.speaker))
        for r in frame.itertuples()
    ]
    return DiarizationResult(turns=turns, speakers=sorted({t.speaker for t in turns}))


def diarize(
    waveform: np.ndarray,
    max_speakers: int | None = None,
    min_speakers: int | None = None,
    device: str = DEVICE,
    model: str = COMMUNITY_MODEL,
    pipeline=None,
) -> DiarizationResult:
    """Diarize a whole in-memory waveform. Never takes a path.

    **`num_speakers` is deliberately not accepted.** An exact count is wrong
    for a window, which may legitimately contain one speaker, and forcing a
    count makes diarization invent a second voice. The expected count is passed
    as `max_speakers` — a ceiling, not a target.
    """
    owned = pipeline is None
    pipeline = pipeline or load_pipeline(device=device, model=model)
    try:
        return _run_pipeline(pipeline, waveform, max_speakers, min_speakers)
    finally:
        if owned:
            release(pipeline)


@dataclass
class Window:
    """One candidate span, diarized independently."""

    index: int
    start: float
    end: float
    result: DiarizationResult

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "start": self.start, "end": self.end, **self.result.as_dict()}


def diarize_windows(
    waveform: np.ndarray,
    spans: list[tuple[float, float]],
    max_speakers: int | None = None,
    min_speakers: int | None = None,
    pad_s: float = WINDOW_PAD_S,
    sample_rate: int = SAMPLE_RATE,
    device: str = DEVICE,
    model: str = COMMUNITY_MODEL,
    on_window: Any = None,
) -> list[Window]:
    """Diarize only the candidate spans, each padded by `pad_s` either side.

    **Speaker labels are per-window and carry no identity across windows.**
    pyannote assigns `SPEAKER_00`, `SPEAKER_01` ... independently per call, so
    `SPEAKER_00` in window 3 is not the person who was `SPEAKER_00` in window
    1. Within a clip that is all S7 needs — it maps labels to faces inside that
    clip. Anything wanting "the same person across the source" must use
    full-source diarization instead (`--full-diarization`).

    The pipeline is loaded once and reused for every window.
    """
    if not spans:
        return []

    pipeline = load_pipeline(device=device, model=model)
    windows: list[Window] = []
    try:
        duration = len(waveform) / sample_rate
        for i, (start, end) in enumerate(spans):
            lo = max(0.0, start - pad_s)
            hi = min(duration, end + pad_s)
            chunk = waveform[int(lo * sample_rate) : int(hi * sample_rate)]
            if len(chunk) < sample_rate:  # under a second is not diarizable
                windows.append(Window(i, start, end, DiarizationResult()))
                continue
            result = _run_pipeline(pipeline, chunk, max_speakers, min_speakers, offset=lo)
            windows.append(Window(i, start, end, result))
            if on_window is not None:
                on_window(i, len(spans), result)
    finally:
        release(pipeline)
    return windows


def assign_speakers(words: list[Word], result: DiarizationResult) -> list[Word]:
    """Label each word with the speaker whose turn overlaps it most.

    Overlap rather than midpoint containment: a word straddling a turn boundary
    should go to whoever holds more of it, and a word inside no turn at all
    keeps `speaker=None` rather than being forced to a neighbour.
    """
    if not result.turns:
        return words
    turns = sorted(result.turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    # Longest turn, so we know how far back a turn could still reach forward
    # and overlap this word. Without it, a scan must start from turn 0 every
    # time: ~20k words x ~1k turns on a 2-hour source is 20M comparisons.
    longest = max(t.end - t.start for t in turns)

    for word in words:
        if not word.aligned:
            continue
        lo = bisect_left(starts, word.start - longest)
        best_speaker, best_overlap = None, 0.0
        for turn in islice(turns, lo, None):
            if turn.start > word.end:
                break  # sorted by start; nothing later can overlap
            overlap = min(word.end, turn.end) - max(word.start, turn.start)
            if overlap > best_overlap:
                best_overlap, best_speaker = overlap, turn.speaker
        word.speaker = best_speaker
    return words


def sample_labels(words: list[Word], count: int = 3) -> list[dict[str, Any]]:
    """Labelled excerpts with timestamps, for verifying by ear.

    **One per distinct speaker first.** Sampling by position instead would
    happily return three excerpts from the same speaker, which cannot answer
    the question these samples exist for: whether the speaker count is right.
    A podcast diarized as three voices when two people are talking is the
    failure worth catching, and it is only visible if every label is shown.
    """
    labelled = [w for w in words if w.speaker and w.aligned]
    if not labelled:
        return []

    by_speaker: dict[str, list[Word]] = {}
    for w in labelled:
        by_speaker.setdefault(w.speaker, []).append(w)

    def excerpt(anchor: Word, pool: list[Word]) -> dict[str, Any] | None:
        window = [w for w in pool if anchor.start <= w.start < anchor.start + 6.0][:18]
        if not window:
            return None
        return {
            "speaker": anchor.speaker,
            "start": window[0].start,
            "end": window[-1].end,
            "word_count": len(pool),
            "text": " ".join(w.word for w in window),
        }

    samples: list[dict[str, Any]] = []
    # Rarest speaker first: a spurious third label has the fewest words, and is
    # the one most worth hearing.
    for speaker in sorted(by_speaker, key=lambda s: len(by_speaker[s])):
        pool = by_speaker[speaker]
        sample = excerpt(pool[len(pool) // 2], pool)
        if sample:
            samples.append(sample)

    # Only once every speaker is represented, add more for breadth.
    if len(samples) < count:
        step = max(1, len(labelled) // (count + 1))
        for i in range(count - len(samples)):
            anchor = labelled[min(step * (i + 1), len(labelled) - 1)]
            sample = excerpt(anchor, by_speaker[anchor.speaker])
            if sample:
                samples.append(sample)
    return samples
