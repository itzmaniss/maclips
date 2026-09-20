"""S3: transcribe and align with whispermlx.

Two models, two runtimes, loaded and released one at a time:

- **Transcription** runs Whisper on MLX (Apple GPU). CTranslate2, which stock
  WhisperX uses, has no Metal backend and falls back to CPU.
- **Alignment** runs wav2vec2 on torch/MPS and is a separate pass. It is what
  produces per-word timings; the Whisper pass alone gives segment text.

Audio is always passed as an in-memory array. `whispermlx.load_audio` is never
used — it shells out to a bare `"ffmpeg"` from PATH, which is broken here
(§2.3), and it would also mean decoding the same file twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .resources import release

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
TRANSCRIBE_DEVICE = "mps"   # only alignment actually uses torch; MLX picks the GPU itself
ALIGN_DEVICE = "mps"


@dataclass
class Word:
    """One word from the aligner.

    `score` is the mean per-character alignment confidence, and its **absence
    is meaningful**: whispermlx sets `start`/`end`/`score` only for characters
    it could actually align, then interpolates missing `start`/`end` across the
    sentence — without ever backfilling `score`. So a word carrying timings but
    no score has *guessed* timings.
    """

    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None

    @property
    def aligned(self) -> bool:
        """Has timings — but they may be interpolated. See `acoustic`."""
        return self.start is not None and self.end is not None

    @property
    def acoustic(self) -> bool:
        """Timings came from the audio, not from interpolation."""
        return self.aligned and self.score is not None

    @property
    def interpolated(self) -> bool:
        """Timings were guessed between neighbours; the audio did not place it."""
        return self.aligned and self.score is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word, "start": self.start, "end": self.end,
            "score": self.score, "speaker": self.speaker,
        }


@dataclass
class Transcript:
    """S3's output: words plus what the gates need to judge it."""

    words: list[Word] = field(default_factory=list)
    language: str = ""
    segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of words carrying timings, interpolated ones included.

        Kept for reporting, **not** as a gate. Build step 2 measured 100.00% on
        a clean two-hour source precisely because interpolation fills every
        gap: the number cannot discriminate, so gating on it gates on nothing.
        `weak_fraction` is the gate.
        """
        if not self.words:
            return 0.0
        return sum(1 for w in self.words if w.aligned) / len(self.words)

    def weak_fraction(self, min_score: float) -> float:
        """Fraction of words that are interpolated, unaligned, or low-scoring.

        This is what S3 gates on. All three states mean "the audio did not
        confidently place this word", and all three produce the same downstream
        harm: a clip boundary snapped to a timestamp nobody measured.
        """
        if not self.words:
            return 1.0
        weak = sum(
            1 for w in self.words
            if not w.aligned or w.interpolated or (w.score is not None and w.score < min_score)
        )
        return weak / len(self.words)

    def score_stats(self) -> dict[str, Any]:
        """Distribution of alignment confidence, for choosing a threshold."""
        scores = sorted(w.score for w in self.words if w.score is not None)
        total = len(self.words)
        stats: dict[str, Any] = {
            "words": total,
            "acoustic": sum(1 for w in self.words if w.acoustic),
            "interpolated": sum(1 for w in self.words if w.interpolated),
            "unaligned": sum(1 for w in self.words if not w.aligned),
        }
        if scores:
            def pct(p: float) -> float:
                return scores[min(int(p * len(scores)), len(scores) - 1)]

            stats["percentiles"] = {
                f"p{int(p * 100)}": round(pct(p), 3)
                for p in (0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.90)
            }
            stats["min"] = round(scores[0], 3)
            stats["mean"] = round(sum(scores) / len(scores), 3)
        return stats

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "word_count": len(self.words),
            "coverage": self.coverage,
            "words": [w.as_dict() for w in self.words],
        }


def transcribe(
    waveform: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    batch_size: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Whisper on MLX. Returns (segments, detected_language).

    The model is released before returning so alignment never loads while
    Whisper is still resident.
    """
    from whispermlx import load_model

    model = load_model(model_name, device=TRANSCRIBE_DEVICE, language=language)
    try:
        result = model.transcribe(waveform, batch_size=batch_size, language=language)
        segments = list(result.get("segments", []))
        detected = str(result.get("language") or language or "")
    finally:
        release(model)
    return segments, detected


def align(
    segments: list[dict[str, Any]],
    waveform: np.ndarray,
    language: str,
    device: str = ALIGN_DEVICE,
) -> list[Word]:
    """wav2vec2 forced alignment → per-word timings.

    Sentence punctuation is preserved: the words come back as the transcript
    spelled them, which is what lets S5 snap spans to sentence boundaries
    (§2.2 item 3). Note that whispermlx tokenizes with a bare `text.split(" ")`
    and does no punctuation handling of its own.
    """
    from whispermlx import align as _align
    from whispermlx import load_align_model

    model, metadata = load_align_model(language_code=language, device=device)
    try:
        result = _align(
            transcript=segments,
            model=model,
            align_model_metadata=metadata,
            audio=waveform,
            device=device,
        )
    finally:
        release(model)

    words: list[Word] = []
    for raw in result.get("word_segments", []):
        text = str(raw.get("word", "")).strip()
        if not text:
            continue
        start, end = raw.get("start"), raw.get("end")
        words.append(
            Word(
                word=text,
                start=float(start) if start is not None else None,
                end=float(end) if end is not None else None,
                score=float(raw["score"]) if raw.get("score") is not None else None,
            )
        )
    return words


def run(
    waveform: np.ndarray,
    model_name: str = DEFAULT_MODEL,
    language: str | None = None,
    on_stage: Any = None,
) -> Transcript:
    """Transcribe then align. `on_stage(name)` marks the split for benchmarking."""
    if on_stage:
        on_stage("transcribe")
    segments, detected = transcribe(waveform, model_name=model_name, language=language)
    if on_stage:
        on_stage("align")
    words = align(segments, waveform, detected or "en")
    return Transcript(words=words, language=detected, segments=segments)
