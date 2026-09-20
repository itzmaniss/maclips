"""Load the S2 WAV into memory, once.

**Why this module exists rather than `whispermlx.load_audio`:** that helper
hardcodes a bare `"ffmpeg"` PATH lookup, and the PATH ffmpeg on this machine is
a broken slim build (§2.3). It would fail, and confusingly — inside a library
call rather than at our seam.

S2 already produced exactly the format we need, so no decoder is required at
all: a 16 kHz mono 16-bit PCM WAV is read with the standard library.

One array is loaded and shared by S3 and S4. A 2-hour source is ~450 MB as
float32, so a second copy is not a rounding error.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


class AudioError(RuntimeError):
    """The WAV was not the mono 16 kHz PCM that S2 is supposed to produce."""


def load_wav(path: Path, expect_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a float32 mono waveform in [-1, 1].

    Strict about format instead of coercing: anything other than what S2 emits
    means S2 changed or the file is not ours, and silently resampling would
    hide that.
    """
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if channels != 1:
        raise AudioError(f"{path}: expected mono, got {channels} channels")
    if width != 2:
        raise AudioError(f"{path}: expected 16-bit PCM, got {width * 8}-bit")
    if rate != expect_rate:
        raise AudioError(f"{path}: expected {expect_rate} Hz, got {rate} Hz")

    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def duration_s(waveform: np.ndarray, rate: int = SAMPLE_RATE) -> float:
    return float(len(waveform)) / rate
