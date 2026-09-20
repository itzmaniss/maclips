"""Audio loading, speaker assignment and the benchmark's swap detection."""
from __future__ import annotations

import subprocess
import wave

import numpy as np
import pytest

from maclips import config
from maclips.audio import SAMPLE_RATE, AudioError, load_wav
from maclips.bench import StageMetrics
from maclips.diarize import DiarizationResult, Turn, assign_speakers, sample_labels
from maclips.transcribe import Word


@pytest.fixture
def wav_16k(tmp_path, tiny_av):
    """A real 16 kHz mono WAV, produced the way S2 produces one."""
    out = tmp_path / "audio.wav"
    from maclips.ffmpeg import extract_audio

    extract_audio(tiny_av, out, sample_rate=SAMPLE_RATE)
    return out


def test_load_wav_returns_float32_in_range(wav_16k):
    data = load_wav(wav_16k)
    assert data.dtype == np.float32
    assert data.ndim == 1
    assert -1.0 <= float(data.min()) and float(data.max()) <= 1.0
    assert len(data) == pytest.approx(3 * SAMPLE_RATE, rel=0.1)


def test_s2_output_needs_no_decoder(wav_16k):
    """The point of S2's format: the standard library can read it (§2.4)."""
    with wave.open(str(wav_16k), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SAMPLE_RATE)


def test_load_wav_rejects_stereo(tmp_path, tiny_av):
    stereo = tmp_path / "stereo.wav"
    subprocess.run(
        [str(config.FFMPEG), "-y", "-loglevel", "error", "-i", str(tiny_av),
         "-ac", "2", "-ar", "16000", "-acodec", "pcm_s16le", str(stereo)],
        check=True, capture_output=True,
    )
    with pytest.raises(AudioError, match="expected mono"):
        load_wav(stereo)


def test_load_wav_rejects_wrong_sample_rate(tmp_path, tiny_av):
    wrong = tmp_path / "44k.wav"
    subprocess.run(
        [str(config.FFMPEG), "-y", "-loglevel", "error", "-i", str(tiny_av),
         "-ac", "1", "-ar", "44100", "-acodec", "pcm_s16le", str(wrong)],
        check=True, capture_output=True,
    )
    with pytest.raises(AudioError, match="expected 16000 Hz"):
        load_wav(wrong)


# --------------------------------------------------------------------------- #
# Speaker assignment
# --------------------------------------------------------------------------- #

def _result(*spans):
    turns = [Turn(s, e, spk) for s, e, spk in spans]
    return DiarizationResult(turns=turns, speakers=sorted({t.speaker for t in turns}))


def test_word_goes_to_the_speaker_holding_most_of_it():
    """A word straddling a boundary belongs to whoever holds more of it.

    Only unambiguous splits are asserted. An exact 50/50 straddle is decided by
    floating-point comparison of two subtractions (1.8->2.0 computes as
    0.19999999999999996, 2.0->2.2 as 0.20000000000000018), so which side wins a
    "tie" is an artifact, not behaviour worth depending on.
    """
    result = _result((0.0, 2.0, "A"), (2.0, 4.0, "B"))

    mostly_a = Word("early", 1.5, 2.05)          # 0.50 s in A, 0.05 s in B
    assign_speakers([mostly_a], result)
    assert mostly_a.speaker == "A"

    mostly_b = Word("late", 1.95, 2.5)           # 0.05 s in A, 0.50 s in B
    assign_speakers([mostly_b], result)
    assert mostly_b.speaker == "B"


def test_long_earlier_turn_is_still_found():
    """The bisect lookback must not skip a long turn that began much earlier.

    A monologue turn can start minutes before the word it contains. Seeking
    only to turns starting near the word would miss it entirely.
    """
    result = _result((0.0, 600.0, "A"), (601.0, 602.0, "B"))
    word = Word("late-in-monologue", 500.0, 500.4)
    assign_speakers([word], result)
    assert word.speaker == "A"


def test_word_outside_every_turn_keeps_no_speaker():
    """Silence is not assigned to a neighbour by proximity."""
    word = Word("orphan", 10.0, 10.5)
    assign_speakers([word], _result((0.0, 2.0, "A")))
    assert word.speaker is None


def test_unaligned_words_are_skipped():
    word = Word("unplaced")
    assign_speakers([word], _result((0.0, 5.0, "A")))
    assert word.speaker is None


def test_no_turns_leaves_words_untouched():
    words = [Word("a", 0.0, 1.0)]
    assert assign_speakers(words, DiarizationResult())[0].speaker is None


def test_sample_labels_returns_verifiable_excerpts():
    words = [
        Word(f"w{i}", i * 0.3, i * 0.3 + 0.25, speaker="SPEAKER_00" if i < 30 else "SPEAKER_01")
        for i in range(60)
    ]
    samples = sample_labels(words, count=3)
    assert 1 <= len(samples) <= 3
    for s in samples:
        assert s["speaker"].startswith("SPEAKER_")
        assert s["end"] >= s["start"]
        assert s["text"]


def test_sample_labels_empty_when_nothing_labelled():
    assert sample_labels([Word("a", 0.0, 1.0)]) == []


# --------------------------------------------------------------------------- #
# Benchmark honesty: swap growth invalidates a timing
# --------------------------------------------------------------------------- #

def test_swap_growth_is_flagged():
    m = StageMetrics(name="S3", swap_before_mb=100.0, swap_after_mb=900.0)
    assert m.swap_delta_mb == 800.0
    assert m.swapped, "a stage that paged must be flagged, not silently reported"


def test_small_swap_jitter_is_not_flagged():
    m = StageMetrics(name="S3", swap_before_mb=100.0, swap_after_mb=110.0)
    assert not m.swapped


def _gated_error(message: str = "403"):
    """Build a real GatedRepoError. It requires an httpx.Response, not None."""
    import httpx
    from huggingface_hub.errors import GatedRepoError

    response = httpx.Response(403, request=httpx.Request("GET", "https://huggingface.co/x"))
    return GatedRepoError(message, response=response)


# --------------------------------------------------------------------------- #
# The gated-access gate
# --------------------------------------------------------------------------- #

def test_access_gate_checks_files_not_just_metadata(monkeypatch):
    """A `gated:auto` repo serves metadata to anyone; only files prove access.

    Build step 2's first run passed a model_info()-only gate and then died
    eight minutes later on a 403 for config.yaml. The gate must fetch a file.
    """
    from maclips import diarize as d

    calls = []

    def fake_download(repo, filename, **kw):
        calls.append((repo, filename))
        raise _gated_error()

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    # model_info deliberately succeeds, as it really does for these repos
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **k: type("I", (), {"sha": "x"})())

    with pytest.raises(d.DiarizationAccessError) as excinfo:
        d.check_access()

    assert calls, "the gate must attempt a real file download"
    assert calls[0][1] == "config.yaml", "should fetch the file pyannote loads first"
    message = str(excinfo.value)
    assert "gated model not accepted or token lacks access" in message
    assert "accept the user conditions" in message, "must say how to fix it"
    assert "gated:auto" in message, "must explain why metadata access misleads"


def test_access_gate_passes_when_the_file_is_fetchable(monkeypatch):
    from maclips import diarize as d

    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "/tmp/config.yaml")
    monkeypatch.setattr("huggingface_hub.model_info", lambda *a, **k: type("I", (), {"sha": "abc123"})())
    assert d.check_access() == "abc123"


def test_access_gate_never_echoes_a_token(monkeypatch):
    from maclips import diarize as d

    def fake_download(repo, filename, **kw):
        raise _gated_error("403 with hf_abcdefghijklmnopqrstuvwxyz012345 in the body")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    with pytest.raises(d.DiarizationAccessError) as excinfo:
        d.check_access()
    assert "hf_abcdefghijklmnop" not in str(excinfo.value)
