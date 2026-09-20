"""Window-only diarization: pipeline reuse, timeline offsets, label scoping."""
from __future__ import annotations

import numpy as np
import pytest

from maclips import diarize as d
from maclips.audio import SAMPLE_RATE


class FakePipeline:
    """Stands in for DiarizationPipeline; records how often it was constructed."""

    constructed = 0

    def __init__(self, *a, **kw):
        FakePipeline.constructed += 1
        self.calls = []

    def __call__(self, waveform, min_speakers=None, max_speakers=None, **kw):
        import pandas as pd

        self.calls.append({"samples": len(waveform), "max_speakers": max_speakers,
                           "min_speakers": min_speakers})
        secs = len(waveform) / SAMPLE_RATE
        # two speakers splitting the window
        return pd.DataFrame([
            {"start": 0.0, "end": secs / 2, "speaker": "SPEAKER_00"},
            {"start": secs / 2, "end": secs, "speaker": "SPEAKER_01"},
        ])


@pytest.fixture
def fake_pipeline(monkeypatch):
    FakePipeline.constructed = 0
    holder = {}

    def make(device=d.DEVICE, model=d.COMMUNITY_MODEL):
        holder["pipeline"] = FakePipeline()
        return holder["pipeline"]

    monkeypatch.setattr(d, "load_pipeline", make)
    return holder


@pytest.fixture
def waveform():
    return np.zeros(300 * SAMPLE_RATE, dtype=np.float32)   # 300 s of silence


def test_pipeline_is_loaded_once_for_all_windows(fake_pipeline, waveform):
    """Loading per window would dominate the cost and defeat the restructure."""
    spans = [(10.0, 40.0), (80.0, 110.0), (150.0, 180.0)]
    windows = d.diarize_windows(waveform, spans)
    assert len(windows) == 3
    assert FakePipeline.constructed == 1
    assert len(fake_pipeline["pipeline"].calls) == 3


def test_windows_are_padded_on_both_sides(fake_pipeline, waveform):
    d.diarize_windows(waveform, [(100.0, 130.0)], pad_s=10.0)
    samples = fake_pipeline["pipeline"].calls[0]["samples"]
    assert samples == pytest.approx(50 * SAMPLE_RATE, rel=0.01), "30s span + 10s each side"


def test_padding_is_clamped_at_the_source_edges(fake_pipeline, waveform):
    d.diarize_windows(waveform, [(0.0, 20.0)], pad_s=10.0)
    samples = fake_pipeline["pipeline"].calls[0]["samples"]
    assert samples == pytest.approx(30 * SAMPLE_RATE, rel=0.01), "no negative start"


def test_turns_are_mapped_back_onto_the_source_timeline(fake_pipeline, waveform):
    """A window's turns must be absolute, or word assignment lands in the wrong place."""
    windows = d.diarize_windows(waveform, [(100.0, 130.0)], pad_s=10.0)
    turns = windows[0].result.turns
    assert turns[0].start == pytest.approx(90.0), "window began at 100-10"
    assert turns[-1].end == pytest.approx(140.0)


def test_expected_count_is_passed_as_a_ceiling_not_a_target(fake_pipeline, waveform):
    """num_speakers would force a second voice into a single-speaker window."""
    d.diarize_windows(waveform, [(10.0, 40.0)], max_speakers=2)
    call = fake_pipeline["pipeline"].calls[0]
    assert call["max_speakers"] == 2
    assert "num_speakers" not in call


def test_diarize_rejects_an_exact_speaker_count():
    import inspect

    assert "num_speakers" not in inspect.signature(d.diarize).parameters
    assert "num_speakers" not in inspect.signature(d.diarize_windows).parameters


def test_windows_shorter_than_a_second_are_skipped(fake_pipeline, waveform):
    windows = d.diarize_windows(waveform, [(10.0, 10.2)], pad_s=0.0)
    assert windows[0].result.turns == []
    assert not fake_pipeline["pipeline"].calls, "must not call the model on a sliver"


def test_no_spans_means_no_pipeline_load(fake_pipeline, waveform):
    assert d.diarize_windows(waveform, []) == []
    assert FakePipeline.constructed == 0


def test_labels_do_not_carry_identity_across_windows(fake_pipeline, waveform):
    """Documented limitation, pinned: SPEAKER_00 differs per window.

    pyannote numbers speakers independently per call, so nothing downstream may
    treat a label as the same person in two windows.
    """
    windows = d.diarize_windows(waveform, [(10.0, 40.0), (150.0, 180.0)])
    assert windows[0].result.speakers == windows[1].result.speakers
    # identical label sets, but they are produced by independent calls —
    # the pipeline was invoked separately for each window
    assert len(fake_pipeline["pipeline"].calls) == 2


def test_window_as_dict_carries_shares_for_the_gate(fake_pipeline, waveform):
    windows = d.diarize_windows(waveform, [(10.0, 40.0)])
    payload = windows[0].as_dict()
    assert payload["index"] == 0
    assert set(payload["shares"]) == {"SPEAKER_00", "SPEAKER_01"}
    assert payload["significant_speakers"] == ["SPEAKER_00", "SPEAKER_01"]
