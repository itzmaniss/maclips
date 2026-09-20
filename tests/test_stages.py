"""S0-S14: the pipeline runs end to end, and every gate can fire.

S3 and S4 are stubbed here. Their real implementations load multi-gigabyte
models and (for S4) reach Hugging Face; neither belongs in a unit suite. The
gates themselves are exercised through the stubs' outputs, which is what the
gates actually read.
"""
from __future__ import annotations

import pytest

from maclips import diarize as diarize_mod
from maclips import transcribe as transcribe_mod
from maclips.orchestrator import RunContext, hash_file, run_pipeline
from maclips.stages import STAGES
from maclips.transcribe import Transcript, Word


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    """Keep every test off the GPU, the model cache and the network."""
    def fake_run(waveform, model_name=None, language=None, on_stage=None):
        return Transcript(
            words=[Word("hello", 0.0, 0.5, 0.9), Word("there", 0.5, 1.0, 0.9)],
            language=language or "en",
            segments=[{"start": 0.0, "end": 1.0, "text": "hello there"}],
        )

    def _result(n=2):
        turns = [diarize_mod.Turn(i * 10.0, (i + 1) * 10.0, f"SPEAKER_{i:02d}") for i in range(n)]
        return diarize_mod.DiarizationResult(
            turns=turns, speakers=sorted({t.speaker for t in turns})
        )

    def fake_diarize(waveform, max_speakers=None, **kw):
        return _result()

    def fake_windows(waveform, spans, **kw):
        return [
            diarize_mod.Window(i, s, e, _result())
            for i, (s, e) in enumerate(spans)
        ]

    monkeypatch.setattr(transcribe_mod, "run", fake_run)
    monkeypatch.setattr(diarize_mod, "diarize", fake_diarize)
    monkeypatch.setattr(diarize_mod, "diarize_windows", fake_windows)
    monkeypatch.setattr(diarize_mod, "check_access", lambda *a, **k: "stub-sha")


def make_ctx(tmp_path, source, **config):
    base = {
        "clip_class": "general-own",
        "min_duration_s": 1.0,          # the 3 s fixture is deliberately short
        "approvals": [{"id": "c1", "commentary_mode": "none"}],
        # S4 is window-only now and needs spans from S5; S5's own gate wants >=5.
        "stub_candidates": [
            {"start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(5)
        ],
    }
    base.update(config)
    return RunContext(
        run_id="test",
        source=source,
        source_hash=hash_file(source),
        workdir=tmp_path / "work",
        config=base,
    )


def test_all_fifteen_stages_are_registered():
    assert sorted((s.id for s in STAGES), key=lambda i: int(i[1:])) == [
        f"S{i}" for i in range(15)
    ]


def test_s4_runs_after_s5_because_diarization_is_window_only():
    """Execution order, not id order: S4 needs S5's candidate spans (§2.2)."""
    order = [s.id for s in STAGES]
    assert order.index("S5") < order.index("S4")
    by_id = {s.id: s for s in STAGES}
    assert by_id["S4"].needs == ("S5",)
    assert "S4" not in by_id["S5"].needs, "S5 must not wait on diarization"
    assert by_id["S5"].needs == ("S1", "S3")


def test_stub_pipeline_runs_end_to_end(tmp_path, tiny_av):
    report = run_pipeline(make_ctx(tmp_path, tiny_av), STAGES)
    assert report.ok, report.render()
    assert [r.status for r in report.records] == ["ran"] * 15


def test_second_end_to_end_run_is_fully_cached(tmp_path, tiny_av):
    ctx = make_ctx(tmp_path, tiny_av)
    run_pipeline(ctx, STAGES)
    ctx.outputs.clear()
    ctx.shared.clear()
    report = run_pipeline(ctx, STAGES)
    assert [r.status for r in report.records] == ["cached"] * 15


def test_every_declared_gate_is_documented():
    gateless = {s.id for s in STAGES if not s.gate}
    assert gateless == {"S2", "S8", "S13"}


# --------------------------------------------------------------------------- #
# S0 gates on real probe data
# --------------------------------------------------------------------------- #

def test_s0_gates_on_missing_audio_stream(tmp_path, video_only):
    gated = run_pipeline(make_ctx(tmp_path, video_only), STAGES).gated
    assert gated is not None and gated.id == "S0"
    assert "no audio stream" in gated.reason


def test_s0_gates_on_short_source_at_the_real_floor(tmp_path, tiny_av):
    """The 2-minute floor from §2.1, against a genuinely 3-second file."""
    ctx = make_ctx(tmp_path, tiny_av, min_duration_s=120.0)
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S0"
    assert "the floor is 120s" in gated.reason


def test_s0_gates_on_unreadable_container(tmp_path):
    junk = tmp_path / "not-media.mp4"
    junk.write_bytes(b"this is not a container")
    gated = run_pipeline(make_ctx(tmp_path, junk), STAGES).gated
    assert gated is not None and gated.id == "S0"
    assert "unreadable container" in gated.reason


# --------------------------------------------------------------------------- #
# Every other gate, fired
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("stage_id", "config", "expected"),
    [
        ("S0", {"clip_class": "general-3p"}, "not offered"),
        ("S1", {"clip_class": "campaign", "brief": {"brand": "B", "campaign_name": "C",
                "platforms_allowed": ["tiktok"], "category": "crypto"}}, "rejected"),
        ("S1", {"clip_class": "campaign", "brief": {"brand": "B", "campaign_name": "C",
                "platforms_allowed": ["tiktok"], "category": "saas"}}, "not confirmed"),
        ("S1", {"clip_class": "campaign", "brief": {"category": "saas"}},
         "missing required fields"),
        ("S1", {"clip_class": "campaign", "brief": {}}, "no campaign brief"),
        ("S5", {"stub_malformed_json": True}, "malformed JSON"),
        ("S5", {"stub_candidates": [1, 2, 3]}, "the floor is 5"),
        ("S9", {"approvals": []}, "no clips approved"),
        ("S11", {"stub_compliance_failures": ["duration out of range"]}, "block export"),
    ],
)
def test_gate_fires(tmp_path, tiny_av, stage_id, config, expected):
    report = run_pipeline(make_ctx(tmp_path, tiny_av, **config), STAGES)
    gated = report.gated
    assert gated is not None, f"{stage_id} gate did not fire\n{report.render()}"
    assert gated.id == stage_id
    assert expected in gated.reason


def test_s3_gates_on_weak_alignment_not_coverage(tmp_path, tiny_av, monkeypatch):
    """Interpolated words carry timings but no score — coverage cannot see them."""
    def interpolated(waveform, model_name=None, language=None, on_stage=None):
        # every word has start/end (so coverage is 100%) but no score
        return Transcript(
            words=[Word(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(10)],
            language="en",
        )

    monkeypatch.setattr(transcribe_mod, "run", interpolated)
    report = run_pipeline(make_ctx(tmp_path, tiny_av), STAGES)
    gated = report.gated
    assert gated is not None and gated.id == "S3"
    assert "weakly aligned" in gated.reason
    assert "10 interpolated" in gated.reason


def test_s3_coverage_alone_would_not_have_fired(tmp_path, tiny_av, monkeypatch):
    """The old gate's blind spot, pinned so it cannot return."""
    words = [Word(f"w{i}", i * 0.1, i * 0.1 + 0.05) for i in range(10)]
    t = Transcript(words=words, language="en")
    assert t.coverage == 1.0, "interpolated words look perfect to coverage"
    assert t.weak_fraction(0.3) == 1.0, "but every one is weakly aligned"


def test_s3_passes_when_words_are_acoustically_aligned(tmp_path, tiny_av, monkeypatch):
    def good(waveform, model_name=None, language=None, on_stage=None):
        return Transcript(
            words=[Word(f"w{i}", i * 0.1, i * 0.1 + 0.05, 0.9) for i in range(20)],
            language="en",
        )

    monkeypatch.setattr(transcribe_mod, "run", good)
    assert run_pipeline(make_ctx(tmp_path, tiny_av), STAGES).ok


def test_s3_gates_on_unexpected_language(tmp_path, tiny_av, monkeypatch):
    seen = {}

    def french(waveform, model_name=None, language=None, on_stage=None):
        seen["language_arg"] = language
        return Transcript(words=[Word("bonjour", 0.0, 0.5, 0.9)], language="fr")

    monkeypatch.setattr(transcribe_mod, "run", french)
    ctx = make_ctx(tmp_path, tiny_av, expected_language="en")
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S3"
    assert "!= expected" in gated.reason
    assert seen["language_arg"] is None, (
        "Whisper must auto-detect; forcing it to expected_language would make "
        "this gate tautological"
    )


def test_s4_gates_on_significant_speaker_count(tmp_path, tiny_av, monkeypatch):
    """Three substantial speakers where two were expected."""
    def three(waveform, spans, **kw):
        turns = [diarize_mod.Turn(i * 10.0, (i + 1) * 10.0, f"SPEAKER_{i:02d}") for i in range(3)]
        result = diarize_mod.DiarizationResult(turns=turns, speakers=[t.speaker for t in turns])
        return [diarize_mod.Window(0, spans[0][0], spans[0][1], result)]

    monkeypatch.setattr(diarize_mod, "diarize_windows", three)
    ctx = make_ctx(tmp_path, tiny_av, expected_speaker_count=2)
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S4"
    assert "you expected 2" in gated.reason
    assert ">=5%" in gated.reason


def test_s4_sliver_label_does_not_trip_the_count_gate(tmp_path, tiny_av, monkeypatch):
    """A 1% backchannel label is not a third speaker."""
    def two_plus_sliver(waveform, spans, **kw):
        turns = [
            diarize_mod.Turn(0.0, 50.0, "SPEAKER_00"),
            diarize_mod.Turn(50.0, 99.0, "SPEAKER_01"),
            diarize_mod.Turn(99.0, 100.0, "SPEAKER_02"),   # 1% — noise
        ]
        result = diarize_mod.DiarizationResult(
            turns=turns, speakers=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]
        )
        return [diarize_mod.Window(0, spans[0][0], spans[0][1], result)]

    monkeypatch.setattr(diarize_mod, "diarize_windows", two_plus_sliver)
    ctx = make_ctx(tmp_path, tiny_av, expected_speaker_count=2)
    assert run_pipeline(ctx, STAGES).ok, "raw label count would have failed this"


def test_s4_gates_when_s5_produced_no_spans(tmp_path, tiny_av):
    ctx = make_ctx(tmp_path, tiny_av, stub_candidates=[])
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S4"
    assert "no candidate spans" in gated.reason


def test_s4_gates_when_hf_access_is_refused(tmp_path, tiny_av, monkeypatch):
    def refuse(*a, **k):
        raise diarize_mod.DiarizationAccessError(
            "gated model not accepted or token lacks access: pyannote/..."
        )

    monkeypatch.setattr(diarize_mod, "check_access", refuse)
    gated = run_pipeline(make_ctx(tmp_path, tiny_av), STAGES).gated
    assert gated is not None and gated.id == "S4"
    assert "gated model not accepted or token lacks access" in gated.reason


def test_s10_blocks_unaccepted_commentary(tmp_path, tiny_av):
    ctx = make_ctx(tmp_path, tiny_av, approvals=[{"id": "c1", "commentary_mode": "auto"}])
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S10"


def test_s12_blocks_duration_drift(tmp_path, tiny_av):
    ctx = make_ctx(
        tmp_path, tiny_av,
        stub_rendered=[{"id": "c1", "planned_duration_s": 42.0, "actual_duration_s": 42.4}],
    )
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S12" and "drifted" in gated.reason


def test_s14_blocks_campaign_clip_without_disclosure(tmp_path, tiny_av):
    ctx = make_ctx(
        tmp_path, tiny_av,
        clip_class="campaign",
        brief={"brand": "B", "campaign_name": "C", "platforms_allowed": ["tiktok"],
               "category": "saas", "confirmed": True},
        post_rows=[{"clip_id": "c1", "disclosure_ticked": False}],
    )
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S14" and "paid-promotion" in gated.reason


def test_general_own_clip_needs_no_disclosure(tmp_path, tiny_av):
    ctx = make_ctx(tmp_path, tiny_av, post_rows=[{"clip_id": "c1", "disclosure_ticked": False}])
    assert run_pipeline(ctx, STAGES).ok


def test_gated_run_resumes_from_the_stage_that_stopped(tmp_path, tiny_av, monkeypatch):
    def french(waveform, model_name=None, language=None, on_stage=None):
        return Transcript(words=[Word("bonjour", 0.0, 0.5, 0.9)], language="fr")

    monkeypatch.setattr(transcribe_mod, "run", french)
    first = run_pipeline(make_ctx(tmp_path, tiny_av, expected_language="en"), STAGES)
    assert first.gated is not None and first.gated.id == "S3"
    assert [r.status for r in first.records[:3]] == ["ran", "ran", "ran"]

    # Re-patch rather than monkeypatch.undo(): undo() would also revert the
    # autouse fixture's patches, letting the real check_access reach the network.
    monkeypatch.setattr(
        transcribe_mod, "run",
        lambda w, model_name=None, language=None, on_stage=None: Transcript(
            words=[Word("hello", 0.0, 0.5, 0.9)], language="en"),
    )
    second = run_pipeline(make_ctx(tmp_path, tiny_av, expected_language="en"), STAGES)
    assert second.ok, second.render()
    assert [r.status for r in second.records[:3]] == ["cached", "cached", "cached"]
