"""S0-S14 stubs: the pipeline runs end to end, and every gate can fire."""
from __future__ import annotations

import pytest

from maclips.orchestrator import RunContext, hash_file, run_pipeline
from maclips.stages import STAGES, STAGES_BY_ID


def make_ctx(tmp_path, **config):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"stub source bytes")
    base = {
        "clip_class": "general-own",
        # S9 is a human gate; an approval stands in for the reviewer.
        "approvals": [{"id": "c1", "commentary_mode": "none"}],
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
    assert [s.id for s in STAGES] == [f"S{i}" for i in range(15)]


def test_stub_pipeline_runs_end_to_end(tmp_path):
    """Build step 1's 'done when': every stage completes on one file."""
    report = run_pipeline(make_ctx(tmp_path), STAGES)
    assert report.ok, report.render()
    assert [r.status for r in report.records] == ["ran"] * 15


def test_second_end_to_end_run_is_fully_cached(tmp_path):
    ctx = make_ctx(tmp_path)
    run_pipeline(ctx, STAGES)
    ctx.outputs.clear()
    report = run_pipeline(ctx, STAGES)
    assert [r.status for r in report.records] == ["cached"] * 15


def test_every_declared_gate_is_documented():
    """A stage that can stop the run must say so; §2.1 lists S2/S8/S13 as gateless."""
    gateless = {s.id for s in STAGES if not s.gate}
    assert gateless == {"S2", "S8", "S13"}


# --------------------------------------------------------------------------- #
# Each gate, fired.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("stage_id", "config", "expected"),
    [
        ("S0", {"clip_class": "general-3p"}, "not offered"),
        ("S0", {"stub_duration_s": 45.0}, "the floor is 120s"),
        ("S1", {"clip_class": "campaign", "brief": {"category": "crypto"}}, "rejected"),
        ("S1", {"clip_class": "campaign", "brief": {"category": "saas"}}, "not confirmed"),
        ("S3", {"stub_coverage": 0.41}, "below 97%"),
        ("S3", {"stub_language": "en", "expected_language": "fr"}, "!= expected"),
        ("S4", {"stub_speakers": 3, "expected_speaker_count": 2}, "you expected 2"),
        ("S5", {"stub_malformed_json": True}, "malformed JSON"),
        ("S5", {"stub_candidates": [1, 2, 3]}, "the floor is 5"),
        ("S9", {"approvals": []}, "no clips approved"),
        ("S11", {"stub_compliance_failures": ["duration out of range"]}, "block export"),
    ],
)
def test_gate_fires(tmp_path, stage_id, config, expected):
    report = run_pipeline(make_ctx(tmp_path, **config), STAGES)
    gated = report.gated
    assert gated is not None, f"{stage_id} gate did not fire\n{report.render()}"
    assert gated.id == stage_id
    assert expected in gated.reason


def test_s10_blocks_unaccepted_commentary(tmp_path):
    ctx = make_ctx(tmp_path, approvals=[{"id": "c1", "commentary_mode": "auto"}])
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S10"
    assert "unaccepted" in gated.reason


def test_s12_blocks_duration_drift(tmp_path):
    ctx = make_ctx(
        tmp_path,
        stub_rendered=[{"id": "c1", "planned_duration_s": 42.0, "actual_duration_s": 42.4}],
    )
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S12"
    assert "drifted" in gated.reason


def test_s14_blocks_campaign_clip_without_disclosure(tmp_path):
    ctx = make_ctx(
        tmp_path,
        clip_class="campaign",
        brief={"category": "saas", "confirmed": True},
        post_rows=[{"clip_id": "c1", "disclosure_ticked": False}],
    )
    gated = run_pipeline(ctx, STAGES).gated
    assert gated is not None and gated.id == "S14"
    assert "paid-promotion" in gated.reason


def test_general_own_clip_needs_no_disclosure(tmp_path):
    """The disclosure gate is class-scoped; it must not fire on general-own."""
    ctx = make_ctx(tmp_path, post_rows=[{"clip_id": "c1", "disclosure_ticked": False}])
    assert run_pipeline(ctx, STAGES).ok


def test_gated_run_resumes_from_the_stage_that_stopped(tmp_path):
    """A gated run is fixed and re-run from its stage, not from the top."""
    ctx = make_ctx(tmp_path, stub_coverage=0.41)
    first = run_pipeline(ctx, STAGES)
    assert first.gated is not None and first.gated.id == "S3"
    # S0-S2 completed, so they are checkpointed and should come back cached.
    assert [r.status for r in first.records[:3]] == ["ran", "ran", "ran"]

    fixed = make_ctx(tmp_path, stub_coverage=1.0)
    second = run_pipeline(fixed, STAGES)
    assert second.ok
    assert [r.status for r in second.records[:3]] == ["cached", "cached", "cached"]
