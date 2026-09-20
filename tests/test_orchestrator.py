"""The three orchestrator guarantees: caching, checkpointing, gates."""
from __future__ import annotations

import pytest

from maclips.orchestrator import (
    GateFailure,
    RunContext,
    StageSpec,
    cache_key,
    hash_file,
    read_checkpoint,
    run_pipeline,
)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"not really a video, but it hashes")
    return path


@pytest.fixture
def ctx(tmp_path, source):
    return RunContext(
        run_id="test",
        source=source,
        source_hash=hash_file(source),
        workdir=tmp_path / "work",
        config={"knob": 1},
    )


def _counting_stage(stage_id="A", **kw):
    """A stage that records how many times it actually executed."""
    calls = []

    def run(ctx):
        calls.append(1)
        return {"ran": len(calls)}

    return StageSpec(stage_id, f"stage-{stage_id}", "", run, **kw), calls


def test_hash_file_tracks_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"one")
    assert hash_file(a) == hash_file(b)
    b.write_bytes(b"two")
    assert hash_file(a) != hash_file(b)


def test_second_run_is_cached_not_rerun(ctx):
    spec, calls = _counting_stage()
    assert run_pipeline(ctx, [spec]).records[0].status == "ran"
    assert len(calls) == 1

    ctx.outputs.clear()
    report = run_pipeline(ctx, [spec])
    assert report.records[0].status == "cached"
    assert len(calls) == 1, "a cached stage must not execute again"


def test_changing_a_declared_param_invalidates(ctx, tmp_path, source):
    spec, calls = _counting_stage(params=("knob",))
    run_pipeline(ctx, [spec])
    assert len(calls) == 1

    changed = RunContext("test", source, ctx.source_hash, ctx.workdir, {"knob": 2})
    assert run_pipeline(changed, [spec]).records[0].status == "ran"
    assert len(calls) == 2


def test_undeclared_param_does_not_invalidate(ctx, source):
    """Only declared params feed the hash; that is what makes the key honest."""
    spec, calls = _counting_stage(params=("knob",))
    run_pipeline(ctx, [spec])
    other = RunContext("t", source, ctx.source_hash, ctx.workdir, {"knob": 1, "unrelated": 9})
    assert run_pipeline(other, [spec]).records[0].status == "cached"
    assert len(calls) == 1


def test_invalidation_cascades_downstream(ctx, source):
    """Re-running an upstream stage must invalidate everything after it."""
    up, up_calls = _counting_stage("A", params=("knob",))
    down, down_calls = _counting_stage("B", needs=("A",))

    run_pipeline(ctx, [up, down])
    assert (len(up_calls), len(down_calls)) == (1, 1)

    changed = RunContext("t", source, ctx.source_hash, ctx.workdir, {"knob": 99})
    report = run_pipeline(changed, [up, down])
    assert [r.status for r in report.records] == ["ran", "ran"]
    assert (len(up_calls), len(down_calls)) == (2, 2)


def test_checkpoint_round_trips(ctx):
    spec, _ = _counting_stage()
    run_pipeline(ctx, [spec])
    key = cache_key(spec, ctx, {})
    assert read_checkpoint(ctx, spec, key) == {"ran": 1}
    assert read_checkpoint(ctx, spec, "a-different-key") is None


def test_corrupt_checkpoint_recomputes_instead_of_crashing(ctx):
    spec, calls = _counting_stage()
    run_pipeline(ctx, [spec])
    (ctx.checkpoint_dir / "A.json").write_text("{ this is not json")
    ctx.outputs.clear()
    assert run_pipeline(ctx, [spec]).records[0].status == "ran"
    assert len(calls) == 2


def test_gate_stops_the_run_and_blocks_downstream(ctx):
    def gated(c):
        raise GateFailure("A", "coverage 41% is below 97%")

    spec_a = StageSpec("A", "gated-stage", "", gated)
    spec_b, b_calls = _counting_stage("B", needs=("A",))

    report = run_pipeline(ctx, [spec_a, spec_b])
    assert [r.status for r in report.records] == ["gated", "blocked"]
    assert report.gated is not None
    assert report.gated.reason == "coverage 41% is below 97%"
    assert not report.ok
    assert not b_calls, "a blocked stage must not execute"


def test_gated_stage_writes_no_checkpoint(ctx):
    """A gate is a stop, not a result. Nothing may be cached from it."""
    spec = StageSpec("A", "gated", "", lambda c: (_ for _ in ()).throw(GateFailure("A", "no")))
    run_pipeline(ctx, [spec])
    assert not (ctx.checkpoint_dir / "A.json").exists()


def test_from_stage_forces_recomputation(ctx):
    up, up_calls = _counting_stage("A")
    down, down_calls = _counting_stage("B", needs=("A",))
    run_pipeline(ctx, [up, down])

    ctx.outputs.clear()
    report = run_pipeline(ctx, [up, down], from_stage="B")
    assert [r.status for r in report.records] == ["cached", "ran"]
    assert (len(up_calls), len(down_calls)) == (1, 2)
