"""Stage runner: content-hash caching, per-stage checkpoints, hard gates.

Three ideas, and nothing else:

1. **Content-hash caching.** A stage's cache key is derived from the source
   hash, the stage's own id and version, the config values the stage declares
   it reads, and the cache keys of the stages it depends on. Change any of
   those and the stage recomputes; change none and it is skipped. Invalidation
   cascades downstream because upstream keys feed into the hash.

2. **Per-stage checkpointing.** A completed stage writes its output to
   `<workdir>/checkpoints/<id>.json`. A re-run reads it back when the cache key
   still matches, so a gated run can be resumed from the stage that stopped it
   rather than from the top.

3. **Hard gates.** A stage raises `GateFailure` and the run stops with a
   reason. It never warns and continues. Downstream stages are reported as
   `blocked`, not silently skipped.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

StageOutput = dict[str, Any]

CACHE_KEY_LENGTH = 16


class GateFailure(Exception):
    """A stage detected a condition that must stop the run.

    Raised by stage bodies. Carries the stage id and the reason so the UI can
    show exactly which stage stopped the run and why (PLAN.md §6.1).
    """

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"[{stage}] {reason}")


@dataclass
class RunContext:
    """Everything a stage is allowed to read."""

    run_id: str
    source: Path
    source_hash: str
    workdir: Path
    config: Mapping[str, Any]
    outputs: dict[str, StageOutput] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)
    """In-memory hand-off between stages, deliberately **not** checkpointed.

    Checkpoints are JSON, but S3 and S4 must share the decoded waveform (~450 MB
    of float32 for a 2-hour source) and the live transcript objects. Putting
    them here keeps one copy in memory and keeps the checkpoint files small.
    A resumed run finds `shared` empty and reloads from the S2 WAV, so nothing
    depends on it surviving."""

    def output(self, stage_id: str) -> StageOutput:
        """Read an upstream stage's output, failing loudly if it is absent."""
        try:
            return self.outputs[stage_id]
        except KeyError:
            raise LookupError(
                f"stage {stage_id} has no output yet; declare it in `needs`"
            ) from None

    @property
    def checkpoint_dir(self) -> Path:
        return self.workdir / "checkpoints"


@dataclass(frozen=True)
class StageSpec:
    """One pipeline stage.

    `gate` is the human-readable condition from PLAN.md §2.1, carried here so
    the UI and the report can show what a stage is capable of stopping on even
    when it passes. `version` is bumped by hand when a stage's logic changes in
    a way that must invalidate its cache.
    """

    id: str
    name: str
    summary: str
    run: Callable[[RunContext], StageOutput]
    needs: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    gate: str = ""
    version: int = 1


@dataclass
class StageRecord:
    """What happened to one stage in one run."""

    id: str
    name: str
    status: str  # ran | cached | gated | blocked
    cache_key: str = ""
    reason: str = ""
    duration_s: float = 0.0


@dataclass
class RunReport:
    run_id: str
    records: list[StageRecord] = field(default_factory=list)

    @property
    def gated(self) -> StageRecord | None:
        return next((r for r in self.records if r.status == "gated"), None)

    @property
    def ok(self) -> bool:
        return self.gated is None

    def render(self) -> str:
        """A per-stage status line, the text form of PLAN.md §6.1's progress view."""
        marks = {"ran": "+", "cached": "=", "gated": "!", "blocked": "-"}
        lines = []
        for r in self.records:
            line = f" {marks.get(r.status, '?')} {r.id:<3} {r.name:<22} {r.status}"
            if r.status == "ran":
                line += f" ({r.duration_s:.2f}s)"
            if r.reason:
                line += f"\n       reason: {r.reason}"
            lines.append(line)
        return "\n".join(lines)


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming sha256 of a file's bytes. Sources are too big to slurp."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(value: Any) -> str:
    """Deterministic text for a config value, so the hash does not drift."""
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def cache_key(spec: StageSpec, ctx: RunContext, upstream: Mapping[str, str]) -> str:
    """Content hash for one stage.

    Includes the upstream stages' cache keys, which is what makes invalidation
    cascade: re-transcribing changes S3's key, which changes S4's, and so on.
    """
    digest = hashlib.sha256()
    digest.update(ctx.source_hash.encode())
    digest.update(f"{spec.id}:{spec.version}".encode())
    for name in spec.params:
        digest.update(f"{name}={_stable(ctx.config.get(name))}".encode())
    for need in spec.needs:
        digest.update(f"{need}={upstream.get(need, '')}".encode())
    return digest.hexdigest()[:CACHE_KEY_LENGTH]


def _checkpoint_path(ctx: RunContext, spec: StageSpec) -> Path:
    return ctx.checkpoint_dir / f"{spec.id}.json"


def read_checkpoint(ctx: RunContext, spec: StageSpec, key: str) -> StageOutput | None:
    """Return the stored output when it was produced under the same cache key."""
    path = _checkpoint_path(ctx, spec)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None  # a corrupt checkpoint just means recompute
    if payload.get("cache_key") != key:
        return None
    output = payload.get("output")
    return output if isinstance(output, dict) else None


def write_checkpoint(ctx: RunContext, spec: StageSpec, key: str, output: StageOutput) -> None:
    path = _checkpoint_path(ctx, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": spec.id,
        "name": spec.name,
        "cache_key": key,
        "completed_at": time.time(),
        "output": output,
    }
    # Write-then-rename so an interrupted write cannot leave a half checkpoint
    # that the next run would have to guess about.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def run_pipeline(
    ctx: RunContext,
    stages: Sequence[StageSpec],
    from_stage: str | None = None,
) -> RunReport:
    """Run `stages` in order, stopping at the first gate.

    `from_stage` forces recomputation from that stage onward, which is what the
    UI's re-run-from-stage button drives. Stages before it still load from
    their checkpoints.
    """
    report = RunReport(run_id=ctx.run_id)
    keys: dict[str, str] = {}
    forcing = False
    stopped = False

    for spec in stages:
        if stopped:
            report.records.append(StageRecord(spec.id, spec.name, "blocked"))
            continue

        if from_stage is not None and spec.id == from_stage:
            forcing = True

        key = cache_key(spec, ctx, keys)
        keys[spec.id] = key

        cached = None if forcing else read_checkpoint(ctx, spec, key)
        if cached is not None:
            ctx.outputs[spec.id] = cached
            report.records.append(StageRecord(spec.id, spec.name, "cached", key))
            continue

        started = time.perf_counter()
        try:
            output = spec.run(ctx)
        except GateFailure as gate:
            report.records.append(
                StageRecord(
                    spec.id, spec.name, "gated", key, gate.reason,
                    time.perf_counter() - started,
                )
            )
            stopped = True
            continue

        elapsed = time.perf_counter() - started
        ctx.outputs[spec.id] = output
        write_checkpoint(ctx, spec, key, output)
        report.records.append(StageRecord(spec.id, spec.name, "ran", key, "", elapsed))

    return report
