"""CLI entry point.

    uv run maclips check                 # run the startup gates and stop
    uv run maclips run <source>          # run the stage pipeline
    uv run maclips run <source> --from S5

The gates run before anything else on `run`, so a missing libass or a CPU-only
MLX stops the process instead of surfacing three stages later.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from . import config
from .gates import EnvironmentGate, check_environment
from .orchestrator import RunContext, hash_file, run_pipeline
from .stages import STAGES, STAGES_BY_ID


def _cmd_check() -> int:
    try:
        report = check_environment()
    except EnvironmentGate as gate:
        print(f"FAILED\n{gate}", file=sys.stderr)
        return 1
    print("environment OK")
    for line in report.passed:
        print(f"  {line}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        check_environment()
    except EnvironmentGate as gate:
        print(f"FAILED\n{gate}", file=sys.stderr)
        return 1

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        print(f"FAILED: no such file: {source}", file=sys.stderr)
        return 1

    if args.from_stage and args.from_stage not in STAGES_BY_ID:
        print(f"FAILED: unknown stage {args.from_stage!r}", file=sys.stderr)
        return 1

    run_id = uuid.uuid4().hex[:12]
    source_hash = hash_file(source)
    workdir = config.WORK_DIR / source_hash[:16]
    workdir.mkdir(parents=True, exist_ok=True)

    ctx = RunContext(
        run_id=run_id,
        source=source,
        source_hash=source_hash,
        workdir=workdir,
        config={
            "clip_class": args.clip_class,
            "expected_speaker_count": args.expected_speakers,
            "expected_language": args.language,
            "candidate_count": args.candidates,
        },
    )

    print(f"run {run_id}  source {source.name}  hash {source_hash[:16]}")
    report = run_pipeline(ctx, STAGES, from_stage=args.from_stage)
    print(report.render())

    gated = report.gated
    if gated is not None:
        print(f"\nrun stopped at {gated.id} ({gated.name}): {gated.reason}", file=sys.stderr)
        print(f"re-run from it with: --from {gated.id}", file=sys.stderr)
        return 2
    print("\nall stages complete")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="maclips", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="run the startup environment gates and exit")

    run = sub.add_parser("run", help="run the stage pipeline over a source")
    run.add_argument("source", help="local media file")
    run.add_argument("--clip-class", default="general-own",
                     choices=["campaign", "general-own"],
                     help="general-3p is deferred (PLAN.md §1.4)")
    run.add_argument("--from", dest="from_stage", default=None,
                     help="force recomputation from this stage id, e.g. S5")
    run.add_argument("--expected-speakers", type=int, default=None,
                     help="enables the S4 speaker-count gate")
    run.add_argument("--language", default=None, help="enables the S3 language gate")
    run.add_argument("--candidates", type=int, default=12)

    args = parser.parse_args()
    if args.command == "check":
        return _cmd_check()
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
