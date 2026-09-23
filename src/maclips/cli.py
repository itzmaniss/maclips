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
from .ingest import DEFAULT_MAX_HEIGHT, IngestError, is_url, probe_url, resolve
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


EDITABLE_HELP = """
Commands: <field>=<value> to edit, 'list' to redisplay, 'confirm' to save,
          'quit' to abandon. List fields take comma-separated values.
"""


def _render_brief(cfg) -> None:
    from dataclasses import fields

    print("\n" + "=" * 70)
    print(f"{cfg.brand or '(no brand)'} — {cfg.campaign_name or '(no campaign name)'}")
    print("=" * 70)
    for f in fields(cfg):
        if f.name in ("raw_brief", "confirmed", "missing"):
            continue
        value = getattr(cfg, f.name)
        shown = ", ".join(str(v) for v in value) if isinstance(value, list) else value
        flag = ""
        if f.name in cfg.missing:
            flag = "   <- brief did not specify this"
        elif shown in ("", None, []):
            flag = "   <- empty"
        print(f"  {f.name:<22} {str(shown) if shown not in (None, '') else '-':<28}{flag}")
    problems = cfg.validate()
    if problems:
        print(f"\n  REQUIRED FIELDS STILL MISSING: {', '.join(problems)}")


def _apply_edit(cfg, line: str) -> str:
    from dataclasses import fields

    if "=" not in line:
        return "expected <field>=<value>"
    name, _, raw = line.partition("=")
    name, raw = name.strip(), raw.strip()
    spec = {f.name: f for f in fields(cfg)}.get(name)
    if spec is None or name in ("raw_brief", "missing"):
        return f"unknown field {name!r}"

    current = getattr(cfg, name)
    try:
        if isinstance(current, list):
            value = [v.strip() for v in raw.split(",") if v.strip()]
        elif isinstance(current, bool) or spec.type == "bool":
            value = raw.lower() in ("1", "true", "yes", "y")
        elif raw == "":
            value = None if current is None else type(current)()
        elif isinstance(current, float) or "float" in str(spec.type):
            value = float(raw)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(raw)
        else:
            value = raw
    except ValueError as exc:
        return f"could not parse {raw!r} for {name}: {exc}"

    setattr(cfg, name, value)
    if name in cfg.missing:
        cfg.missing.remove(name)
    return ""


def _cmd_brief(args: argparse.Namespace) -> int:
    """Extract a brief with Haiku, show it, let it be edited, then save it."""
    from .brief import BriefInvalid, BriefRejected, extract, load, save

    path = Path(args.source).expanduser()
    out = Path(args.out) if args.out else config.WORK_DIR / "campaigns" / f"{path.stem}.json"

    if args.reuse and out.is_file():
        cfg = load(out)
        print(f"loaded existing config: {out}")
    else:
        if not path.is_file():
            print(f"FAILED: no such brief file: {path}", file=sys.stderr)
            return 1
        try:
            cfg = extract(path.read_text(), model=args.model)
        except BriefRejected as exc:
            print(f"REJECTED\n{exc}", file=sys.stderr)
            return 3
        except (BriefInvalid, RuntimeError) as exc:
            print(f"FAILED\n{exc}", file=sys.stderr)
            return 2

    _render_brief(cfg)
    if args.no_confirm:
        save(cfg, out)
        print(f"\nsaved UNCONFIRMED: {out}  (S5 will refuse to run)")
        return 0

    print(EDITABLE_HELP)
    while True:
        try:
            line = input("brief> ").strip()
        except EOFError:
            print()
            return 1
        if not line:
            continue
        if line in ("quit", "q"):
            print("abandoned; nothing saved")
            return 1
        if line in ("list", "l"):
            _render_brief(cfg)
            continue
        if line in ("confirm", "c"):
            problems = cfg.validate()
            if problems:
                print(f"  cannot confirm — still missing: {', '.join(problems)}")
                continue
            cfg.confirmed = True
            save(cfg, out)
            print(f"confirmed and saved: {out}")
            return 0
        error = _apply_edit(cfg, line)
        print(f"  {error}" if error else f"  {line.split('=')[0].strip()} updated")


def _cmd_capture(args: argparse.Namespace) -> int:
    """S1 front door: fetch a campaign URL into briefs/raw/, then optionally
    run it straight through S1 extraction (the existing `brief` flow)."""
    from .capture import CaptureError, capture_campaign

    try:
        result = capture_campaign(
            args.url,
            out_dir=Path(args.out) if args.out else None,
            headless=args.headless,
        )
    except CaptureError as exc:
        print(f"FAILED\n{exc}", file=sys.stderr)
        return 1

    print(f"captured  : {result.path}")
    print(f"title     : {result.title}")
    if result.fetched_docs:
        print("linked docs pulled in (Google Docs / Notion allowlist):")
        for u in result.fetched_docs:
            print(f"  {u}")
    if result.undecided_links:
        print(f"reference links NOT fetched, for you to decide ({len(result.undecided_links)}):")
        for u in result.undecided_links:
            print(f"  {u}")

    if not args.extract:
        print(f"\nnext: uv run maclips brief {result.path}")
        return 0

    brief_args = argparse.Namespace(
        source=str(result.path), out=None, model=None, reuse=False, no_confirm=False,
    )
    return _cmd_brief(brief_args)


def _cmd_probe(args: argparse.Namespace) -> int:
    """Metadata only — judge a source before spending bandwidth on it."""
    try:
        meta = probe_url(args.url)
    except IngestError as exc:
        print(f"FAILED\n{exc}", file=sys.stderr)
        return 1
    d = int(meta.duration_s)
    print(f"id       : {meta.video_id}")
    print(f"title    : {meta.title}")
    print(f"channel  : {meta.channel}")
    print(f"duration : {d // 3600}h{(d % 3600) // 60:02d}m{d % 60:02d}s ({d}s)")
    print(f"uploaded : {meta.upload_date}")
    seen = sorted({(f["width"], f["height"]) for f in meta.formats if f.get("width")},
                  key=lambda wh: wh[1])
    print("resolutions:")
    for w, h in seen:
        if h >= 240:
            print(f"  {w}x{h}  (aspect {w / h:.3f}, 9:16 crop would be {int(h * 9 / 16)}px wide)")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Register a source from a local path or a URL."""
    try:
        check_environment()
    except EnvironmentGate as gate:
        print(f"FAILED\n{gate}", file=sys.stderr)
        return 1

    dest = config.WORK_DIR / "sources"
    try:
        record = resolve(args.target, dest, max_height=args.max_height)
    except IngestError as exc:
        print(f"FAILED\n{exc}", file=sys.stderr)
        return 2

    size = record.path.stat().st_size / 2**20 if record.path.is_file() else 0
    print(f"source   : {record.path}")
    print(f"size     : {size:.0f} MiB")
    if record.video_id:
        print(f"id       : {record.video_id}")
        print(f"title    : {record.title}")
        print(f"channel  : {record.channel}")
        print(f"uploaded : {record.upload_date}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        check_environment()
    except EnvironmentGate as gate:
        print(f"FAILED\n{gate}", file=sys.stderr)
        return 1

    source_record: dict = {}
    record = None
    if is_url(args.source):
        import time

        started = time.perf_counter()

        def audio_ready(path: Path) -> None:
            print(f"audio ready after {time.perf_counter() - started:.1f}s: {path.name} "
                  f"— S2 can start; video still downloading", flush=True)

        try:
            record = resolve(args.source, config.WORK_DIR / "sources",
                             max_height=args.max_height, on_audio_ready=audio_ready)
        except IngestError as exc:
            print(f"FAILED\n{exc}", file=sys.stderr)
            return 2
        source = record.audio_source
        source_record = record.as_dict()
    else:
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
            "source_record": source_record,
            "full_diarization": args.full_diarization,
        },
    )
    if record is not None:
        # S6 needs this to wait for the video stream; it is an object, not
        # checkpointable state, so it goes in `shared`.
        ctx.shared["source_record"] = record

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

    brief = sub.add_parser("brief", help="extract a campaign brief and confirm it")
    brief.add_argument("source", help="a text file containing the brief")
    brief.add_argument("--out", default=None, help="where to save the campaign config")
    brief.add_argument("--model", default=None, help="override the extraction model")
    brief.add_argument("--reuse", action="store_true",
                       help="reload a saved config instead of re-extracting")
    brief.add_argument("--no-confirm", action="store_true",
                       help="extract and save without confirming (S5 will refuse it)")

    capture = sub.add_parser("capture", help="fetch a campaign URL into briefs/raw/ (S1 front door)")
    capture.add_argument("url", help="the campaign page to capture")
    capture.add_argument("--out", default=None, help="override briefs/raw/")
    capture.add_argument("--headless", action="store_true",
                         help="no login pause; needs a session already saved in "
                              "the browser profile from a prior headful capture")
    capture.add_argument("--extract", action="store_true",
                         help="immediately run the captured file through `brief`")

    probe = sub.add_parser("probe", help="fetch a URL's metadata only, no download")
    probe.add_argument("url")

    ing = sub.add_parser("ingest", help="register a source from a local path or a URL")
    ing.add_argument("target", help="local media file, or a URL")
    ing.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT,
                     help=f"cap source height (default {DEFAULT_MAX_HEIGHT}); "
                          "a 9:16 crop is height*9/16 wide, so 1080x1920 output "
                          "needs height>=1920 to avoid upscaling")

    run = sub.add_parser("run", help="run the stage pipeline over a source")
    run.add_argument("source", help="local media file, or a URL")
    run.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT,
                     help="cap source height when the source is a URL")
    run.add_argument("--clip-class", default="general-own",
                     choices=["campaign", "general-own"],
                     help="general-3p is deferred (PLAN.md §1.4)")
    run.add_argument("--from", dest="from_stage", default=None,
                     help="force recomputation from this stage id, e.g. S5")
    run.add_argument("--expected-speakers", type=int, default=None,
                     help="enables the S4 speaker-count gate")
    run.add_argument("--language", default=None, help="enables the S3 language gate")
    run.add_argument("--candidates", type=int, default=12)
    run.add_argument("--full-diarization", action="store_true",
                     help="diarize the whole source instead of candidate windows. "
                          "Slow (9m per 2h, measured) and not the default; kept for "
                          "the step-4 ranking comparison and cross-source speaker identity")

    args = parser.parse_args()
    if args.command == "check":
        return _cmd_check()
    if args.command == "brief":
        return _cmd_brief(args)
    if args.command == "capture":
        return _cmd_capture(args)
    if args.command == "probe":
        return _cmd_probe(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    return _cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
