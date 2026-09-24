"""Render an explicitly imported ranking artifact without crossing human gates."""
import json
from pathlib import Path
from .orchestrator import RunContext, GateFailure, hash_file, run_pipeline


def render_review(manifest_path, diagnostic_finals=0):
    from .stages import STAGES_BY_ID
    from .render import render_clip, RenderError
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text())
    video, audio = Path(data['video']).resolve(), Path(data['audio']).resolve()
    words = json.loads(Path(data['transcript']).read_text())['words']
    candidates = data['candidates']
    if not candidates:
        raise ValueError('manifest has no candidates')
    for c in candidates:
        a, b = c['start_word'], c['end_word']
        if not (0 <= a <= b < len(words)):
            raise ValueError('candidate word indices out of range')
        if abs(c['start'] - words[a]['start']) > 0.001 or abs(c['end'] - words[b]['end']) > 0.001:
            raise ValueError('candidate timing does not match the aligned words')
    workdir = manifest_path.parent / data.get('render_dir', 'renders')
    ctx = RunContext('review-materials', audio, hash_file(audio), workdir,
                     {'clip_class': 'general-own', 'experiment_only': True}, outputs={
        'S0': {'clip_class': 'general-own', 'video_path': str(video)},
        'S1': {'applicable': False}, 'S3': {'words': words},
        'S5': {'candidates': candidates}, 'S4': {'imported': True}})
    # Only render stages; S0-S5 evidence is imported, not claimed as rerun.
    specs = [STAGES_BY_ID[s] for s in ('S6', 'S7', 'S8')]
    # Artifact content is part of the identity; edits cannot reuse stale previews.
    ctx.source_hash += hash_file(manifest_path) + hash_file(Path(data['transcript'])) + hash_file(video)
    ctx.shared['on_preview'] = lambda cid, name, result: print(f'preview {cid} {name}: {result["wall_s"]:.2f}s', flush=True)
    report = run_pipeline(ctx, specs)
    print(report.render(), flush=True)
    if report.gated:
        return 2
    from .tracking import import_previews
    import_previews(ctx)
    from . import db, config
    with db.connect(config.DB_PATH) as conn:
        sid=ctx.shared['source_id']
        state=json.loads(conn.execute('SELECT state_json FROM sources WHERE id=?',(sid,)).fetchone()[0])
        state['progress']={r.id:r.__dict__ for r in report.records}
        conn.execute('UPDATE sources SET state_json=? WHERE id=?',(json.dumps(state),sid))
    finals = []
    from .tracking import effective_candidate
    for original in candidates[:diagnostic_finals]:
        c=effective_candidate(ctx,original)
        plans = ctx.output('S7')['plans'][str(c['rank'])]
        name = next(iter(plans))
        try:
            result = render_clip(video, audio, c, words, plans[name],
                                 workdir / 'diagnostic' / f"{c['rank']}-{name}.mp4",
                                 proxy=False, diagnostic=True)
        except RenderError as exc:
            raise GateFailure('S12', f'diagnostic renderer: {exc}') from exc
        finals.append(result)
        print(json.dumps(result), flush=True)
    receipt = {'imported_manifest': str(manifest_path), 'approved': False,
               'exportable': False, 'stages': [r.__dict__ for r in report.records],
               'diagnostic_finals': finals}
    (workdir / 'render-receipt.json').write_text(json.dumps(receipt, indent=2))
    return 0
