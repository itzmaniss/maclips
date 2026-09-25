"""S8 and S12: render checkpointed candidates from the original streams."""
from pathlib import Path
import time
from .orchestrator import GateFailure


def proxy_render(ctx):
    from .render import RenderError, render_clip
    media = ctx.output('S6')
    words = ctx.output('S3')['words']
    from .tracking import register_context, record_preview, effective_candidate
    source_id = register_context(ctx)
    previews = {}
    started = time.perf_counter()
    for original in ctx.output('S5')['candidates']:
        candidate = effective_candidate(ctx, original)
        from . import db, config
        from .tracking import candidate_key
        with db.connect(config.DB_PATH) as conn:
            posted=conn.execute('SELECT 1 FROM posts p JOIN clips c ON c.id=p.clip_id WHERE c.source_id=? AND c.candidate_key=? AND p.post_url IS NOT NULL',
                                (source_id,candidate_key(original))).fetchone()
        if posted:
            raise GateFailure('S8','posted clip is immutable; cannot regenerate its previews')
        cid = str(candidate['rank'])
        previews[cid] = {}
        for name, layout in ctx.output('S7')['plans'][cid].items():
            path = ctx.workdir / config.PREVIEW_DIR / f'{cid}-{name}.mp4'
            try:
                result = render_clip(Path(media['video_path']), Path(media['audio_path']),
                                     candidate, words, layout, path, proxy=True)
            except RenderError as exc:
                raise GateFailure('S8', str(exc)) from exc
            previews[cid][name] = result
            record_preview(ctx,original,name,result)
            callback = ctx.shared.get('on_preview')
            if callback:
                callback(cid, name, result)
    return {'previews': previews, 'wall_s': time.perf_counter() - started}


def final_render(ctx):
    from .render import RenderError, render_clip
    from .ranking import DEFAULT_MAX_DURATION_S, DEFAULT_MIN_DURATION_S
    media = ctx.output('S6')
    rendered = []
    # S11 checks the source span; dead-air removal shortens the clip, so the
    # edited duration is gated against the same brief range before encoding.
    brief = ctx.outputs.get('S1', {}).get('brief') or {}
    low, high = brief.get('min_duration_s'), brief.get('max_duration_s')
    allowed = (DEFAULT_MIN_DURATION_S if low is None else float(low),
               DEFAULT_MAX_DURATION_S if high is None else float(high))
    for clip in ctx.output('S9')['approved']:
        cid = str(clip['rank'])
        name = clip['layout']
        plans = ctx.output('S7')['plans'].get(cid, {})
        if name not in plans:
            raise GateFailure('S12', f'clip {cid}: unavailable layout {name}')
        try:
            result = render_clip(Path(media['video_path']), Path(media['audio_path']),
                                 clip, ctx.output('S3')['words'], plans[name],
                                 ctx.workdir / 'finals' / f'{cid}.mp4', proxy=False,
                                 duration_range=allowed)
        except RenderError as exc:
            raise GateFailure('S12', str(exc)) from exc
        rendered.append({**clip, **result})
    return {'rendered': rendered}
