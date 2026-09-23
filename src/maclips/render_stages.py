"""S8 and S12: render checkpointed candidates from the original streams."""
from pathlib import Path
import time
from .orchestrator import GateFailure


def proxy_render(ctx):
    from .render import RenderError, render_clip
    media = ctx.output('S6')
    words = ctx.output('S3')['words']
    previews = {}
    started = time.perf_counter()
    for candidate in ctx.output('S5')['candidates']:
        cid = str(candidate['rank'])
        previews[cid] = {}
        for name, layout in ctx.output('S7')['plans'][cid].items():
            path = ctx.workdir / 'previews' / f'{cid}-{name}.mp4'
            try:
                result = render_clip(Path(media['video_path']), Path(media['audio_path']),
                                     candidate, words, layout, path, proxy=True)
            except RenderError as exc:
                raise GateFailure('S8', str(exc)) from exc
            previews[cid][name] = result
            callback = ctx.shared.get('on_preview')
            if callback:
                callback(cid, name, result)
    return {'previews': previews, 'wall_s': time.perf_counter() - started}


def final_render(ctx):
    from .render import RenderError, render_clip
    media = ctx.output('S6')
    rendered = []
    for clip in ctx.output('S9')['approved']:
        cid = str(clip['rank'])
        name = clip['layout']
        plans = ctx.output('S7')['plans'].get(cid, {})
        if name not in plans:
            raise GateFailure('S12', f'clip {cid}: unavailable layout {name}')
        try:
            result = render_clip(Path(media['video_path']), Path(media['audio_path']),
                                 clip, ctx.output('S3')['words'], plans[name],
                                 ctx.workdir / 'finals' / f'{cid}.mp4', proxy=False)
        except RenderError as exc:
            raise GateFailure('S12', str(exc)) from exc
        rendered.append({**clip, **result})
    return {'rendered': rendered}
