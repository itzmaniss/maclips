"""S11/S13/S14: hard compliance, manual-post bundles and tracking."""
import json
import shutil
from pathlib import Path
from . import config, db
from .compliance import ComplianceError, validate_clip, human_checklist
from .orchestrator import GateFailure
from .render import end_card_text


def compliance_stage(ctx):
    if ctx.config.get('experiment_only'):
        raise GateFailure('S11','experiment-only source cannot be exported or posted')
    brief=ctx.output('S1').get('brief',{})
    words=ctx.output('S3')['words']
    for clip in ctx.output('S9')['approved']:
        clip['transcript_text']=' '.join(w['word'] for w in words[clip['start_word']:clip['end_word']+1])
        try:
            validate_clip(clip,brief,ctx.output('S0')['clip_class'])
        except ComplianceError as exc:
            raise GateFailure('S11',f"clip {clip['rank']}: {exc}") from exc
    return {'passed':True,'human_checklist':human_checklist(brief)}


def export_stage(ctx):
    if ctx.config.get('experiment_only'):
        raise GateFailure('S13','experiment-only source cannot be exported')
    brief=ctx.output('S1').get('brief',{})
    bundles=[]
    for clip in ctx.output('S12')['rendered']:
        platform=clip['platform']
        if platform not in {'instagram','tiktok','youtube'}:
            raise GateFailure('S13',f'unsupported platform {platform!r}')
        if clip.get('diagnostic'):
            raise GateFailure('S13','diagnostic render cannot be exported')
        cid=int(clip['id'])
        with db.connect(config.DB_PATH) as conn:
            approved=conn.execute('SELECT * FROM clips WHERE id=?',(cid,)).fetchone()
            if not approved or approved['review_decision']!='approved':
                raise GateFailure('S13','export requires the persisted human approval')
        target=ctx.workdir/'exports'/str(cid)/platform
        target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(clip['path'],target/'clip.mp4')
        (target/'caption.txt').write_text(clip['caption'].strip()+'\n')
        window=brief.get('submission_window_min')
        checklist=['# Posting checklist','',f'- Platform: {platform}',f"- Account: {clip['account']}",
                   '- [ ] Review the video and caption before posting.']
        if ctx.output('S0')['clip_class']=='campaign':
            checklist += ['- [ ] Enable the platform paid-partnership / paid-promotion toggle.',
                          f'- [ ] Submit the post link within {window} minutes after posting.' if window else '- [ ] Check the original brief for the submission deadline.']
        card=end_card_text(clip)
        if card:
            checklist.append(f'- End card burned into the last {config.END_CARD_SECONDS:g} s: "{card}"')
        checklist += ['- [ ] '+item for item in human_checklist(brief)]
        (target/'checklist.md').write_text('\n'.join(checklist)+'\n')
        with db.connect(config.DB_PATH) as conn:
            row=conn.execute('SELECT * FROM clips WHERE id=?',(cid,)).fetchone()
            if not row or row['review_decision']!='approved':
                raise GateFailure('S13','export requires the persisted human approval')
            conn.execute('UPDATE clips SET render_path=? WHERE id=?',(clip['path'],cid))
            old=conn.execute('SELECT id FROM posts WHERE clip_id=? AND platform=? AND account=?',(cid,platform,clip['account'])).fetchone()
            if old:
                pid=old['id'];conn.execute('UPDATE posts SET bundle_path=? WHERE id=?',(str(target),pid))
            else:
                pid=conn.execute("INSERT INTO posts(clip_id,platform,account,status,bundle_path) VALUES(?,?,?,'draft',?)",(cid,platform,clip['account'],str(target))).lastrowid
            db.log_event(conn,'bundle_ready',row['source_id'],cid)
        bundles.append({'clip_id':cid,'post_id':pid,'platform':platform,'path':str(target)})
    return {'bundles':bundles}


def post_stage(ctx):
    tracked=[]
    for row in ctx.config.get('post_rows',[]) or []:
        try:
            with db.connect(config.DB_PATH) as conn:
                tracked.append(db.record_post(conn,int(row['clip_id']),row['platform'],row['account'],row))
        except (ValueError,ComplianceError) as exc:
            raise GateFailure('S14',str(exc)) from exc
    return {'tracked':tracked}
