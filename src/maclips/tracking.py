"""Persist real source/candidate artifacts and timestamp review operations."""
import json
from pathlib import Path
from . import config, db


def register_context(ctx):
    media=ctx.output('S6')
    transcript=ctx.workdir/'transcript.json'
    transcript.parent.mkdir(parents=True,exist_ok=True)
    transcript.write_text(json.dumps({'words':ctx.output('S3')['words']}))
    state={'video':media['video_path'],'audio':media['audio_path'],
           'transcript':str(transcript.resolve()),'workdir':str(ctx.workdir.resolve()),
           'config':dict(ctx.config),'plans':ctx.output('S7')['plans'],
           'experiment_only':bool(ctx.config.get('experiment_only')),
           'brief':ctx.outputs.get('S1',{}).get('brief',{}),
           'active_keys':[candidate_key(c) for c in ctx.output('S5')['candidates']]}
    from .orchestrator import hash_file
    with db.connect(config.DB_PATH) as conn:
        sid=db.register_source(conn,hash_file(Path(media['audio_path'])),media['audio_path'],
                               ctx.config.get('clip_class','general-own'),ctx.config.get('campaign_id'),state=state)
        for c in ctx.output('S5')['candidates']:
            db.save_candidate(conn,sid,ctx.config.get('clip_class','general-own'),c)
    ctx.shared['source_id']=sid
    return sid


def candidate_key(candidate):
    return f"{candidate['rank']}:{candidate.get('start_word')}:{candidate.get('end_word')}"


def record_preview(ctx,candidate,layout,result):
    sid=ctx.shared.get('source_id') or register_context(ctx)
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT * FROM clips WHERE source_id=? AND candidate_key=?',
                         (sid,candidate_key(candidate))).fetchone()
        data=json.loads(row['data_json']); data.setdefault('previews',{})[layout]=result
        data.setdefault('layout',layout)
        conn.execute('UPDATE clips SET data_json=?,layout=? WHERE id=?',(json.dumps(data),data['layout'],row['id']))
        if not conn.execute("SELECT 1 FROM time_log WHERE source_id=? AND event='first_preview'",(sid,)).fetchone():
            db.log_event(conn,'first_preview',sid,row['id'])


def import_previews(ctx):
    register_context(ctx)
    for c in ctx.output('S5')['candidates']:
        for name,result in ctx.output('S8')['previews'].get(str(c['rank']),{}).items():
            record_preview(ctx,c,name,result)


def approved_clips(ctx):
    with db.connect(config.DB_PATH) as conn:
        source=conn.execute('SELECT * FROM sources WHERE content_hash=?',(ctx.source_hash,)).fetchone()
        if not source: return []
        active={candidate_key(c) for c in ctx.output('S5')['candidates']}
        return [{**json.loads(row['data_json']),'id':row['id']} for row in
                conn.execute("SELECT * FROM clips WHERE source_id=? AND review_decision='approved'",(source['id'],))
                if row['candidate_key'] in active]


def effective_candidate(ctx,candidate):
    sid=ctx.shared.get('source_id') or register_context(ctx)
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT data_json FROM clips WHERE source_id=? AND candidate_key=?',(sid,candidate_key(candidate))).fetchone()
    return json.loads(row[0]) if row else candidate
