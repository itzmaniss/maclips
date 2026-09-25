"""Local server-rendered review UI; one serial worker owns all heavy jobs."""
from __future__ import annotations
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from . import config, db
from .compliance import ComplianceError, validate_clip, human_checklist
from .brief import BriefInvalid, BriefRejected

app=FastAPI(title='maclips')
ROOT=Path(__file__).parent
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')
templates=Jinja2Templates(directory=ROOT/'templates')
CSRF=secrets.token_urlsafe(32)
WORKER=ThreadPoolExecutor(max_workers=1,thread_name_prefix='maclips')
JOBS={}
JOB_LOCK=Lock()


@app.middleware('http')
async def local_only(request,call_next):
    host=request.headers.get('host','').split(':')[0]
    if host not in {'127.0.0.1','localhost','testserver'}:
        return JSONResponse({'error':'local host only'},status_code=403)
    if request.method not in {'GET','HEAD'} and request.headers.get('x-csrf-token') != CSRF:
        return JSONResponse({'error':'invalid local session token'},status_code=403)
    return await call_next(request)


@app.exception_handler(BriefInvalid)
@app.exception_handler(BriefRejected)
@app.exception_handler(ValueError)
@app.exception_handler(ComplianceError)
async def bad_input(request,exc):
    return JSONResponse({'error':str(exc)},status_code=400)


def source_row(conn,sid):
    row=conn.execute('SELECT * FROM sources WHERE id=?',(sid,)).fetchone()
    if not row: raise HTTPException(404,'source not found')
    return {**dict(row),'state':json.loads(row['state_json'])}


def clip_row(conn,cid):
    row=conn.execute('SELECT * FROM clips WHERE id=?',(cid,)).fetchone()
    if not row: raise HTTPException(404,'clip not found')
    return {**dict(row),'data':json.loads(row['data_json'])}


def job(fn,*args):
    jid=secrets.token_hex(6)
    with JOB_LOCK: JOBS[jid]={'status':'queued','error':None}
    def run():
        with JOB_LOCK: JOBS[jid]['status']='running'
        try:
            result=fn(*args)
            with JOB_LOCK: JOBS[jid].update(status='complete',result=result)
        except Exception as exc:
            with JOB_LOCK: JOBS[jid].update(status='gated',error=str(exc))
    WORKER.submit(run)
    return {'job_id':jid}


@app.get('/jobs')
def jobs():
    with JOB_LOCK: return dict(JOBS)


@app.get('/')
def index(request:Request,tab:str='ingest',source:int|None=None,clip:int|None=None):
    if tab not in {'ingest','review','posted'}: raise HTTPException(404)
    with db.connect(config.DB_PATH) as conn:
        sources=[{**dict(r),'state':json.loads(r['state_json'])} for r in conn.execute('SELECT * FROM sources ORDER BY id DESC')]
        campaigns=[dict(r) for r in conn.execute('SELECT * FROM campaigns ORDER BY id DESC')]
        selected=source_row(conn,source) if source else (sources[0] if sources else None)
        clips=[];chosen=None;words=[];brief={}
        if selected:
            active=selected['state'].get('active_keys')
            clips=[{**dict(r),'data':json.loads(r['data_json'])} for r in conn.execute('SELECT * FROM clips WHERE source_id=? ORDER BY id',(selected['id'],)) if active is None or r['candidate_key'] in active]
            chosen=next((c for c in clips if c['id']==clip),clips[0] if clips else None)
            brief=selected['state'].get('brief',{})
            if chosen:
                allwords=json.loads(Path(selected['state']['transcript']).read_text())['words']
                a,b=chosen['data']['start']-10,chosen['data']['end']+10
                words=[{**w,'index':i} for i,w in enumerate(allwords) if w.get('start') is not None and w.get('end') is not None and w['end']>=a and w['start']<=b]
        mechanical = "Choose a candidate to check."
        if chosen:
            check=dict(chosen['data'])
            check['transcript_text']=' '.join(w['word'] for w in words if check['start_word']<=w['index']<=check['end_word'])
            try:
                validate_clip(check,brief,selected['clip_class'])
                mechanical="Mechanical checks pass for the current posting details."
            except ComplianceError as exc:
                mechanical=f"Blocked: {exc}"
        timing={'review_elapsed_min':None,'minutes_per_approved':None,'approved':0,'post_submit_min':None}
        if selected:
            events=conn.execute('SELECT event,at FROM time_log WHERE source_id=?',(selected['id'],)).fetchall()
            first=[r['at'] for r in events if r['event']=='first_preview']
            decisions=[r['at'] for r in events if r['event'] in {'clip_approved','clip_rejected','trim_edit'}]
            timing['approved']=sum(c['review_decision']=='approved' for c in clips)
            if first and decisions:
                timing['review_elapsed_min']=round((max(decisions)-min(first))/60,2)
                if timing['approved']: timing['minutes_per_approved']=round(timing['review_elapsed_min']/timing['approved'],2)
        posts=[]
        for row in conn.execute('SELECT posts.*,clips.source_id FROM posts JOIN clips ON clips.id=posts.clip_id ORDER BY posts.id DESC'):
            item=dict(row);src=source_row(conn,row['source_id']);window=src['state'].get('brief',{}).get('submission_window_min')
            item['campaign_name']=src['state'].get('brief',{}).get('campaign_name') or 'Own content'
            item['deadline_at']=row['posted_at']+window*60 if row['posted_at'] and window else None
            posts.append(item)
    return templates.TemplateResponse(request=request,name='app.html',context={'tab':tab,'sources':sources,'campaigns':campaigns,'source':selected,'clips':clips,'clip':chosen,'words':words,'brief':brief,'checklist':human_checklist(brief),'mechanical':mechanical,'end_card_default':config.END_CARD_TEXT,'end_card_max':config.END_CARD_MAX_CHARS,'end_card_seconds':config.END_CARD_SECONDS,'timing':timing,'posts':posts,'preview_count':sum(len(c['data'].get('previews',{})) for c in clips),'csrf':CSRF,'jobs':JOBS})


@app.get('/media/{cid}/{layout}')
def media(cid:int,layout:str):
    with db.connect(config.DB_PATH) as conn: clip=clip_row(conn,cid)
    preview=clip['data'].get('previews',{}).get(layout)
    if not preview or not Path(preview['path']).is_file(): raise HTTPException(404,'preview not ready')
    return FileResponse(preview['path'],media_type='video/mp4')


@app.post('/brief/extract')
async def extract_brief(request:Request):
    payload=await request.json()
    raw=str(payload.get('raw','')).strip()
    if not raw: raise ValueError('paste a brief')
    # User-triggered extraction only. No model call happens on page load.
    from .brief import extract
    import asyncio
    cfg=await asyncio.get_running_loop().run_in_executor(WORKER,extract,raw)
    return {'config':cfg.as_dict()}


@app.post('/brief/confirm')
async def confirm_brief(request:Request):
    from .brief import CampaignConfig
    payload=await request.json();cfg=CampaignConfig.from_dict(payload['config'])
    cfg.confirmed=False;cfg.check_category();cfg.confirm()
    with db.connect(config.DB_PATH) as conn:
        cid=conn.execute('INSERT INTO campaigns(brand,campaign_name,category,config_json,raw_brief,confirmed_at) VALUES(?,?,?,?,?,?)',
                         (cfg.brand,cfg.campaign_name,cfg.category,json.dumps(cfg.as_dict()),cfg.raw_brief,time.time())).lastrowid
        db.log_event(conn,'brief_confirmed',campaign_id=cid)
    return {'campaign_id':cid}


def run_source(payload,from_stage=None):
    import argparse
    from .cli import _cmd_run
    from .orchestrator import GateFailure
    campaign_id=payload.get('campaign_id') or None
    brief={}
    if payload['clip_class']=='campaign':
        with db.connect(config.DB_PATH) as conn:
            row=conn.execute('SELECT config_json FROM campaigns WHERE id=?',(campaign_id,)).fetchone()
        if not row: raise ValueError('select a confirmed campaign')
        brief=json.loads(row[0])
    args=argparse.Namespace(source=payload['source'],max_height=1440,clip_class=payload['clip_class'],
       from_stage=from_stage,expected_speakers=payload.get('expected_speakers'),language=payload.get('language') or None,
       candidates=int(payload.get('candidates') or 12),full_diarization=False,brief=brief,campaign_id=campaign_id,
       clip_min_duration=payload.get('min_duration_s'),clip_max_duration=payload.get('max_duration_s'),
       length_preset=payload.get('length_preset') or 'default',on_stage=None)
    status=_cmd_run(args)
    if status: raise ValueError('Pipeline stopped; see the source stage status and gate reason.')
    return {'status':'complete'}


@app.post('/ingest')
async def ingest(request:Request):
    payload=await request.json()
    if payload.get('clip_class') not in {'campaign','general-own'}: raise ValueError('unsupported clip class')
    if not str(payload.get('source','')).strip(): raise ValueError('source is required')
    count=int(payload.get('candidates') or 12)
    if not 5<=count<=30: raise ValueError('candidate count must be 5–30')
    low=float(payload.get('min_duration_s') or 10); high=float(payload.get('max_duration_s') or 180)
    if not 0<low<=high: raise ValueError('invalid clip duration range')
    from .ranking import LENGTH_PRESETS
    if (payload.get('length_preset') or 'default') not in LENGTH_PRESETS: raise ValueError('unknown length preset')
    speakers=payload.get('expected_speakers')
    if speakers is not None and (not isinstance(speakers,int) or speakers<1): raise ValueError('invalid expected speaker count')
    return job(run_source,payload)


@app.post('/rerun/{sid}')
async def rerun(sid:int,request:Request):
    from .stages import STAGES_BY_ID
    payload=await request.json();stage=payload.get('stage')
    if stage not in STAGES_BY_ID: raise ValueError('unknown stage')
    with db.connect(config.DB_PATH) as conn: source=source_row(conn,sid)
    state=source['state']
    if state.get('experiment_only'): raise ValueError('Imported experiment: use preview edits; pipeline reruns require a real ingest.')
    cfg=state['config']
    return job(run_source,{'source':state.get('origin') or source['path'],'clip_class':source['clip_class'],
        'campaign_id':source['campaign_id'],'language':cfg.get('expected_language'),'expected_speakers':cfg.get('expected_speaker_count'),
        'candidates':cfg.get('candidate_count',12),
        # The resolved range (preset or explicit) reruns unchanged, so S5 stays cached.
        'min_duration_s':cfg.get('clip_min_duration'),'max_duration_s':cfg.get('clip_max_duration')},stage)


def edit_clip(cid,payload):
    from .render import render_clip
    with db.connect(config.DB_PATH) as conn:
        clip=clip_row(conn,cid);source=source_row(conn,clip['source_id'])
        if conn.execute('SELECT 1 FROM posts WHERE clip_id=? AND post_url IS NOT NULL',(cid,)).fetchone():
            raise ValueError('posted clip is immutable')
    state=source['state'];words=json.loads(Path(state['transcript']).read_text())['words'];data=clip['data']
    if payload.get('revision',data.get('revision',0)) != data.get('revision',0): raise ValueError('clip changed; reload before editing')
    start=int(payload.get('start_word',data['start_word']));end=int(payload.get('end_word',data['end_word']))
    if not 0<=start<=end<len(words): raise ValueError('word bounds out of range')
    a,b=words[start].get('start'),words[end].get('end')
    if a is None or b is None or b<=a: raise ValueError('selected boundaries have no usable aligned timing')
    card=payload.get('end_card',data.get('end_card',False))
    if not isinstance(card,bool): raise ValueError('end card toggle must be true or false')
    card_text=str(payload.get('end_card_text',data.get('end_card_text') or config.END_CARD_TEXT)).strip()
    if not 0<len(card_text)<=config.END_CARD_MAX_CHARS: raise ValueError(f'end card text must be 1–{config.END_CARD_MAX_CHARS} characters')
    data.update(start_word=start,end_word=end,start=a,end=b,duration=b-a,hook_text=str(payload.get('hook_text',data.get('hook_text',''))),
                end_card=card,end_card_text=card_text)
    for toggle in ('dead_air','zoom'):
        if toggle in payload:
            if not isinstance(payload[toggle],bool): raise ValueError(f'{toggle} must be true or false')
            data[toggle]=payload[toggle]
    layout=payload.get('layout') or data.get('layout') or next(iter(data['previews']))
    plans=state['plans'][str(data['rank'])]
    if layout not in plans: raise ValueError('layout unavailable')
    segments=plans[layout].get('segments')
    if segments and (a<segments[0]['start'] or b>segments[-1]['end']):
        from .vision import analyze_window
        from .layouts import plan_layouts
        analysis=analyze_window(state['video'],a,b,Path(state['workdir'])/'face-checks'/f"{data['rank']}-edit-{data.get('revision',0)+1}")
        changes=next(iter(state['plans'][str(data['rank'])].values()),{}).get('speaker_changes',[])
        plans=plan_layouts({str(data['rank']):analysis})[str(data['rank'])]
        for plan in plans.values(): plan['speaker_changes']=changes
        if layout not in plans:
            raise ValueError('new span has different face availability; choose letterbox and retry')
        state['plans'][str(data['rank'])]=plans
    # Re-render next to the clip's current preview, so an edit never overwrites
    # an older preview folder kept for comparison.
    current=next((p['path'] for p in data.get('previews',{}).values() if p.get('path')),None)
    folder=Path(current).parent if current else Path(state['workdir'])/config.PREVIEW_DIR
    path=folder/f"{data['rank']}-{layout}.mp4"
    result=render_clip(Path(state['video']),Path(state['audio']),data,words,plans[layout],path,proxy=True)
    # Other layouts now refer to stale boundaries; they are regenerated on demand.
    data['previews']={layout:result};data['layout']=layout;data['revision']=data.get('revision',0)+1
    with db.connect(config.DB_PATH) as conn:
        conn.execute('UPDATE sources SET state_json=? WHERE id=?',(json.dumps(state),source['id']))
        conn.execute('UPDATE clips SET start_word=?,end_word=?,start_s=?,end_s=?,hook_text=?,layout=?,data_json=?,review_decision=NULL,review_reason=NULL,render_path=NULL WHERE id=?',
                     (start,end,a,b,data['hook_text'],layout,json.dumps(data),cid))
        conn.execute("DELETE FROM posts WHERE clip_id=? AND status='draft'",(cid,))
        db.log_event(conn,'trim_edit',source['id'],cid,source['campaign_id'])
    return {'clip_id':cid,'preview':result}


@app.post('/clips/{cid}/edit')
async def edit(cid:int,request:Request):
    return job(edit_clip,cid,await request.json())


def finish_clip(cid):
    from .orchestrator import RunContext,run_pipeline
    from .stages import STAGES_BY_ID
    with db.connect(config.DB_PATH) as conn:
        clip=clip_row(conn,cid);source=source_row(conn,clip['source_id'])
    if clip['review_decision']!='approved' or source['state'].get('experiment_only'):
        raise ValueError('final rendering requires current human approval on a non-experiment source')
    state=source['state'];data=clip['data'];words=json.loads(Path(state['transcript']).read_text())['words']
    ctx=RunContext('review-final',Path(state['audio']),source['content_hash'],Path(state['workdir']),state['config'],outputs={
        'S0':{'clip_class':source['clip_class']},'S1':{'brief':state.get('brief',{})},'S3':{'words':words},
        'S6':{'video_path':state['video'],'audio_path':state['audio']},'S7':{'plans':state['plans']},'S9':{'approved':[{**data,'id':cid}]}})
    report=run_pipeline(ctx,[STAGES_BY_ID[s] for s in ('S10','S11','S12','S13')],from_stage='S10')
    if report.gated: raise ValueError(report.gated.reason)
    return ctx.output('S13')


@app.post('/clips/{cid}/decision')
async def decision(cid:int,request:Request):
    return job(apply_decision,cid,await request.json())


def apply_decision(cid,payload):
    choice=payload.get('decision')
    if choice not in {'approved','rejected'}: raise ValueError('invalid decision')
    with db.connect(config.DB_PATH) as conn:
        clip=clip_row(conn,cid);source=source_row(conn,clip['source_id']);data=clip['data']
        if conn.execute('SELECT 1 FROM posts WHERE clip_id=? AND post_url IS NOT NULL',(cid,)).fetchone():
            raise ValueError('posted clip is immutable')
        if payload.get('revision',data.get('revision',0)) != data.get('revision',0): raise ValueError('clip changed; reload before deciding')
        if choice=='approved':
            if source['state'].get('experiment_only'): raise ValueError('experiment-only source cannot be approved for export')
            for field in ('platform','account','account_class','caption','layout','hook_text'):
                if field in payload: data[field]=payload[field]
            if not str(data.get('account','')).strip(): raise ValueError('posting account is required')
            if data.get('layout') not in source['state']['plans'][str(data['rank'])]: raise ValueError('layout unavailable')
            words=json.loads(Path(source['state']['transcript']).read_text())['words']
            data['transcript_text']=' '.join(w['word'] for w in words[data['start_word']:data['end_word']+1])
            validate_clip(data,source['state'].get('brief',{}),source['clip_class'])
        elif payload.get('reason') not in {'off-brief','weak hook','bad framing','needs context','other'}:
            raise ValueError('choose a rejection reason')
        conn.execute("DELETE FROM posts WHERE clip_id=? AND status='draft'",(cid,))
        conn.execute('UPDATE clips SET render_path=NULL WHERE id=?',(cid,))
        conn.execute('UPDATE clips SET data_json=?,review_decision=?,review_reason=? WHERE id=?',(json.dumps(data),choice,payload.get('reason'),cid))
        db.log_event(conn,'clip_'+choice,source['id'],cid,source['campaign_id'])
        # Max timestamp of this event is the last approval for the source.
        if choice=='approved': db.log_event(conn,'last_clip_approved',source['id'],cid,source['campaign_id'])
    return finish_clip(cid) if choice=='approved' else {'status':'rejected'}


@app.post('/posts/{pid}')
async def post(pid:int,request:Request):
    return job(save_post,pid,await request.json())


def save_post(pid,payload):
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT * FROM posts WHERE id=?',(pid,)).fetchone()
        if not row: raise HTTPException(404)
        db.record_post(conn,row['clip_id'],row['platform'],row['account'],payload)
    return {'status':'saved'}


@app.post('/bundles/{pid}/open')
def open_bundle(pid:int):
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT posts.*,clips.source_id,clips.review_decision FROM posts JOIN clips ON clips.id=posts.clip_id WHERE posts.id=?',(pid,)).fetchone()
        if not row or not row['bundle_path'] or row['review_decision']!='approved': raise HTTPException(404)
        if source_row(conn,row['source_id'])['state'].get('experiment_only'): raise HTTPException(403)
        db.log_event(conn,'bundle_opened',row['source_id'],row['clip_id'])
    return {'url':f'/bundles/{pid}/clip.mp4','files':[{'name':f,'url':f'/bundles/{pid}/{f}'} for f in ('clip.mp4','caption.txt','checklist.md')]}


@app.get('/bundles/{pid}/{filename}')
def bundle_file(pid:int,filename:str):
    if filename not in {'clip.mp4','caption.txt','checklist.md'}: raise HTTPException(404)
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT posts.bundle_path,clips.source_id,clips.review_decision FROM posts JOIN clips ON clips.id=posts.clip_id WHERE posts.id=?',(pid,)).fetchone()
        if not row or not row['bundle_path'] or row['review_decision']!='approved': raise HTTPException(404)
        if source_row(conn,row['source_id'])['state'].get('experiment_only'): raise HTTPException(403)
    return FileResponse(Path(row['bundle_path'])/filename,filename=filename)


@app.get('/progress')
def progress():
    with db.connect(config.DB_PATH) as conn:
        return {'sources':[{'id':r['id'],'progress':json.loads(r['state_json']).get('progress',{}), 'preview_count':conn.execute("SELECT count(*) FROM clips,json_each(clips.data_json,'$.previews') WHERE source_id=?",(r['id'],)).fetchone()[0]}
                           for r in conn.execute('SELECT id,state_json FROM sources')]}
