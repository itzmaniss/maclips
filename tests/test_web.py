import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from maclips import config,db,web


@pytest.fixture
def studio(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.db')
    words=[{'word':f'w{i}','start':i,'end':i+1} for i in range(40)]
    transcript=tmp_path/'transcript.json';transcript.write_text(json.dumps({'words':words}))
    media=tmp_path/'fixture.mp4';media.write_bytes(b'fixture')
    state={'audio':str(media),'video':str(media),'transcript':str(transcript),'workdir':str(tmp_path),
       'experiment_only':False,'brief':{},'config':{'clip_class':'general-own'},'plans':{'1':{'centre':{'kind':'centre'},'letterbox':{'kind':'letterbox'}}}}
    with db.connect(config.DB_PATH) as conn:
        sid=db.register_source(conn,'fixture',media,'general-own',state=state)
        cid=db.save_candidate(conn,sid,'general-own',{'rank':1,'start_word':0,'end_word':29,'start':0,'end':30,'hook_text':'Hook'},
            {'centre':{'path':str(media)},'letterbox':{'path':str(media)}})
    monkeypatch.setattr(web,'job',lambda fn,*args:{'result':fn(*args)})
    client=TestClient(web.app)
    return client,{'X-CSRF-Token':web.CSRF},sid,cid


def test_tabs_and_cross_origin_mutation_gate(studio):
    client,headers,sid,cid=studio
    for tab in ('ingest','review','posted'):
        assert client.get('/',params={'tab':tab,'source':sid}).status_code==200
    assert client.post(f'/clips/{cid}/decision',json={'decision':'approved'}).status_code==403
    assert client.get('/',headers={'Host':'evil.example'}).status_code==403


def test_experiment_cannot_approve(studio):
    client,headers,sid,cid=studio
    with db.connect(config.DB_PATH) as conn:
        state=json.loads(conn.execute('SELECT state_json FROM sources WHERE id=?',(sid,)).fetchone()[0]);state['experiment_only']=True
        conn.execute('UPDATE sources SET state_json=? WHERE id=?',(json.dumps(state),sid))
    response=client.post(f'/clips/{cid}/decision',headers=headers,json={'decision':'approved'})
    assert response.status_code==400 and 'experiment-only' in response.text
    with db.connect(config.DB_PATH) as conn:
        assert conn.execute('SELECT review_decision FROM clips WHERE id=?',(cid,)).fetchone()[0] is None


def test_fixture_trim_renders_only_selected_preview_and_invalidates_approval(studio,monkeypatch):
    client,headers,sid,cid=studio
    from maclips import render
    calls=[]
    def fake(video,audio,candidate,words,layout,path,**kw):
        calls.append((candidate['start'],candidate['end'],layout))
        return {'path':str(path),'wall_s':0,'width':540,'height':960}
    monkeypatch.setattr(render,'render_clip',fake)
    web.edit_clip(cid,{'start_word':1,'end_word':28,'layout':'centre'})
    assert calls==[(1,29,{'kind':'centre'})]
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT * FROM clips WHERE id=?',(cid,)).fetchone()
        assert list(json.loads(row['data_json'])['previews'])==['centre']
        assert row['review_decision'] is None
        assert conn.execute("SELECT count(*) FROM time_log WHERE event='trim_edit'").fetchone()[0]==1


def test_fixture_approval_is_explicit_and_enqueues_final(studio,monkeypatch):
    client,headers,sid,cid=studio
    calls=[]
    monkeypatch.setattr(web,'job',lambda fn,*args:calls.append((fn,args)) or {'job_id':'fixture'})
    response=client.post(f'/clips/{cid}/decision',headers=headers,json={'decision':'approved','platform':'instagram',
        'account':'fixture','account_class':'general','caption':'test','layout':'centre'})
    assert response.status_code==200
    assert calls[0][0] is web.apply_decision
    monkeypatch.setattr(web,'finish_clip',lambda cid:{'fixture':cid})
    calls[0][0](*calls[0][1])
    with db.connect(config.DB_PATH) as conn:
        assert conn.execute('SELECT review_decision FROM clips WHERE id=?',(cid,)).fetchone()[0]=='approved'


def test_ingest_rejects_invalid_range_without_job(studio,monkeypatch):
    client,headers,_,_=studio
    monkeypatch.setattr(web,'job',lambda *args:pytest.fail('invalid ingest queued'))
    response=client.post('/ingest',headers=headers,json={'source':'fixture','clip_class':'general-own','min_duration_s':40,'max_duration_s':10})
    assert response.status_code==400


def test_fixture_approval_through_final_bundle_and_post(studio,monkeypatch):
    client,headers,sid,cid=studio
    from maclips import render
    def fake(video,audio,candidate,words,layout,path,**kw):
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(b'fixture final')
        return {'path':str(path),'layout':'centre','diagnostic':False,'width':1080,'height':1920,'has_audio':True}
    monkeypatch.setattr(render,'render_clip',fake)
    monkeypatch.setattr(web,'job',lambda fn,*args:{'result':fn(*args)})
    response=client.post(f'/clips/{cid}/decision',headers=headers,json={'decision':'approved','platform':'instagram',
        'account':'fixture','account_class':'general','caption':'test','layout':'centre'})
    assert response.status_code==200,response.text
    with db.connect(config.DB_PATH) as conn:
        post=dict(conn.execute('SELECT * FROM posts WHERE clip_id=?',(cid,)).fetchone())
    assert (Path(post['bundle_path'])/'caption.txt').read_text()=='test\n'
    response=client.post(f"/posts/{post['id']}",headers=headers,json={'post_url':'https://example.test/fixture','status':'submitted','disclosure_ticked':False})
    assert response.status_code==200,response.text
    assert client.post(f'/clips/{cid}/decision',headers=headers,json={'decision':'rejected','reason':'other'}).status_code==400


def test_reject_revokes_draft_bundle_download(studio):
    client,headers,sid,cid=studio
    with db.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE clips SET review_decision='approved',render_path='fixture' WHERE id=?",(cid,))
        pid=conn.execute("INSERT INTO posts(clip_id,platform,account,status,bundle_path) VALUES(?,'instagram','test','draft','fixture')",(cid,)).lastrowid
    response=client.post(f'/clips/{cid}/decision',headers=headers,json={'decision':'rejected','reason':'other'})
    assert response.status_code==200
    assert client.get(f'/bundles/{pid}/clip.mp4').status_code==404
    with db.connect(config.DB_PATH) as conn:
        assert conn.execute('SELECT render_path FROM clips WHERE id=?',(cid,)).fetchone()[0] is None


def test_stale_decision_does_not_approve(studio):
    client,headers,sid,cid=studio
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()
        data=json.loads(row[0]);data['revision']=1
        conn.execute('UPDATE clips SET data_json=? WHERE id=?',(json.dumps(data),cid))
    response=client.post(f'/clips/{cid}/decision',headers=headers,json={'decision':'approved','revision':0})
    assert response.status_code==400 and 'reload' in response.text


def test_pacing_toggle_rerenders_only_that_preview_and_is_saved(studio,monkeypatch):
    client,headers,sid,cid=studio
    from maclips import render
    calls=[]
    def fake(video,audio,candidate,words,layout,path,**kw):
        calls.append((candidate.get('dead_air',True),candidate.get('zoom',True),Path(path).parent))
        return {'path':str(path),'wall_s':0,'width':540,'height':960}
    monkeypatch.setattr(render,'render_clip',fake)
    web.edit_clip(cid,{'layout':'centre','dead_air':False,'zoom':True})
    web.edit_clip(cid,{'layout':'centre','zoom':False,'revision':1})
    # Toggles persist per clip, and the preview is re-rendered in its existing folder.
    assert [c[:2] for c in calls]==[(False,True),(False,False)]
    assert calls[0][2]==config.DB_PATH.parent
    with db.connect(config.DB_PATH) as conn:
        data=json.loads(conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()[0])
    assert (data['dead_air'],data['zoom'],list(data['previews']))==(False,False,['centre'])
    with pytest.raises(ValueError,match='dead_air must be true or false'):
        web.edit_clip(cid,{'layout':'centre','dead_air':'no','revision':2})
    assert client.get('/',params={'tab':'review','source':sid}).status_code==200


def fake_render(calls):
    def fake(video,audio,candidate,words,layout,path,**kw):
        calls.append(dict(candidate))
        return {'path':str(path),'wall_s':0,'width':540,'height':960}
    return fake


def test_end_card_off_by_default_in_review(studio):
    import re
    client,headers,sid,cid=studio
    page=client.get('/',params={'tab':'review','source':sid,'clip':cid}).text
    box=re.search(r'<input id="end-card" type="checkbox"[^>]*>',page).group(0)
    assert 'checked' not in box
    assert f'value="{config.END_CARD_TEXT}"' in page
    with db.connect(config.DB_PATH) as conn:
        assert 'end_card' not in json.loads(conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()[0])


def test_end_card_toggle_and_text_persist_and_rerender_one_preview(studio,monkeypatch):
    client,headers,sid,cid=studio
    from maclips import render
    calls=[];monkeypatch.setattr(render,'render_clip',fake_render(calls))
    web.edit_clip(cid,{'layout':'centre','end_card':True,'end_card_text':'  Share this one  '})
    assert len(calls)==1 and calls[0]['end_card'] is True and calls[0]['end_card_text']=='Share this one'
    web.edit_clip(cid,{'start_word':2,'end_word':28,'layout':'centre','revision':1})  # a trim keeps the card
    assert calls[1]['end_card'] is True and calls[1]['end_card_text']=='Share this one'
    with db.connect(config.DB_PATH) as conn:
        data=json.loads(conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()[0])
    assert (data['end_card'],data['end_card_text'],list(data['previews']))==(True,'Share this one',['centre'])
    page=client.get('/',params={'tab':'review','source':sid,'clip':cid}).text
    assert 'type="checkbox" checked' in page and 'value="Share this one"' in page
    web.edit_clip(cid,{'layout':'centre','end_card':False,'revision':2})
    assert calls[2]['end_card'] is False and len(calls)==3


@pytest.mark.parametrize('payload,message',[({'end_card':'true'},'true or false'),
    ({'end_card':True,'end_card_text':'  '},'end card text'),({'end_card':True,'end_card_text':'x'*61},'end card text')])
def test_end_card_edit_input_is_validated_before_render(studio,monkeypatch,payload,message):
    client,headers,sid,cid=studio
    from maclips import render
    monkeypatch.setattr(render,'render_clip',lambda *a,**kw:pytest.fail('rendered invalid end card'))
    response=client.post(f'/clips/{cid}/edit',headers=headers,json={'layout':'centre',**payload})
    assert response.status_code==400 and message in response.text


def test_end_card_change_revokes_approval(studio,monkeypatch):
    client,headers,sid,cid=studio
    from maclips import render
    monkeypatch.setattr(render,'render_clip',fake_render([]))
    with db.connect(config.DB_PATH) as conn:
        conn.execute("UPDATE clips SET review_decision='approved',render_path='final.mp4' WHERE id=?",(cid,))
        conn.execute("INSERT INTO posts(clip_id,platform,account,status,bundle_path) VALUES(?,'instagram','a','draft','b')",(cid,))
    web.edit_clip(cid,{'layout':'centre','end_card':True})
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT * FROM clips WHERE id=?',(cid,)).fetchone()
        assert row['review_decision'] is None and row['render_path'] is None
        assert json.loads(row['data_json'])['end_card'] is True
        assert conn.execute('SELECT count(*) FROM posts WHERE clip_id=?',(cid,)).fetchone()[0]==0


def test_review_shows_hook_strength_for_display_only(studio):
    client,headers,sid,cid=studio
    with db.connect(config.DB_PATH) as conn:
        data=json.loads(conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()[0]);data['hook_strength']=.72
        conn.execute('UPDATE clips SET data_json=? WHERE id=?',(json.dumps(data),cid))
    assert 'hook 0.72' in client.get('/',params={'tab':'review','source':sid}).text


def test_review_shows_boundary_notes_overrun_and_the_playhead_timeline(studio):
    client,headers,sid,cid=studio
    with db.connect(config.DB_PATH) as conn:
        data=json.loads(conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()[0])
        data['boundary_notes']=['speaker ran on; extended 5 words to the next pause']
        data['layout']='centre'
        data['previews']['centre']={'path':'x','timeline':[[0.0,-0.12,12.0],[12.0,14.5,30.12]],
                                    'speech_overrun':{'from':30.0,'to':30.74}}
        conn.execute('UPDATE clips SET data_json=? WHERE id=?',(json.dumps(data),cid))
    page=client.get('/',params={'tab':'review','source':sid}).text
    assert 'extended 5 words to the next pause' in page and 'turn 0.74 s past the last aligned word' in page
    assert "data-timeline='[[0.0, -0.12, 12.0], [12.0, 14.5, 30.12]]'" in page
    js=(Path(web.__file__).parent/'static'/'app.js').read_text()
    assert 'editor.dataset.timeline' in js
