import json
from pathlib import Path
import pytest
from maclips import config,db,export
from maclips.compliance import ComplianceError
from maclips.orchestrator import GateFailure,RunContext


def test_real_sqlite_post_gate_and_events(tmp_path):
    with db.connect(tmp_path/'test.db') as conn:
        sid=db.register_source(conn,'abc','source.mp4','campaign')
        cid=db.save_candidate(conn,sid,'campaign',{'rank':1,'start':0,'end':30,'start_word':0,'end_word':20})
        conn.execute("UPDATE clips SET review_decision='approved',render_path='fixture.mp4' WHERE id=?",(cid,))
        row={'post_url':'https://example.test/post/1','status':'submitted','disclosure_ticked':False}
        with pytest.raises(ComplianceError,match='checkbox'):
            db.record_post(conn,cid,'instagram','fixture-account',row)
        assert conn.execute('SELECT count(*) FROM posts').fetchone()[0]==0
        row['disclosure_ticked']=True
        pid=db.record_post(conn,cid,'instagram','fixture-account',row)
        assert conn.execute('SELECT status FROM posts WHERE id=?',(pid,)).fetchone()[0]=='submitted'
        assert {'source_registered','post_url_pasted','post_submitted'} <= {r[0] for r in conn.execute('SELECT event FROM time_log')}


def test_export_blocks_before_writing_without_persisted_approval(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.db')
    ctx=RunContext('test',tmp_path/'source','hash',tmp_path/'work',{},outputs={
      'S0':{'clip_class':'general-own'},'S1':{},'S12':{'rendered':[{'id':1,'platform':'instagram','path':'missing','caption':'hi','account':'fixture'}]}})
    with pytest.raises(GateFailure,match='persisted human approval'):
        export.export_stage(ctx)
    assert not (ctx.workdir/'exports').exists()


def test_fixture_bundle_and_draft_post(tmp_path,monkeypatch):
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.db')
    source=tmp_path/'fixture.mp4';source.write_bytes(b'fixture video')
    with db.connect(config.DB_PATH) as conn:
        sid=db.register_source(conn,'abc',source,'general-own')
        cid=db.save_candidate(conn,sid,'general-own',{'rank':1,'start':0,'end':30})
        conn.execute("UPDATE clips SET review_decision='approved' WHERE id=?",(cid,))
    ctx=RunContext('test',source,'hash',tmp_path/'work',{},outputs={
      'S0':{'clip_class':'general-own'},'S1':{},'S12':{'rendered':[{'id':cid,'platform':'instagram','path':str(source),'caption':'caption #tag','account':'fixture'}]}})
    bundle=Path(export.export_stage(ctx)['bundles'][0]['path'])
    assert {p.name for p in bundle.iterdir()}=={'clip.mp4','caption.txt','checklist.md'}
    assert 'caption #tag' in (bundle/'caption.txt').read_text()
    with db.connect(config.DB_PATH) as conn:
        assert conn.execute('SELECT status FROM posts').fetchone()[0]=='draft'
        assert conn.execute('SELECT post_url FROM posts').fetchone()[0] is None


def test_experiment_protection_survives_reingest(tmp_path):
    with db.connect(tmp_path/'test.db') as conn:
        sid=db.register_source(conn,'abc','source','general-own',state={'experiment_only':True})
        db.register_source(conn,'abc','source','general-own',state={'experiment_only':False})
        assert json.loads(conn.execute('SELECT state_json FROM sources WHERE id=?',(sid,)).fetchone()[0])['experiment_only'] is True


def test_rerender_uses_persisted_trim(tmp_path,monkeypatch):
    from maclips import tracking
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.db')
    original={'rank':1,'start_word':0,'end_word':20,'start':0,'end':30}
    with db.connect(config.DB_PATH) as conn:
        sid=db.register_source(conn,'abc','source','general-own')
        cid=db.save_candidate(conn,sid,'general-own',original)
        edited={**original,'start_word':1,'start':1,'hook_text':'edited'}
        conn.execute('UPDATE clips SET data_json=? WHERE id=?',(json.dumps(edited),cid))
    ctx=RunContext('test',tmp_path/'source','abc',tmp_path,{},shared={'source_id':sid})
    assert tracking.effective_candidate(ctx,original)==edited


def test_regenerated_same_layout_revokes_approval(tmp_path,monkeypatch):
    from maclips import tracking
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.db')
    candidate={'rank':1,'start_word':0,'end_word':20,'start':0,'end':30}
    with db.connect(config.DB_PATH) as conn:
        sid=db.register_source(conn,'abc','source','general-own')
        cid=db.save_candidate(conn,sid,'general-own',candidate,{'split':{'path':'old'}})
        data=json.loads(conn.execute('SELECT data_json FROM clips WHERE id=?',(cid,)).fetchone()[0]);data['layout']='split'
        conn.execute("UPDATE clips SET review_decision='approved',render_path='old-final',data_json=? WHERE id=?",(json.dumps(data),cid))
    ctx=RunContext('test',tmp_path/'source','abc',tmp_path,{},outputs={'S7':{'plans':{'1':{'split':{'kind':'split'}}}}},shared={'source_id':sid})
    tracking.record_preview(ctx,candidate,'split',{'path':'new'})
    with db.connect(config.DB_PATH) as conn:
        row=conn.execute('SELECT * FROM clips WHERE id=?',(cid,)).fetchone()
        assert row['review_decision'] is None and row['render_path'] is None
        assert json.loads(row['data_json'])['revision']==1


def test_posted_clip_blocks_preview_regeneration(tmp_path,monkeypatch):
    from maclips import tracking,render_stages,render
    monkeypatch.setattr(config,'DB_PATH',tmp_path/'test.db')
    candidate={'rank':1,'start_word':0,'end_word':20,'start':0,'end':30}
    with db.connect(config.DB_PATH) as conn:
        sid=db.register_source(conn,'abc','source','general-own')
        cid=db.save_candidate(conn,sid,'general-own',candidate)
        conn.execute("INSERT INTO posts(clip_id,platform,account,status,post_url) VALUES(?,'instagram','fixture','posted','https://example.test/post')",(cid,))
    ctx=RunContext('test',tmp_path/'source','abc',tmp_path,{},outputs={'S6':{},'S3':{'words':[]},'S5':{'candidates':[candidate]}},shared={'source_id':sid})
    monkeypatch.setattr(tracking,'register_context',lambda ctx:sid)
    monkeypatch.setattr(render,'render_clip',lambda *a,**kw:pytest.fail('posted clip rendered'))
    with pytest.raises(GateFailure,match='posted clip is immutable'):
        render_stages.proxy_render(ctx)
