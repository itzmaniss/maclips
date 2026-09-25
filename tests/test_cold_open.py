"""Cold open (idea 3): one line from the clip plays first, then the whole clip."""
import json

import pytest

from maclips import compliance, config, db, export, pacing, render, web
from maclips.compliance import ComplianceError
from maclips.orchestrator import RunContext
from maclips.stages import STAGES_BY_ID
from test_pacing import caption_events, fake_ffmpeg, w

# 24 words at 2 words/s from 20 s: no pause long enough to cut.
SPEECH = [w(f"w{i}", 20 + i * .5, 20 + i * .5 + .4) for i in range(24)]
TEASE = {"start_word": 10, "end_word": 13, "start": 25.0, "end": 26.9, "duration": 1.9,
         "problem": None, "text": "w10 w11 w12 w13"}
PAYOFF = {"start_word": 18, "end_word": 21, "start": 29.0, "end": 30.9, "duration": 1.9,
          "problem": None, "text": "w18 w19 w20 w21"}
PLAN = {"kind": "face-centred", "speaker_changes": [24.0],
        "segments": [{"start": 20, "end": 32, "kind": "face-centred", "x": .5, "y": .4}]}


def clip(**extra):
    return {"start": 20, "end": 32, "start_word": 0, "end_word": 23, "tease": TEASE,
            "payoff": PAYOFF, "hook_text": "Hook", **extra}


def graph_of(cmd):
    return cmd[cmd.index("-filter_complex") + 1]


def test_line_plays_first_then_the_whole_clip_with_a_flash_cut(tmp_path, monkeypatch):
    source, commands, asses = fake_ffmpeg(monkeypatch, tmp_path, lambda cmd: 14.0)
    result = render.render_clip(source, source, clip(cold_open="tease", end_card=True),
                                SPEECH, PLAN, tmp_path / "c.mp4")
    # The line is padded like a clip edge (half of each 0.1 s gap), then half-frame snapped.
    assert result["cold_open"] == {"mode": "tease", "source": pytest.approx([24.94, 26.94]),
                                   "duration_s": pytest.approx(2.0), "text": "w10 w11 w12 w13"}
    assert result["planned_duration_s"] == pytest.approx(14.0), "line + full clip"
    cmd = commands[0]
    assert cmd[cmd.index("-ss") + 1] == "20.000000"
    graph = graph_of(cmd)
    # The line (half-frame grid), then the clip from its first word, in order.
    assert "[source0]trim=start=4.940000:end=6.940000" in graph
    assert "[source1]trim=start=0.000000:end=4.020000" in graph
    assert "[source2]trim=start=4.020000:end=12.000000,setpts=PTS-STARTPTS,crop=526:938" in graph
    # One white frame on the clip's first frame only; the line is never zoomed.
    assert graph.count("drawbox") == 1 and "crop=606:1080:656:0,scale=540:960" + render.FLASH in graph
    assert [p["zoomed"] for p in result["pacing"]["pieces"]] == [False, False, True]
    assert result["pacing"]["switches"] == pytest.approx([6.0])
    # Audio follows the same order, crossfaded at the join against a click.
    assert "[1:a]asplit=2" in graph and "atrim=start=4.940000:end=6.965000" in graph
    assert "acrossfade=n=2:d=0.025" in graph
    events = caption_events(asses[0])
    assert events[0][0] == pytest.approx(0.06) and events[-1][1] == pytest.approx(14.0)  # first word after the pad
    for (_, b0, _), (a1, _, _) in zip(events, events[1:]):
        assert a1 == pytest.approx(b0), "captions continuous across the cold-open join"
    assert "Dialogue: 1,0:00:00.00,0:00:03.00,Hook" in asses[0]
    # A fresh caption phrase starts at the join: no line word is shown after it.
    after = [line for line in asses[0].splitlines() if line.startswith("Dialogue: 0,0:00:02.00")]
    assert after and all(f"w1{i}" not in line for line in after for i in range(4))
    assert f"Dialogue: 1,{render._ass_time(11.0)},{render._ass_time(14.0)},EndCard" in asses[0]


def test_payoff_mode_and_dead_air_cuts_inside_the_main_clip(tmp_path, monkeypatch):
    speech = SPEECH[:12] + [w(f"x{i}", 27 + i * .5, 27 + i * .5 + .4) for i in range(12)]  # 1.1 s pause cut
    payoff = {**PAYOFF, "start": 30.0, "end": 31.9, "start_word": 18, "end_word": 21}
    source, commands, asses = fake_ffmpeg(monkeypatch, tmp_path, lambda cmd: 0)
    with pytest.raises(render.RenderError, match="duration drift"):
        render.render_clip(source, source, clip(cold_open="payoff", payoff=payoff, end=33),
                           speech, {"kind": "letterbox"}, tmp_path / "p.mp4")
    graph = graph_of(commands[0])
    # The padded payoff line (source 29.94-31.94 s, input seek 20.02 s), then both keeps in order.
    assert graph.index("[source0]trim=start=9.920000:end=11.920000") < graph.index("[source1]trim=start=0.000000:end=6.000000")
    assert "[source2]trim=start=6.880000:end=13.000000" in graph
    assert "asplit=3" in graph, "line + two keeps around the cut"
    events = caption_events(asses[0])
    for (_, b0, _), (a1, _, _) in zip(events, events[1:]):
        assert a1 == pytest.approx(b0)


def test_split_margins_follow_the_combined_timeline(tmp_path, monkeypatch):
    plan = {"kind": "split", "speaker_changes": [], "segments": [
        {"start": 20, "end": 26, "kind": "face-centred", "x": .3, "y": .3},
        {"start": 26, "end": 32, "kind": "split", "boxes": [[.1, .2, .1, .2], [.7, .2, .1, .2]]}]}
    source, _, asses = fake_ffmpeg(monkeypatch, tmp_path, lambda cmd: 14.0)
    render.render_clip(source, source, clip(cold_open="tease", zoom=False), SPEECH, plan, tmp_path / "s.mp4")
    events = caption_events(asses[0])
    # The line straddles the 26 s shot change: bottom, then seam, then the clip restarts at the bottom.
    assert {m for a, b, m in events if b <= 1.0} == {230}
    assert {m for a, b, m in events if 1.1 <= a and b <= 2.0} == {880}
    assert {m for a, b, m in events if 2.0 <= a and b <= 8.0} == {230}
    assert {m for a, b, m in events if a >= 8.02} == {880}  # shot change on the half-frame grid


def test_off_and_missing_render_the_same_command(tmp_path, monkeypatch):
    source, commands, _ = fake_ffmpeg(monkeypatch, tmp_path, lambda cmd: 12.0)
    render.render_clip(source, source, clip(), SPEECH, PLAN, tmp_path / "a.mp4")
    render.render_clip(source, source, clip(cold_open="off"), SPEECH, PLAN, tmp_path / "b.mp4")
    assert graph_of(commands[0]).split("ass=")[0] == graph_of(commands[1]).split("ass=")[0]
    assert "drawbox" not in graph_of(commands[0])


@pytest.mark.parametrize("extra, message", [
    ({"cold_open": "reveal"}, "unknown cold open mode"),
    ({"cold_open": "tease", "start_word": 12}, "outside the clip"),
    ({"cold_open": "tease", "tease": {**TEASE, "problem": "overlaps the tease"}}, "overlaps"),
    ({"cold_open": "payoff", "payoff": None}, "no line"),
])
def test_an_unplayable_line_stops_before_encoding(tmp_path, monkeypatch, extra, message):
    source, commands, _ = fake_ffmpeg(monkeypatch, tmp_path)
    with pytest.raises(render.RenderError, match=message):
        render.render_clip(source, source, clip(**extra), SPEECH, PLAN, tmp_path / "x.mp4")
    assert commands == []


def test_final_duration_gate_uses_the_combined_duration(tmp_path, monkeypatch):
    source, commands, _ = fake_ffmpeg(monkeypatch, tmp_path)
    # The gate sees 13.9 s: the 14.0 s combined timeline less the line's 0.1 s of pad.
    with pytest.raises(render.RenderError, match="edited duration 13.900s outside 5–13s"):
        render.render_clip(source, source, clip(cold_open="tease"), SPEECH, PLAN, tmp_path / "f.mp4",
                           proxy=False, duration_range=(5, 13))
    assert commands == []


def test_combined_words_caption_the_line_then_the_clip():
    timed = pacing.combined_words(SPEECH, [(25.0, 26.9)], [(20.0, 32.0)])
    first = [(x["word"], round(x["start"], 2)) for x in timed if x["start"] < 1.85]
    assert first == [("w10", 0), ("w11", .5), ("w12", 1.0), ("w13", 1.5)]
    assert [x for x in timed if x["word"] == "w0"][0]["start"] == pytest.approx(1.9)


def test_stage_versions_bumped_for_the_cold_open():
    assert STAGES_BY_ID["S5"].version >= 2
    assert STAGES_BY_ID["S8"].version >= 5 and STAGES_BY_ID["S12"].version >= 4


# --------------------------------------------------------------------------- #
# Compliance, export and Review
# --------------------------------------------------------------------------- #

def test_cold_open_adds_a_human_check_but_never_blocks_automatically():
    brief = {"forbidden_edits": ["no reordering"]}
    assert compliance.human_checklist(brief, clip(cold_open="tease"))[0] == compliance.COLD_OPEN_CHECK
    assert compliance.COLD_OPEN_CHECK not in compliance.human_checklist(brief, clip())
    base = {**clip(cold_open="tease"), "platform": "tiktok", "account_class": "general"}
    compliance.validate_clip(base, {}, "general-own")
    with pytest.raises(ComplianceError, match="cannot play"):
        compliance.validate_clip({**base, "start_word": 12}, {}, "general-own")


def test_export_checklist_names_the_mode_and_line(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "t.db")
    mp4 = tmp_path / "final.mp4"; mp4.write_bytes(b"x")
    with db.connect(config.DB_PATH) as conn:
        sid = db.register_source(conn, "h", "s.mp4", "general-own")
        cid = db.save_candidate(conn, sid, "general-own", {"rank": 1, **clip()})
        conn.execute("UPDATE clips SET review_decision='approved' WHERE id=?", (cid,))
    rendered = {**clip(cold_open="tease"), "id": cid, "platform": "tiktok", "path": str(mp4),
                "caption": "c", "account": "@me"}
    ctx = RunContext("r", tmp_path / "s", "h", tmp_path / "w", {}, outputs={
        "S0": {"clip_class": "general-own"}, "S1": {}, "S12": {"rendered": [rendered]}})
    export.export_stage(ctx)
    text = (tmp_path / "w/exports" / str(cid) / "tiktok/checklist.md").read_text()
    assert '- Cold open (tease) plays first, then the full clip: "w10 w11 w12 w13"' in text
    assert "- [ ] " + compliance.COLD_OPEN_CHECK in text


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "test.db")
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps({"words": [{**x, "word": x["word"]} for x in SPEECH]}))
    media = tmp_path / "fixture.mp4"; media.write_bytes(b"fixture")
    state = {"audio": str(media), "video": str(media), "transcript": str(transcript), "workdir": str(tmp_path),
             "experiment_only": False, "brief": {}, "config": {}, "plans": {"1": {"letterbox": {"kind": "letterbox"}}}}
    with db.connect(config.DB_PATH) as conn:
        sid = db.register_source(conn, "fixture", media, "general-own", state=state)
        cid = db.save_candidate(conn, sid, "general-own", {"rank": 1, **clip()}, {"letterbox": {"path": str(media)}})
        conn.execute("UPDATE clips SET review_decision='approved' WHERE id=?", (cid,))
    calls = []
    monkeypatch.setattr(render, "render_clip", lambda v, a, cand, words, layout, path, **kw:
                        calls.append(cand.get("cold_open")) or {"path": str(path)})
    return sid, cid, calls


def stored(cid):
    with db.connect(config.DB_PATH) as conn:
        row = conn.execute("SELECT * FROM clips WHERE id=?", (cid,)).fetchone()
    return row, json.loads(row["data_json"])


def test_review_toggle_rerenders_and_revokes_approval(studio):
    sid, cid, calls = studio
    web.edit_clip(cid, {"cold_open": "tease"})
    row, data = stored(cid)
    assert calls == ["tease"] and data["cold_open"] == "tease" and row["review_decision"] is None
    web.edit_clip(cid, {"cold_open": "off"})
    assert calls == ["tease", "off"] and stored(cid)[1]["cold_open"] == "off"


def test_only_valid_modes_are_offered_and_accepted(studio):
    sid, cid, calls = studio
    with db.connect(config.DB_PATH) as conn:
        _, data = stored(cid); data["payoff"] = {**PAYOFF, "problem": "overlaps the tease"}
        conn.execute("UPDATE clips SET data_json=? WHERE id=?", (json.dumps(data), cid))
    with pytest.raises(ValueError, match="no valid payoff line"):
        web.edit_clip(cid, {"cold_open": "payoff"})
    with pytest.raises(ValueError, match="off, tease or payoff"):
        web.edit_clip(cid, {"cold_open": "both"})
    from fastapi.testclient import TestClient
    page = TestClient(web.app).get("/", params={"tab": "review", "source": sid}).text
    assert 'id="cold-open" type="checkbox"  >' in page, "unchecked by default, enabled"
    assert '<option value="tease"' in page and '<option value="payoff"' not in page
    web.edit_clip(cid, {"cold_open": "tease"})
    page = TestClient(web.app).get("/", params={"tab": "review", "source": sid}).text
    assert compliance.COLD_OPEN_CHECK in page


def test_a_trim_past_the_line_falls_back_to_off(studio):
    sid, cid, calls = studio
    web.edit_clip(cid, {"cold_open": "tease"})
    web.edit_clip(cid, {"start_word": 14})
    assert stored(cid)[1]["cold_open"] == "off" and calls[-1] == "off"
