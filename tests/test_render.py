"""Renderer hard output gates and original-input single-pass contract."""
from pathlib import Path

import pytest

from maclips import config, render


def info(**kw):
    return {"duration_s": 12.0, "width": 1080, "height": 1920,
            "has_audio": True, "has_video": True, **kw}


@pytest.mark.parametrize("changes,message", [
    ({"duration_s": 12.101}, "duration drift"),
    ({"width": 1920, "height": 1080}, "wrong resolution"),
    ({"has_audio": False}, "without an audio"),
])
def test_each_output_gate(changes, message):
    with pytest.raises(render.RenderError, match=message):
        render.validate_output(info(**changes), 12)


def test_tolerance_boundary_and_proxy_resolution():
    render.validate_output(info(duration_s=12.1), 12)
    render.validate_output(info(width=540, height=960), 12, proxy=True)


@pytest.mark.parametrize("proxy,encoder", [(True, "h264_videotoolbox"), (False, "libx264")])
def test_separate_original_inputs_single_graph_and_atomic_output(tmp_path, monkeypatch, proxy, encoder):
    video, audio = tmp_path / "original.mp4", tmp_path / "original.m4a"
    video.touch(); audio.touch()
    out = tmp_path / "finished.mp4"
    commands = []
    def probe(path):
        if path == video:
            return info(width=2560, height=1280)
        return info(width=540 if proxy else 1080, height=960 if proxy else 1920)
    def run(cmd):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"complete media")
        ass_path = cmd[cmd.index("-filter_complex")+1].split("ass=filename='")[1].split("'")[0]
        assert "UNAPPROVED TEST RENDER" in Path(ass_path).read_text()
    monkeypatch.setattr(render.ffmpeg, "probe", probe)
    monkeypatch.setattr(render.ffmpeg, "_run", run)
    result = render.render_clip(video, audio, {"start": 5, "end": 17, "hook_text": "hello"}, [],
                                "centre", out, proxy=proxy, diagnostic=True)
    cmd = commands[0]
    assert cmd[0] == str(config.FFMPEG)
    assert cmd.count("-i") == 2 and cmd.count("-filter_complex") == 1
    assert cmd[cmd.index("-c:v") + 1] == encoder
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[1:a]atrim" in graph and "loudnorm=I=-14" in graph
    assert "crop=720:1280" in graph
    assert result["path"] == str(out) and out.read_bytes() == b"complete media"
    assert not list(tmp_path.glob(".render-*"))


def test_failed_output_is_not_published(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"; source.touch()
    out = tmp_path / "good.mp4"; out.write_bytes(b"previous good result")
    monkeypatch.setattr(render.ffmpeg, "probe", lambda path: info(width=2560, height=1280) if path == source else info(has_audio=False))
    monkeypatch.setattr(render.ffmpeg, "_run", lambda cmd: Path(cmd[-1]).write_bytes(b"broken"))
    with pytest.raises(render.RenderError, match="without an audio"):
        render.render_clip(source, source, {"start": 0, "end": 12}, [], "letterbox", out, proxy=False)
    assert out.read_bytes() == b"previous good result"
    assert not list(tmp_path.glob(".render-*"))


@pytest.mark.parametrize("candidate,layout", [({"start": -1, "end": 2}, "centre"), ({"start": 4, "end": 2}, "centre"), ({"start": 0, "end": 2}, "follow")])
def test_bad_plan_rejected_before_encoder(tmp_path, candidate, layout):
    with pytest.raises(render.RenderError):
        render.render_clip(tmp_path / "missing", tmp_path / "missing", candidate, [], layout, tmp_path / "out.mp4")


def test_ass_word_highlight_relative_timings_and_injection_escape(tmp_path):
    path = tmp_path / "caption.ass"
    render.write_ass(path, [{"text": "{\\pos(1,1)}Hello", "start": 11, "end": 11.5},
                            {"text": "world", "start": 11.5, "end": 12}], 10, 13, "Hook", split=True)
    text = path.read_text()
    assert "0:00:01.00,0:00:01.50" in text
    assert "\\pos(" not in text
    assert "{\\c&H00FFFF&}" in text
    assert "65,65,880,1" in text


def test_split_left_person_is_top_and_captions_at_seam(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"; source.touch()
    commands = []
    monkeypatch.setattr(render.ffmpeg, "probe", lambda path: info(width=2560, height=1280) if path == source else info())
    def run(cmd):
        commands.append(cmd)
        Path(cmd[-1]).write_bytes(b"video")
    monkeypatch.setattr(render.ffmpeg, "_run", run)
    render.render_clip(source, source, {"start": 0, "end": 12}, [],
                       {"kind": "split", "boxes": [[.75,.2,.1,.2], [.1,.2,.1,.2]]},
                       tmp_path / "split.mp4", proxy=False)
    graph = commands[0][commands[0].index("-filter_complex")+1]
    assert "[top]crop=1216:1080:0:0" in graph
    assert "[bottom]crop=1344:1194:1216:0" in graph
    assert "vstack=inputs=2" in graph
    assert "scale=1080:960" in graph


def test_face_crop_clamps_to_source_edge():
    assert render._crop(0, 9/16, 2560, 1280) == "crop=720:1280:0:0"
    assert render._crop(1, 9/16, 2560, 1280) == "crop=720:1280:1840:0"


@pytest.mark.parametrize("segments", [
    [{"start":10,"end":15,"kind":"letterbox"},{"start":16,"end":20,"kind":"letterbox"}],
    [{"start":10,"end":16,"kind":"letterbox"},{"start":15,"end":20,"kind":"letterbox"}],
    [{"start":11,"end":20,"kind":"letterbox"}],
    [{"start":10,"end":19,"kind":"letterbox"}],
    [{"start":10,"end":float('nan'),"kind":"letterbox"}],
])
def test_segment_gaps_overlaps_uncovered_and_nonfinite_stop(segments):
    with pytest.raises(render.RenderError):
        render.clipped_segments(segments, 10, 20)


def test_saved_segments_clip_to_new_word_bounds():
    saved=[{"start":10,"end":15,"kind":"letterbox"},{"start":15,"end":20,"kind":"face-centred","x":.3}]
    clipped=render.clipped_segments(saved,12,18)
    assert [(s['start'],s['end']) for s in clipped] == [(12,15),(15,18)]
    assert saved[0]['start'] == 10


def test_segment_graph_uses_trim_concat_and_single_overlay(tmp_path, monkeypatch):
    source=tmp_path/'source.mp4';source.touch()
    commands=[]
    monkeypatch.setattr(render.ffmpeg,'probe',lambda path: info(width=2560,height=1280) if path==source else info())
    def run(cmd):
        commands.append(cmd);Path(cmd[-1]).write_bytes(b'media')
    monkeypatch.setattr(render.ffmpeg,'_run',run)
    plan={'kind':'split','segments':[{'start':100,'end':105,'kind':'letterbox'},
          {'start':105,'end':112,'kind':'split','boxes':[[.1,.2,.1,.2],[.7,.2,.1,.2]]}]}
    render.render_clip(source,source,{'start':100,'end':112},[],plan,tmp_path/'out.mp4',proxy=False)
    graph=commands[0][commands[0].index('-filter_complex')+1]
    assert 'trim=start=0.000000:end=5.000000' in graph
    assert 'trim=start=5.000000:end=12.000000' in graph
    assert 'concat=n=2:v=1:a=0' in graph
    assert graph.count('ass=filename') == graph.count('loudnorm=') == 1


def test_caption_crossing_shot_cut_changes_position(tmp_path):
    path=tmp_path/'caption.ass'
    render.write_ass(path,[{'word':'hello','start':11,'end':13}],10,14,'',segments=[
        {'start':10,'end':12,'kind':'letterbox'}, {'start':12,'end':14,'kind':'split'}])
    text=path.read_text()
    assert '0:00:01.00,0:00:02.00,Caption,,0,0,230,,' in text
    assert '0:00:02.00,0:00:04.00,Caption,,0,0,880,,' in text


def caption_events(text):
    import re
    def t(s):
        h, m, rest = s.split(':'); return int(h)*3600 + int(m)*60 + float(rest)
    return sorted((t(a), t(b), margin, body) for a, b, margin, body in
                  re.findall(r'^Dialogue: 0,([^,]+),([^,]+),Caption,,0,0,(\d+),,(.*)$', text, re.M))


def test_captions_tile_gaps_and_long_pause_with_one_highlight(tmp_path):
    words = [{'text': f'w{i}', 'start': s, 'end': s + .3}
             for i, s in enumerate([10.2, 10.6, 11.0, 11.5, 12.0, 12.4, 13.0, 21.0, 21.5])]
    words.append({'text': 'broken', 'start': 14, 'end': 13})
    path = tmp_path / 'caption.ass'
    render.write_ass(path, words, 10, 25, '', segments=[
        {'start': 10, 'end': 16.05, 'kind': 'face-centred'}, {'start': 16.05, 'end': 25, 'kind': 'split'}])
    events = caption_events(path.read_text())
    assert events[0][0] == pytest.approx(.2) and events[-1][1] == pytest.approx(15)
    for (a0, b0, *_), (a1, *_) in zip(events, events[1:]):
        assert a1 == pytest.approx(b0)
    assert all(body.count('{\\c&H00FFFF&}') == 1 for *_, body in events)
    # The pause 13.3-21.0 is covered by w6 across the layout switch at 16.05.
    pause = [e for e in events if 'w6' in e[3].split('{\\c&HFFFFFF&}')[0].split('}')[-1]]
    assert [(a, b, m) for a, b, m, _ in pause] == [(3, 6.05, '230'), (6.05, 11, '880')]
    assert '0:00:15.00,Caption' in path.read_text()


def crop_rects(boxes, width, height):
    return [tuple(int(v) for v in c[5:].split(':')) for c in render._split_crops(boxes, width, height)]


def assert_split_geometry(boxes, width, height):
    (w1, h1, x1, y1), (w2, h2, x2, y2) = rects = crop_rects(boxes, width, height)
    assert x1 + w1 <= x2, "panels overlap horizontally"
    for (w, h, x, y), (bx, by, bw, bh) in zip(rects, sorted(boxes, key=lambda b: b[0] + b[2]/2)):
        assert 0 <= x and x + w <= width and 0 <= y and y + h <= height
        assert abs(w / h - 1080/960) < .01
        assert x <= bx*width and (bx+bw)*width <= x + w and y <= by*height and (by+bh)*height <= y + h
    return rects


def test_side_by_side_call_panels_hold_only_their_own_face():
    # Video 3 candidate 1 shot 1: host left, guest right, 1920x1080.
    boxes = [[0.1299, 0.1806, 0.2040, 0.3627], [0.6139, 0.2807, 0.2976, 0.5291]]
    (w1, h1, x1, y1), (w2, h2, x2, y2) = assert_split_geometry(boxes, 1920, 1080)
    assert x1 + w1 == x2 == 954
    assert (x2 + w2) > (0.6139 + 0.2976) * 1920 and x1 + w1 < 0.6139 * 1920


def test_benchmark_wide_shot_split_still_fits():
    boxes = [[0.719, 0.381, 0.027, 0.054], [0.296, 0.383, 0.024, 0.047]]
    (w1, h1, *_), (w2, h2, *_) = assert_split_geometry(boxes, 2560, 1280)
    assert min(h1, h2) >= 1000


@pytest.mark.parametrize("boxes", [
    [[0, 0, .05, .1], [.95, .9, .05, .1]],
    [[.02, .8, .2, .2], [.4, 0, .1, .1]],
    [[.3, .45, .1, .1], [.55, .45, .1, .1]],
])
def test_split_crops_never_leave_frame(boxes):
    assert_split_geometry(boxes, 1920, 1080)


def test_stacked_or_touching_faces_are_not_split():
    stacked = [[0.863, 0.812, 0.063, 0.112], [0.859, 0.476, 0.055, 0.099]]
    with pytest.raises(render.RenderError, match="too close"):
        render._split_crops(stacked, 1920, 1080)
