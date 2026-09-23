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
    assert "[top]crop=1440:1280:0:0" in graph
    assert "[bottom]crop=1440:1280:1120:0" in graph
    assert "vstack=inputs=2" in graph
    assert "scale=1080:960" in graph


def test_face_crop_clamps_to_source_edge():
    assert render._crop(0, 9/16, 2560, 1280) == "crop=720:1280:0:0"
    assert render._crop(1, 9/16, 2560, 1280) == "crop=720:1280:1840:0"
