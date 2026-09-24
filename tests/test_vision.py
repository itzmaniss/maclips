"""Shot-local tracking tested without decoding real media or using Vision/GPU."""
from types import SimpleNamespace

import pytest

from maclips import vision


BOX = (.1, .2, .2, .3)


def frames(count=7, shot=0, start=0):
    return [{"frame": i, "time_s": start+i/5, "shot_index": shot, "boxes": [BOX]}
            for i in range(count)]


def test_iou():
    assert vision.intersection_over_union(BOX, BOX) == pytest.approx(1)
    assert vision.intersection_over_union(BOX, (.8,.2,.1,.1)) == 0
    assert 0 < vision.intersection_over_union(BOX, (.2,.2,.2,.3)) < 1


def test_top_left_conversion():
    rect = SimpleNamespace(origin=SimpleNamespace(x=.1,y=.2), size=SimpleNamespace(width=.3,height=.4))
    assert vision.top_left_box(rect) == pytest.approx((.1,.4,.3,.4))


def test_track_duration_drop_and_no_mouth():
    assert vision.associate_frames(frames(5), 5) == []
    track = vision.associate_frames(frames(6), 5)[0]
    assert track.first_s == 0 and track.last_s == 1
    assert track.mouth_signal == () and track.observations == 6


def test_shot_change_resets_even_identical_box():
    result = vision.associate_frames(frames(7) + frames(7, 1, 1.4), 5)
    assert len(result) == 2
    assert [track.shot_index for track in result] == [0, 1]


def test_long_unobserved_gap_resets_track():
    result = vision.associate_frames(frames(7) + frames(7, 0, 3), 5)
    assert len(result) == 2


def test_one_to_one_assignment():
    data = frames()
    for frame in data:
        frame["boxes"] = [BOX, (.7,.2,.2,.3)]
    tracks = vision.associate_frames(data, 5)
    assert len(tracks) == 2
    assert all(t.observations == 7 for t in tracks)


def test_window_decode_only_and_shot_metadata(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"; source.touch()
    commands = []
    def run(cmd):
        commands.append(cmd)
        pattern = cmd[-1]
        from pathlib import Path
        for i in range(1, 13):
            Path(pattern.replace("%06d", f"{i:06d}")).write_bytes(b"mock png")
        return SimpleNamespace(stderr="lavfi.scd.time=1.2\n")
    monkeypatch.setattr(vision.ffmpeg, "_run", run)
    monkeypatch.setattr(vision, "_detect", lambda png: [BOX])
    monkeypatch.setattr(vision, "_check_image", lambda png,boxes,shot,time,path: path.touch())
    result = vision.analyze_window(source, 100, 102.4, tmp_path / "checks")
    assert result["frames_processed"] == 12
    assert len(result["tracks"]) == 2
    assert result["shots"][1]["start_s"] == 101.2
    assert result["coordinate_origin"] == "top-left"
    cmd = commands[0]
    assert cmd[cmd.index("-ss")+1] == "100.000000"
    assert cmd[cmd.index("-t")+1] == "2.400000"
    assert "scdet" in cmd[cmd.index("-vf")+1]
    assert len(result["check_images"]) == 2
