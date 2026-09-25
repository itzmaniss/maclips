"""Dead-air removal and punch-in zoom: timeline arithmetic and the one-pass graph."""
import dataclasses
from pathlib import Path

import pytest

from maclips import config, pacing, render
from maclips.layouts import speaker_changes
from maclips.orchestrator import RunContext, StageSpec, run_pipeline
from maclips.stages import STAGES_BY_ID


def flat(pairs):
    return [t for pair in pairs for t in pair]


def w(text, a, b):
    return {"word": text, "start": a, "end": b}


# Speech with silences of 0.5 s (kept), 2.0 s and 1.2 s (both cut).
WORDS = [w("a", 10.0, 10.51), w("b", 11.01, 11.41), w("c", 13.41, 14.01), w("d", 15.21, 16.0)]


def test_keeps_pad_both_sides_of_each_cut_and_keep_short_gaps():
    keeps = pacing.dead_air_keeps(WORDS, 10.0, 16.0)
    assert flat(keeps) == pytest.approx([10.0, 11.53, 13.29, 14.13, 15.09, 16.0])


def test_all_speech_and_threshold_gap_are_one_keep():
    speech = [w("a", 0, 1), w("b", 1.1, 2), w("c", 2.8, 4)]  # gap 0.8 is not above the threshold
    assert pacing.dead_air_keeps(speech, 0, 4) == [(0, 4)]


def test_long_overlapping_word_is_never_cut_into():
    words = [w("a", 0, 5), w("b", 1, 1.5), w("c", 5.5, 6)]  # "a" still speaks over b's gap
    assert pacing.dead_air_keeps(words, 0, 6) == [(0, 6)]


def test_snap_moves_bounds_to_half_frames_and_merges_touching_keeps():
    snapped = pacing.snap_keeps([(10.0, 11.52), (11.53, 12.0), (13.28, 14.12)], 25)
    assert flat(snapped) == pytest.approx([10.02, 12.02, 13.3, 14.14])
    for a, b in snapped:
        assert ((b - a) * 25) == pytest.approx(round((b - a) * 25))  # whole frames


def test_edited_time_maps_keeps_and_collapses_cuts():
    keeps = [(10.0, 11.5), (13.0, 14.0)]
    assert pacing.edited_time(10.5, keeps) == pytest.approx(0.5)
    assert pacing.edited_time(12.0, keeps) == pytest.approx(1.5)   # inside the cut
    assert pacing.edited_time(13.5, keeps) == pytest.approx(2.0)
    assert pacing.edited_time(20.0, keeps) == pytest.approx(2.5)


def test_zoom_rate_limit_holds_every_framing_three_seconds():
    assert pacing.zoom_switches([1, 3.5, 4, 8, 9.5, 10], 12) == [3.5, 8]
    assert pacing.zoom_switches([2.9], 12) == []            # too soon after the start
    assert pacing.zoom_switches([9.5], 12) == []            # too close to the end


def test_speaker_changes_come_from_s4_window_turns():
    words = [w("a", 1, 1.5), w("b", 2, 2.5), w("c", 3, 3.5), w("d", 4, 4.5), w("e", 9, 9.5)]
    turns = [{"start": 0, "end": 2.8, "speaker": "SPEAKER_00"},
             {"start": 2.8, "end": 3.9, "speaker": "SPEAKER_01"},
             {"start": 3.9, "end": 8, "speaker": "SPEAKER_00"}]
    assert speaker_changes(words, turns) == [3, 4]  # "e" lies outside every turn
    assert speaker_changes(words, []) == []


def test_s7_puts_window_speaker_changes_and_face_centre_on_every_plan(tmp_path):
    from maclips.stages import s7_attribution
    words = [w("a", 1, 1.5), w("b", 2, 2.5), w("c", 3, 3.5)]
    ctx = RunContext("r", tmp_path / "src", "h", tmp_path, {}, outputs={
        "S3": {"words": words},
        "S4": {"windows": [{"start": 1, "end": 3.5, "turns": [
            {"start": 0, "end": 2.8, "speaker": "SPEAKER_00"}, {"start": 2.8, "end": 9, "speaker": "SPEAKER_01"}]}]},
        "S5": {"candidates": [{"rank": 1, "start": 1, "end": 3.5}]},
        "S6": {"analysis": {"1": {"shots": [{"shot_index": 0, "start_s": 1, "end_s": 3.5}],
                                  "tracks": [{"median_box": [.4, .2, .1, .2], "first_s": 1, "last_s": 3.5, "shot_index": 0}]}}}})
    plans = s7_attribution(ctx)["plans"]["1"]
    assert {name: plan["speaker_changes"] for name, plan in plans.items()} == {"face-centred": [3], "letterbox": [3]}
    segment = plans["face-centred"]["segments"][0]
    assert segment["y"] == pytest.approx(.3) and segment["box"] == [.4, .2, .1, .2]


def test_triggers_are_joins_speaker_changes_and_pauses_when_cutting_is_off():
    plan = {"kind": "face-centred", "x": .5, "speaker_changes": [13.5, 99]}
    on = pacing.plan_pacing({}, WORDS, plan, 10.0, 16.0, grid=lambda: (25, 0.0))
    assert len(on["keeps"]) == 3 and on["triggers"] == pytest.approx([1.52, 1.72, 2.36])
    off = pacing.plan_pacing({"dead_air": False}, WORDS, plan, 10.0, 16.0)
    # 2.0 s pause resumes at 13.41 (trigger 0.12 s earlier); the 1.2 s pause is under 1.5 s.
    assert off["keeps"] == [(10.0, 16.0)] and off["triggers"] == pytest.approx([3.29, 3.5])
    assert off["switches"] == []  # a 6 s clip cannot hold both framings for 3 s
    none = pacing.plan_pacing({"zoom": False}, WORDS, plan, 10.0, 16.0, grid=lambda: (25, 0.0))
    assert none["switches"] == []


def test_letterbox_and_split_plans_never_zoom():
    long = [w(str(i), i * 2.0, i * 2.0 + 1) for i in range(10)]  # 1 s gaps -> 9 cuts
    for plan in ({"kind": "letterbox"}, {"kind": "split", "segments": [{"start": 0, "end": 20, "kind": "split"}]}):
        assert pacing.plan_pacing({}, long, plan, 0, 19, grid=lambda: (25, 0.0))["switches"] == []


def test_zoomed_crop_keeps_face_inside_and_scales_about_it():
    for box, (width, height) in [([.05, .1, .1, .2], (1920, 1080)), ([.85, .05, .12, .2], (1920, 1080)),
                                 ([.45, .3, .06, .12], (2560, 1280)), ([.4, .6, .1, .2], (1920, 1080))]:
        cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
        cw, ch, x, y = map(int, render._zoom_crop(cx, cy, width, height, 1.15)[5:].split(":"))
        assert ch == int(height / 1.15) // 2 * 2 and 0 <= x and x + cw <= width and 0 <= y and y + ch <= height
        assert x <= box[0] * width and (box[0] + box[2]) * width <= x + cw
        assert y <= box[1] * height and (box[1] + box[3]) * height <= y + ch


def fake_ffmpeg(monkeypatch, tmp_path, duration_of=lambda cmd: 6.0):
    source = tmp_path / "source.mp4"; source.touch()
    commands, asses = [], []
    def probe(path):
        if Path(path) == source:
            return {"width": 1920, "height": 1080, "has_video": True, "has_audio": True, "duration_s": 999}
        return {"width": 540, "height": 960, "has_video": True, "has_audio": True, "duration_s": duration_of(commands[-1])}
    def run(cmd, timeout=None):
        commands.append(cmd)
        graph = cmd[cmd.index("-filter_complex") + 1]
        asses.append(Path(graph.split("ass=filename='")[1].split("'")[0]).read_text())
        Path(cmd[-1]).write_bytes(b"media")
    monkeypatch.setattr(render.ffmpeg, "probe", probe)
    monkeypatch.setattr(render.ffmpeg, "_run", run)
    monkeypatch.setattr(render, "_frame_grid", lambda video: (25.0, 0.0))
    return source, commands, asses


def caption_events(text):
    import re
    def t(s):
        h, m, rest = s.split(":"); return int(h) * 3600 + int(m) * 60 + float(rest)
    return sorted((t(a), t(b), int(m)) for a, b, m in
                  re.findall(r"^Dialogue: 0,([^,]+),([^,]+),Caption,,0,0,(\d+),,", text, re.M))


PLAN = {"kind": "split", "speaker_changes": [], "segments": [
    {"start": 10.0, "end": 12.5, "kind": "face-centred", "x": .3, "y": .3},
    {"start": 12.5, "end": 16.0, "kind": "split", "boxes": [[.1, .2, .1, .2], [.7, .2, .1, .2]]}]}


def test_cut_clip_is_one_pass_with_mapped_captions_segments_and_hook(tmp_path, monkeypatch):
    source, commands, asses = fake_ffmpeg(monkeypatch, tmp_path, lambda cmd: 3.28)
    result = render.render_clip(source, source, {"start": 10.0, "end": 16.0, "hook_text": "Hook"},
                                WORDS, PLAN, tmp_path / "out.mp4")
    keeps = result["pacing"]["keeps"]
    assert flat(keeps) == pytest.approx([10.02, 11.54, 13.3, 14.14, 15.1, 16.02])
    assert result["planned_duration_s"] == pytest.approx(3.28) and result["pacing"]["cuts"] == 2
    cmd = commands[0]
    assert cmd.count("-i") == 2 and cmd.count("-filter_complex") == 1
    assert cmd[cmd.index("-ss") + 1] == "10.020000"
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "[1:a]asplit=3" in graph and "acrossfade=n=3:d=0.025" in graph
    assert "atrim=start=0.000000:end=1.545000" in graph  # keep + 25 ms crossfade tail
    assert "atrim=start=5.080000:end=6.000000" in graph  # last keep has no tail
    assert graph.count("loudnorm") == graph.count("ass=filename") == 1
    # Segment switch at 12.5 lies inside the first cut, so the split starts at the 2nd keep.
    assert "trim=start=0.000000:end=1.520000" in graph and "crop=606:1080" in graph
    assert "trim=start=3.280000:end=4.120000,setpts=PTS-STARTPTS,split=2" in graph
    events = caption_events(asses[0])
    assert events[0][0] == 0 and events[-1][1] == pytest.approx(3.28)
    for (a0, b0, _), (a1, _, _) in zip(events, events[1:]):
        assert a1 == pytest.approx(b0)  # continuous captions on the edited timeline
    assert {m for a, b, m in events if b <= 1.52} == {230} and {m for a, b, m in events if a >= 1.52} == {880}
    assert "Dialogue: 1,0:00:00.00,0:00:03.00,Hook" in asses[0]


def test_zoom_switch_splits_a_face_segment_into_normal_and_zoomed_crops(tmp_path, monkeypatch):
    source, commands, _ = fake_ffmpeg(monkeypatch, tmp_path, lambda cmd: 12.0)
    speech = [w(str(i), 20 + i * .5, 20 + i * .5 + .4) for i in range(24)]
    plan = {"kind": "face-centred", "speaker_changes": [24.0],
            "segments": [{"start": 20, "end": 32, "kind": "face-centred", "x": .5, "y": .4}]}
    result = render.render_clip(source, source, {"start": 20, "end": 32}, speech, plan, tmp_path / "z.mp4")
    assert result["pacing"]["cuts"] == 0 and result["pacing"]["switches"] == [4.0]
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert "trim=start=0.000000:end=4.020000,setpts=PTS-STARTPTS,crop=606:1080" in graph
    assert "trim=start=4.020000:end=12.000000,setpts=PTS-STARTPTS,crop=526:938" in graph
    assert "[1:a]atrim=duration=12.000000" in graph and "concat=n=2" in graph


def test_toggles_off_reproduce_todays_command(tmp_path, monkeypatch):
    source, commands, asses = fake_ffmpeg(monkeypatch, tmp_path)
    plan = {**PLAN, "speaker_changes": [11.0, 15.2]}
    render.render_clip(source, source, {"start": 10.0, "end": 16.0, "dead_air": False, "zoom": False},
                       WORDS, plan, tmp_path / "off.mp4")
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    top, bottom = render._split_crops(PLAN["segments"][1]["boxes"], 1920, 1080)
    expected = ("[0:v]split=2[source0][source1];"
                "[source0]trim=start=0.000000:end=2.500000,setpts=PTS-STARTPTS,crop=606:1080:272:0,scale=540:960,setsar=1[segment0];"
                "[source1]trim=start=2.500000:end=6.000000,setpts=PTS-STARTPTS,split=2[top1][bottom1];"
                f"[top1]{top},scale=540:480[t1];[bottom1]{bottom},scale=540:480[b1];[t1][b1]vstack=inputs=2,setsar=1[segment1];"
                "[segment0][segment1]concat=n=2:v=1:a=0,setsar=1,ass=filename='")
    assert graph.startswith(expected)
    assert graph.endswith("'[v];[1:a]atrim=duration=6.000000,asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    assert commands[0][commands[0].index("-ss") + 1] == "10.000000"
    before = caption_events(asses[0])
    render.write_ass(tmp_path / "today.ass", WORDS, 10.0, 16.0, "", True, False,
                     render.clipped_segments(PLAN["segments"], 10.0, 16.0))
    assert before == caption_events((tmp_path / "today.ass").read_text())


def test_edited_duration_outside_brief_range_stops_before_encoding(tmp_path, monkeypatch):
    source, commands, _ = fake_ffmpeg(monkeypatch, tmp_path)
    with pytest.raises(render.RenderError, match="edited duration 3.280s outside 5–180s"):
        render.render_clip(source, source, {"start": 10.0, "end": 16.0}, WORDS, {"kind": "letterbox"},
                           tmp_path / "short.mp4", proxy=False, duration_range=(5, 180))
    assert commands == []


def test_version_bump_forces_recompute(tmp_path):
    calls = []
    spec = StageSpec("S7", "attribution", "", lambda ctx: calls.append(1) or {"n": len(calls)})
    ctx = RunContext("r", tmp_path / "src", "hash", tmp_path, {})
    assert run_pipeline(ctx, [spec]).records[0].status == "ran"
    assert run_pipeline(ctx, [spec]).records[0].status == "cached"
    bumped = dataclasses.replace(spec, version=spec.version + 1)
    assert run_pipeline(ctx, [bumped]).records[0].status == "ran" and len(calls) == 2
    # The stages whose output this session changed were bumped.
    assert STAGES_BY_ID["S7"].version >= 4 and STAGES_BY_ID["S8"].version >= 4
    assert STAGES_BY_ID["S12"].version >= 3 and "S4" in STAGES_BY_ID["S7"].needs


def test_pacing_constants_are_config():
    assert (config.DEAD_AIR_GAP_S, config.DEAD_AIR_PAD_S, config.ZOOM_FACTOR,
            config.ZOOM_MIN_HOLD_S, config.ZOOM_PAUSE_S) == (0.8, 0.12, 1.15, 3.0, 1.5)
