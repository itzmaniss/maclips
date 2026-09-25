"""Static-layout single-pass rendering from original split video/audio streams."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

from . import config, ffmpeg
from .layouts import split_divider
from .pacing import cold_open_keep, combined_words, edited_time, half_frame, padded_span, plan_pacing, video_pieces
from .ranking import cold_open_choice


class RenderError(RuntimeError):
    """Invalid render plan, failed encoder, or failed output gate."""


def _number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RenderError(f"invalid {name}") from exc
    if not math.isfinite(result):
        raise RenderError(f"invalid {name}")
    return result


def validate_output(probe: dict, duration: float, proxy: bool = False) -> None:
    expected = (540, 960) if proxy else (1080, 1920)
    actual = _number(probe.get("duration_s"), "output duration")
    if abs(actual - duration) > 0.1 + 1e-8:
        raise RenderError(f"duration drift: planned {duration:.3f}s, actual {actual:.3f}s")
    if (probe.get("width"), probe.get("height")) != expected:
        raise RenderError(f"wrong resolution: expected {expected[0]}x{expected[1]}")
    if not probe.get("has_audio"):
        raise RenderError("rendered without an audio stream")


def _ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, rem = divmod(centis, 360000)
    minutes, rem = divmod(rem, 6000)
    seconds, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02}:{seconds:02}.{centis:02}"


def _text(value: object) -> str:
    # User text must not introduce ASS override tags or escape sequences.
    return str(value).replace("\\", "／").replace("{", "(").replace("}", ")").replace("\n", " ").replace("\r", " ")


def write_ass(path: Path, words: list[dict], start: float, end: float,
              hook: str, split: bool = False, diagnostic: bool = False,
              segments: list[dict] | None = None, end_card: str = "",
              phrase_break: float | None = None) -> None:
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{config.CAPTION_FONT},58,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,65,65,{880 if split else 230},1
Style: Hook,{config.CAPTION_FONT},65,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,8,80,80,180,1
Style: Diagnostic,{config.CAPTION_FONT},32,&H0000FFFF,&H0000FFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,8,40,40,40,1
{END_CARD_STYLE}
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    visible = []
    for word in words:
        if word.get("start") is None or word.get("end") is None:
            continue
        a, b = float(word["start"]), float(word["end"])
        if math.isfinite(a) and math.isfinite(b) and b > start and a < end and b > a:
            visible.append((max(a, start) - start, _text(word.get("text", word.get("word", "")))))
    visible.sort(key=lambda item: item[0])
    # Each word holds until the next word starts, and the last until clip end,
    # so the caption never blinks off between words or across pauses.
    visible = [(a, b, w) for (a, w), b in zip(visible, [a for a, _ in visible[1:]] + [end - start])]
    # Phrases of up to 6 words; a new phrase always starts at `phrase_break`
    # (the cold-open join), so the line's words never linger after the cut.
    phrases: list[list] = []
    for item in visible:
        crossing = (phrase_break is not None and bool(phrases)
                    and phrases[-1][-1][0] < phrase_break - 1e-6 <= item[0])
        if not phrases or len(phrases[-1]) == 6 or crossing:
            phrases.append([])
        phrases[-1].append(item)
    for phrase in phrases:
        for selected, (a, b, _) in enumerate(phrase):
            text = " ".join(("{\\c&H00FFFF&}" + w + "{\\c&HFFFFFF&}")
                            if i == selected else w for i, (_, _, w) in enumerate(phrase))
            spans = [(a, b, 880 if split else 230)] if not segments else [
                (max(a, segment["start"]-start), min(b, segment["end"]-start),
                 880 if segment["kind"] == "split" else 230)
                for segment in segments if segment["end"] > start+a and segment["start"] < start+b]
            for left, right, margin in spans:
                if right > left:
                    lines.append(f"Dialogue: 0,{_ass_time(left)},{_ass_time(right)},Caption,,0,0,{margin},,{text}\n")
    if hook:
        lines.append(f"Dialogue: 1,0:00:00.00,{_ass_time(min(3, end-start))},Hook,,0,0,0,,{_text(hook)}\n")
    if diagnostic:
        lines.append(f"Dialogue: 2,0:00:00.00,{_ass_time(end-start)},Diagnostic,,0,0,0,,UNAPPROVED TEST RENDER\n")
    # END-CARD TIMING DEPENDENCY: `end - start` must be the duration the
    # renderer actually outputs. If cuts shorten the timeline, pass that value.
    lines += _end_card_events(end_card, end - start, bool(hook))
    path.write_text("".join(lines))


# Top band above faces and below the diagnostic label; captions sit at the
# bottom (MarginV 230) or the split seam (880), never up here (PLAN.md §6.6).
END_CARD_STYLE = (f"Style: EndCard,{config.CAPTION_FONT},52,&H00FFFFFF,&H0000FFFF,&H00101010,"
                  "&H80000000,-1,0,0,0,100,100,0,0,1,3,1,8,80,80,100,1")


def end_card_text(clip: dict) -> str:
    """The burned-in share prompt for this clip, or "" when the card is off."""
    if clip.get("end_card") is not True:
        return ""
    text = str(clip.get("end_card_text") or config.END_CARD_TEXT).strip()
    if not text or len(text) > config.END_CARD_MAX_CHARS:
        raise RenderError(f"end card text must be 1-{config.END_CARD_MAX_CHARS} characters")
    return text


def _end_card_events(text: str, duration: float, has_hook: bool) -> list[str]:
    """Final END_CARD_SECONDS of the rendered duration; never alongside the hook."""
    if not text:
        return []
    hook_end = min(3, duration) if has_hook else 0.0  # matches the Hook event
    begin = duration - config.END_CARD_SECONDS
    if begin < hook_end:
        raise RenderError(f"clip too short for a {config.END_CARD_SECONDS:g}s end card after the hook")
    return [f"Dialogue: 1,{_ass_time(begin)},{_ass_time(duration)},EndCard,,0,0,0,,{_text(text)}\n"]


def _box(box: object) -> tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RenderError("split layout requires two normalised face boxes")
    x, y, width, height = [_number(v, "face box") for v in box]
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1.00001 or y + height > 1.00001:
        raise RenderError("face box lies outside source")
    return x, y, width, height


SPLIT_FACE_Y = 0.4  # face centre this far down its panel ("upper-middle") [I]


def _split_crops(boxes: object, width: int, height: int) -> list[str]:
    """Left-in-source face first. The two crops never overlap horizontally.

    Each crop is as wide as its side of the divider allows (capped by full
    height at 1080:960), placed on its face and clamped to its side and frame.
    """
    if not isinstance(boxes, list) or len(boxes) != 2:
        raise RenderError("split layout requires exactly two face boxes")
    faces = sorted((_box(box) for box in boxes), key=lambda b: b[0] + b[2] / 2)
    divider = split_divider(faces)
    if divider is None:
        raise RenderError("split faces are too close for non-overlapping panels")
    divider = int(divider * width) // 2 * 2
    crops = []
    for (x, y, w, h), low, high in ((faces[0], 0, divider), (faces[1], divider, width)):
        crop_width = min(high - low, int(height * 1080 / 960)) // 2 * 2
        crop_height = min(height, int(crop_width * 960 / 1080)) // 2 * 2
        left = max(low, min(high - crop_width, round((x + w / 2) * width - crop_width / 2))) // 2 * 2
        top = max(0, min(height - crop_height, round((y + h / 2) * height - SPLIT_FACE_Y * crop_height))) // 2 * 2
        crops.append(f"crop={crop_width}:{crop_height}:{left}:{top}")
    return crops


def _crop(cx: float, aspect: float, width: int, height: int) -> str:
    # Face-centred/centre only: full source height, horizontal clamp to source.
    crop_width = min(width, int(height * aspect) // 2 * 2)
    x = max(0, min(width - crop_width, round(cx * width - crop_width / 2))) // 2 * 2
    return f"crop={crop_width}:{height}:{x}:0"


def _zoom_crop(cx: float, cy: float, width: int, height: int, zoom: float) -> str:
    """The normal full-height crop scaled by 1/zoom about the face centre.

    The face keeps its on-screen position and grows by `zoom`; the crop is then
    clamped to the frame.
    """
    crop_height = int(height / zoom) // 2 * 2
    crop_width = min(width, int(crop_height * 9 / 16) // 2 * 2)
    base_width = min(width, int(height * 9 / 16) // 2 * 2)
    base_x = max(0, min(width - base_width, round(cx * width - base_width / 2))) // 2 * 2
    fx, fy = cx * width, cy * height
    x = max(0, min(width - crop_width, round(fx - (fx - base_x) / zoom))) // 2 * 2
    y = max(0, min(height - crop_height, round(fy - fy / zoom))) // 2 * 2
    return f"crop={crop_width}:{crop_height}:{x}:{y}"


def _zoom_fits(segment: dict, width: int, height: int) -> bool:
    """Hard rule: a face-centred shot zooms only if its whole face box stays in
    the zoomed crop. A close-up whose face is wider than the crop keeps normal
    framing. Centre crops have no face box and always may zoom."""
    if segment.get("kind") != "face-centred" or not segment.get("box"):
        return True
    bx, by, bw, bh = _box(segment["box"])
    cw, ch, x, y = map(int, _zoom_crop(_number(segment.get("x", .5), "face centre"),
                                       _number(segment.get("y", .4), "face centre"),
                                       width, height, config.ZOOM_FACTOR)[5:].split(":"))
    return x <= bx * width and (bx + bw) * width <= x + cw and y <= by * height and (by + bh) * height <= y + ch



def clipped_segments(segments: list[dict], start: float, end: float) -> list[dict]:
    """Allow trimming inside a saved plan, but never leave a gap or overlap."""
    if not isinstance(segments, list) or not segments:
        raise RenderError("layout segments must be a nonempty list")
    clipped = []
    previous = None
    for segment in segments:
        if not isinstance(segment, dict):
            raise RenderError("invalid layout segment")
        a, b = _number(segment.get("start"), "segment start"), _number(segment.get("end"), "segment end")
        if a < 0 or b <= a or (previous is not None and a < previous-1e-6):
            raise RenderError("layout segments overlap or are unordered")
        previous = b
        if segment.get("kind") not in {"face-centred", "split", "letterbox", "centre", "centre-crop"}:
            raise RenderError("unknown segment layout")
        a, b = max(a, start), min(b, end)
        if b > a:
            clipped.append({**segment, "start": a, "end": b})
    cursor = start
    for segment in clipped:
        if abs(segment["start"]-cursor) > 1e-6:
            raise RenderError("layout segments do not cover clip continuously")
        cursor = segment["end"]
    if abs(cursor-end) > 1e-6:
        raise RenderError("layout segments do not cover clip continuously")
    return clipped


def _layout(segment: dict, i: int, width: int, height: int, ow: int, oh: int, zoomed: bool = False) -> str:
    kind = segment["kind"]
    if kind == "letterbox":
        return f",scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"
    if kind == "split":
        top, bottom = _split_crops(segment.get("boxes"), width, height)
        return (f",split=2[top{i}][bottom{i}];[top{i}]{top},scale={ow}:{oh//2}[t{i}];"
                f"[bottom{i}]{bottom},scale={ow}:{oh//2}[b{i}];[t{i}][b{i}]vstack=inputs=2")
    cx = _number(segment.get("x", .5), "face centre") if kind == "face-centred" else .5
    cy = _number(segment.get("y", .4), "face centre") if kind == "face-centred" else .4
    if not (0 <= cx <= 1 and 0 <= cy <= 1):
        raise RenderError("face centre lies outside source")
    crop = _zoom_crop(cx, cy, width, height, config.ZOOM_FACTOR) if zoomed else _crop(cx, 9/16, width, height)
    return f",{crop},scale={ow}:{oh}"


# The cold-open separator: a hard cut, with the clip's first frame drawn white [I].
FLASH = ",drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(n,0)'"


def _piece_graph(pieces, ss: float, width: int, height: int, ow: int, oh: int,
                 flash: int | None = None) -> str:
    count = len(pieces)
    graph = "[0:v]split=" + str(count) + "".join(f"[source{i}]" for i in range(count)) + ";"
    for i, (a, b, segment, zoomed) in enumerate(pieces):
        graph += f"[source{i}]trim=start={a-ss:.6f}:end={b-ss:.6f},setpts=PTS-STARTPTS"
        graph += _layout(segment, i, width, height, ow, oh, zoomed) + (FLASH if i == flash else "")
        graph += f",setsar=1[segment{i}];"
    return graph + "".join(f"[segment{i}]" for i in range(count)) + f"concat=n={count}:v=1:a=0"


def _segment_graph(segments: list[dict], start: float, width: int, height: int,
                   ow: int, oh: int) -> str:
    return _piece_graph([(s["start"], s["end"], s, False) for s in segments], start, width, height, ow, oh)


def _audio_graph(keeps: list[tuple[float, float]], ss: float) -> str:
    loud = "loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    if len(keeps) == 1:
        return f"[1:a]atrim=duration={keeps[0][1]-keeps[0][0]:.6f},asetpts=PTS-STARTPTS,{loud}"
    fade, count = config.DEAD_AIR_XFADE_S, len(keeps)
    # Each keep but the last takes `fade` more from the next cut's silent pad;
    # acrossfade overlaps exactly that much, so no join shifts the audio.
    graph = f"[1:a]asplit={count}" + "".join(f"[x{i}]" for i in range(count)) + ";"
    for i, (a, b) in enumerate(keeps):
        tail = fade if i < count - 1 else 0.0
        graph += f"[x{i}]atrim=start={a-ss:.6f}:end={b-ss+tail:.6f},asetpts=PTS-STARTPTS[p{i}];"
    return graph + "".join(f"[p{i}]" for i in range(count)) + f"acrossfade=n={count}:d={fade}:c1=tri:c2=tri,{loud}"


def _frame_grid(video: Path) -> tuple[float, float]:
    """(fps, first-frame offset) of the source video, as ffmpeg's -ss sees it."""
    proc = ffmpeg._run([str(config.FFPROBE), "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=r_frame_rate,start_time:format=start_time",
                        "-of", "json", str(video)], timeout=120.0)
    try:
        data = json.loads(proc.stdout)
        num, _, den = data["streams"][0]["r_frame_rate"].partition("/")
        fps = float(num) / float(den or 1)
        origin = float(data["streams"][0].get("start_time") or 0) - float(data.get("format", {}).get("start_time") or 0)
    except (KeyError, IndexError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        raise RenderError("could not read the source frame rate") from exc
    if not (math.isfinite(fps) and fps > 0):
        raise RenderError("could not read the source frame rate")
    return fps, origin


def render_clip(video: Path, audio: Path, candidate: dict, words: list[dict],
                layout: dict | str, out_path: Path, *, proxy: bool = True,
                diagnostic: bool = False,
                duration_range: tuple[float, float] | None = None) -> dict:
    """Render once, probe hard gates, and atomically publish the complete MP4.

    The candidate's `dead_air` and `zoom` toggles (missing = on) edit the
    timeline inside the same single pass. A `cold_open` of tease or payoff
    plays that line first, then a white-flash cut, then the whole clip. The
    planned duration, which the output gate checks, is the combined edited
    duration; `duration_range` gates it before encoding.
    """
    start = _number(candidate.get("start"), "start")
    end = _number(candidate.get("end"), "end")
    if start < 0 or end <= start:
        raise RenderError("invalid clip span")
    plan = {"kind": layout} if isinstance(layout, str) else dict(layout)
    kind = plan.get("kind", plan.get("layout"))
    if kind not in {"centre", "centre-crop", "letterbox", "face-centred", "split"}:
        raise RenderError(f"unknown layout: {kind}")
    video, audio, outpath = Path(video).resolve(), Path(audio).resolve(), Path(out_path).resolve()
    if not video.is_file() or not audio.is_file():
        raise RenderError("original video or audio is missing")
    if outpath in (video, audio):
        raise RenderError("output may not replace an original input")
    segments = clipped_segments(plan["segments"], start, end) if "segments" in plan else None
    start, end = padded_span(words, start, end)
    if segments:
        segments[0]["start"], segments[-1]["end"] = start, end
    begun = time.perf_counter()
    try:
        source = ffmpeg.probe(video)
        width, height = int(source.get("width", 0)), int(source.get("height", 0))
        if not source.get("has_video") or min(width, height) <= 0:
            raise RenderError("source has no usable video stream")
        pacing = plan_pacing(candidate, words, plan, start, end, grid=lambda: _frame_grid(video))
        keeps, switches, duration = pacing["keeps"], pacing["switches"], pacing["duration"]
        try:
            cold = cold_open_choice(candidate)
        except ValueError as exc:
            raise RenderError(str(exc)) from exc
        lead = []
        if cold:
            if pacing["fps"] is None:
                pacing["fps"], pacing["origin"] = _frame_grid(video)
            lead = cold_open_keep(float(cold[1]["start"]), float(cold[1]["end"]), pacing["fps"], pacing["origin"])
            if not lead:
                raise RenderError("cold open line has no whole frame")
        # Everything below runs on the combined timeline: the line, then the clip.
        lead_s = sum(b - a for a, b in lead)
        order, duration = lead + keeps, lead_s + duration
        switches = [lead_s + t for t in switches]
        ss = min(a for a, _ in order)
        if duration_range and not duration_range[0] <= duration <= duration_range[1]:
            raise RenderError(f"edited duration {duration:.3f}s outside "
                              f"{duration_range[0]:g}–{duration_range[1]:g}s")
        ow, oh = (540, 960) if proxy else (1080, 1920)
        base = f"[0:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS"
        pieces = []
        if len(order) > 1 or switches:
            if pacing["fps"] is None:
                pacing["fps"], pacing["origin"] = _frame_grid(video)
            fps, origin = pacing["fps"], pacing["origin"]
            spans = [dict(s) for s in segments] if segments else [{**plan, "kind": kind, "start": start, "end": end}]
            spans[0]["start"], spans[-1]["end"] = min(spans[0]["start"], ss), max(spans[-1]["end"], *(b for _, b in order))
            for span in spans:
                span["zoom_ok"] = _zoom_fits(span, width, height)
            pieces = video_pieces(order, spans, switches, lambda t: half_frame(t, fps, origin))
            placed = [0.0]
            for a, b, _, _ in pieces:
                placed.append(placed[-1] + b - a)
            flash = next(i for i, t in enumerate(placed) if t >= lead_s - 1e-6) if lead else None
            vf = _piece_graph(pieces, ss, width, height, ow, oh, flash)
        elif segments:
            vf = _segment_graph(segments, start, width, height, ow, oh)
        elif kind == "letterbox":
            vf = base + f",scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"
        elif kind == "split":
            top, bottom = _split_crops(plan.get("boxes"), width, height)
            vf = base + ",split=2[top][bottom];"
            vf += f"[top]{top},scale={ow}:{oh//2}[t];"
            vf += f"[bottom]{bottom},scale={ow}:{oh//2}[b];[t][b]vstack=inputs=2"
        else:
            cx = _number(plan.get("x", 0.5), "face centre") if kind == "face-centred" else 0.5
            if not 0 <= cx <= 1:
                raise RenderError("face centre lies outside source")
            vf = base + f",{_crop(cx, 9/16, width, height)},scale={ow}:{oh}"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        # Keep subtitle paths shell/filter-safe even if the user output path is not.
        with tempfile.TemporaryDirectory(prefix="maclips-render-") as scratch:
            ass = Path(scratch) / "captions.ass"
            if lead:
                # Captions, layout margins, hook and end card on the combined timeline;
                # margins follow the pieces actually shown.
                mapped = [{**s, "start": placed[i], "end": placed[i + 1]} for i, (_, _, s, _) in enumerate(pieces)] if segments else None
                write_ass(ass, combined_words(words, lead, keeps), 0.0, duration, candidate.get("hook_text", ""),
                          kind == "split", diagnostic, mapped, end_card=end_card_text(candidate), phrase_break=lead_s)
            elif len(keeps) > 1:
                # Captions, layout margins and the hook all live on the edited timeline.
                timed = [{**w, "start": edited_time(float(w["start"]), keeps), "end": edited_time(float(w["end"]), keeps)}
                         for w in words if w.get("start") is not None and w.get("end") is not None]
                mapped = None
                if segments:
                    mapped = [{**s, "start": 0.0 if i == 0 else edited_time(s["start"], keeps),
                               "end": duration if i == len(segments) - 1 else edited_time(s["end"], keeps)}
                              for i, s in enumerate(segments)]
                    mapped = [s for s in mapped if s["end"] > s["start"]]
                write_ass(ass, timed, 0.0, duration, candidate.get("hook_text", ""), kind == "split", diagnostic, mapped,
                          end_card=end_card_text(candidate))
            else:
                write_ass(ass, words, start, end, candidate.get("hook_text", ""), kind == "split", diagnostic, segments,
                          end_card=end_card_text(candidate))
            vf += f",setsar=1,ass=filename='{ass}'[v];"
            vf += _audio_graph(order, ss)
            fd, temporary = tempfile.mkstemp(prefix=".render-", suffix=".mp4", dir=outpath.parent)
            os.close(fd)
            partial = Path(temporary)
            try:
                command = [str(config.FFMPEG), "-y", "-nostdin", "-loglevel", "error",
                           "-ss", f"{ss:.6f}", "-i", str(video),
                           "-ss", f"{ss:.6f}", "-i", str(audio),
                           "-filter_complex", vf, "-map", "[v]", "-map", "[a]",
                           "-c:v", "h264_videotoolbox" if proxy else "libx264"]
                command += ["-b:v", "2500k"] if proxy else ["-preset", "fast", "-crf", "18"]
                command += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                            "-ar", "48000", "-movflags", "+faststart", str(partial)]
                ffmpeg._run(command)
                measured = ffmpeg.probe(partial)
                validate_output(measured, duration, proxy)
                partial.replace(outpath)
            finally:
                partial.unlink(missing_ok=True)
    except ffmpeg.FFmpegError as exc:
        raise RenderError(str(exc)) from exc
    return {"path": str(outpath), "planned_duration_s": duration,
            "probe": measured, "actual_duration_s": measured["duration_s"],
            "width": measured["width"], "height": measured["height"],
            "has_audio": measured["has_audio"], "wall_s": time.perf_counter() - begun,
            "layout": kind, "proxy": proxy, "diagnostic": diagnostic,
            "source_span_s": end - start,
            "pacing": {"dead_air": pacing["dead_air"], "zoom": pacing["zoom"],
                       "cuts": len(keeps) - 1, "removed_s": (end - start) - (duration - lead_s),
                       "keeps": keeps, "triggers": pacing["triggers"], "switches": switches,
                       "fps": pacing["fps"],
                       "pieces": [{"source": [a, b], "edited_start": edited_time(a, keeps) if not lead else placed[i],
                                   "kind": s["kind"], "zoomed": z} for i, (a, b, s, z) in enumerate(pieces)]},
            "cold_open": {"mode": cold[0], "source": list(lead[0]), "duration_s": lead_s,
                          "text": cold[1].get("text", "")} if lead else None}
