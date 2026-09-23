"""Static-layout single-pass rendering from original split video/audio streams."""
from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path

from . import config, ffmpeg


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
              hook: str, split: bool = False, diagnostic: bool = False) -> None:
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
            visible.append((max(a, start) - start, min(b, end) - start,
                            _text(word.get("text", word.get("word", "")))))
    for offset in range(0, len(visible), 6):
        phrase = visible[offset:offset + 6]
        for selected, (a, b, _) in enumerate(phrase):
            text = " ".join(("{\\c&H00FFFF&}" + w + "{\\c&HFFFFFF&}")
                            if i == selected else w for i, (_, _, w) in enumerate(phrase))
            lines.append(f"Dialogue: 0,{_ass_time(a)},{_ass_time(b)},Caption,,0,0,0,,{text}\n")
    if hook:
        lines.append(f"Dialogue: 1,0:00:00.00,{_ass_time(min(3, end-start))},Hook,,0,0,0,,{_text(hook)}\n")
    if diagnostic:
        lines.append(f"Dialogue: 2,0:00:00.00,{_ass_time(end-start)},Diagnostic,,0,0,0,,UNAPPROVED TEST RENDER\n")
    path.write_text("".join(lines))


def _centre(box: object) -> float:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise RenderError("split layout requires two normalised face boxes")
    x, y, width, height = [_number(v, "face box") for v in box]
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > 1.00001 or y + height > 1.00001:
        raise RenderError("face box lies outside source")
    return x + width / 2


def _crop(cx: float, aspect: float, width: int, height: int) -> str:
    # Full source height is preserved. Clamp horizontal crop to source bounds.
    crop_width = min(width, int(height * aspect) // 2 * 2)
    x = max(0, min(width - crop_width, round(cx * width - crop_width / 2))) // 2 * 2
    return f"crop={crop_width}:{height}:{x}:0"


def render_clip(video: Path, audio: Path, candidate: dict, words: list[dict],
                layout: dict | str, out_path: Path, *, proxy: bool = True,
                diagnostic: bool = False) -> dict:
    """Render once, probe hard gates, and atomically publish the complete MP4."""
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
    duration = end - start
    begun = time.perf_counter()
    try:
        source = ffmpeg.probe(video)
        width, height = int(source.get("width", 0)), int(source.get("height", 0))
        if not source.get("has_video") or min(width, height) <= 0:
            raise RenderError("source has no usable video stream")
        ow, oh = (540, 960) if proxy else (1080, 1920)
        base = f"[0:v]trim=duration={duration:.6f},setpts=PTS-STARTPTS"
        if kind == "letterbox":
            vf = base + f",scale={ow}:{oh}:force_original_aspect_ratio=decrease,pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:black"
        elif kind == "split":
            boxes = plan.get("boxes", [])
            if len(boxes) != 2:
                raise RenderError("split layout requires exactly two face boxes")
            centres = sorted(_centre(box) for box in boxes)
            vf = base + ",split=2[top][bottom];"
            vf += f"[top]{_crop(centres[0], 1080/960, width, height)},scale={ow}:{oh//2}[t];"
            vf += f"[bottom]{_crop(centres[1], 1080/960, width, height)},scale={ow}:{oh//2}[b];[t][b]vstack=inputs=2"
        else:
            cx = _number(plan.get("x", 0.5), "face centre") if kind == "face-centred" else 0.5
            if not 0 <= cx <= 1:
                raise RenderError("face centre lies outside source")
            vf = base + f",{_crop(cx, 9/16, width, height)},scale={ow}:{oh}"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        # Keep subtitle paths shell/filter-safe even if the user output path is not.
        with tempfile.TemporaryDirectory(prefix="maclips-render-") as scratch:
            ass = Path(scratch) / "captions.ass"
            write_ass(ass, words, start, end, candidate.get("hook_text", ""), kind == "split", diagnostic)
            vf += f",setsar=1,ass=filename='{ass}'[v];"
            vf += f"[1:a]atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
            fd, temporary = tempfile.mkstemp(prefix=".render-", suffix=".mp4", dir=outpath.parent)
            os.close(fd)
            partial = Path(temporary)
            try:
                command = [str(config.FFMPEG), "-y", "-nostdin", "-loglevel", "error",
                           "-ss", f"{start:.6f}", "-i", str(video),
                           "-ss", f"{start:.6f}", "-i", str(audio),
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
            "layout": kind, "proxy": proxy, "diagnostic": diagnostic}
