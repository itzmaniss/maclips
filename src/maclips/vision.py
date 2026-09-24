"""Candidate-only Apple Vision rectangles, shot-local IoU tracks; no attribution."""
from __future__ import annotations

import bisect
import io
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

from . import config, ffmpeg


class VisionError(RuntimeError):
    """Frame decode or Vision request failed; callers must stop the stage."""


@dataclass(frozen=True)
class FaceTrack:
    track_id: int
    shot_index: int
    # Top-left origin, normalised xywh, median over the track.
    median_box: tuple[float, float, float, float]
    first_frame: int
    last_frame: int
    mouth_signal: tuple[float, ...] = ()  # reserved shape; never computed here
    first_s: float = 0.0  # absolute source time
    last_s: float = 0.0
    observations: int = 0


def intersection_over_union(a: tuple, b: tuple) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    union = a[2]*a[3] + b[2]*b[3] - intersection
    return intersection / union if union > 0 else 0.0


def top_left_box(rect) -> tuple[float, float, float, float]:
    """Vision uses bottom-left; renderer/check images use top-left."""
    x, y = float(rect.origin.x), float(rect.origin.y)
    width, height = float(rect.size.width), float(rect.size.height)
    return x, 1-y-height, width, height


def _detect(png: bytes) -> list[tuple[float, float, float, float]]:
    import Foundation
    import Quartz
    import Vision

    data = Foundation.NSData.dataWithBytes_length_(png, len(png))
    image_source = Quartz.CGImageSourceCreateWithData(data, None)
    if image_source is None:
        raise VisionError("could not decode sampled PNG")
    image = Quartz.CGImageSourceCreateImageAtIndex(image_source, 0, None)
    request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, {})
    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise VisionError(f"Vision rectangle request failed: {error}")
    return [top_left_box(face.boundingBox()) for face in (request.results() or [])]


def associate_frames(frames: list[dict], fps: float, threshold: float = .3,
                     min_duration_s: float = 1.0) -> list[FaceTrack]:
    """Greedy one-to-one IoU matching, reset per shot and after >1 missed sample."""
    active: list[dict] = []
    finished: list[dict] = []
    serial = 0
    for frame in frames:
        keep = []
        for track in active:
            if track["shot"] != frame["shot_index"] or frame["time_s"]-track["last_s"] > 2.01/fps:
                finished.append(track)
            else:
                keep.append(track)
        active = keep
        boxes = frame["boxes"]
        pairs = sorted(((intersection_over_union(track["boxes"][-1], box), ti, bi)
                        for ti, track in enumerate(active) for bi, box in enumerate(boxes)), reverse=True)
        used_tracks, used_boxes = set(), set()
        for score, ti, bi in pairs:
            if score < threshold or ti in used_tracks or bi in used_boxes:
                continue
            track = active[ti]
            track["boxes"].append(boxes[bi])
            track["last_frame"], track["last_s"] = frame["frame"], frame["time_s"]
            used_tracks.add(ti); used_boxes.add(bi)
        for bi, box in enumerate(boxes):
            if bi in used_boxes:
                continue
            active.append({"id": serial, "shot": frame["shot_index"], "boxes": [box],
                           "first_frame": frame["frame"], "last_frame": frame["frame"],
                           "first_s": frame["time_s"], "last_s": frame["time_s"]})
            serial += 1
    finished.extend(active)
    result = []
    for track in finished:
        if track["last_s"] - track["first_s"] + 1e-8 < min_duration_s:
            continue
        box = tuple(median(b[i] for b in track["boxes"]) for i in range(4))
        result.append(FaceTrack(track["id"], track["shot"], box,
                                track["first_frame"], track["last_frame"], (),
                                track["first_s"], track["last_s"], len(track["boxes"])))
    return sorted(result, key=lambda t: t.track_id)


def _check_image(png: bytes, boxes: list, shot: int, time_s: float, path: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(png)).convert("RGB")
    draw = ImageDraw.Draw(image)
    for number, (x, y, width, height) in enumerate(boxes):
        corners = (x*image.width, y*image.height, (x+width)*image.width, (y+height)*image.height)
        draw.rectangle(corners, outline="lime", width=4)
        draw.text((corners[0]+5, corners[1]+5), str(number), fill="lime", stroke_width=1, stroke_fill="black")
    draw.text((15, 15), f"shot {shot} | source {time_s:.2f}s", fill="yellow", stroke_width=2, stroke_fill="black", font_size=24)
    image.thumbnail((1280, 1280))
    image.save(path)


def analyze_window(video_path: str | Path, start_s: float, end_s: float,
                   out_dir: str | Path, fps: float = 5.0) -> dict:
    if not all(math.isfinite(v) for v in (start_s, end_s, fps)) or start_s < 0 or end_s <= start_s or fps <= 0:
        raise VisionError("invalid candidate window or sampling rate")
    video, output = Path(video_path).resolve(), Path(out_dir).resolve()
    if not video.is_file():
        raise VisionError("original video is missing")
    output.mkdir(parents=True, exist_ok=True)
    begun = time.perf_counter()
    frames, images = [], []
    with tempfile.TemporaryDirectory(prefix="maclips-vision-") as temp:
        try:
            proc = ffmpeg._run([
                str(config.FFMPEG), "-y", "-nostdin", "-loglevel", "info",
                "-ss", f"{start_s:.6f}", "-i", str(video), "-t", f"{end_s-start_s:.6f}",
                "-an", "-vf", f"trim=duration={end_s-start_s:.6f},setpts=PTS-STARTPTS,scdet=threshold=10,metadata=print:key=lavfi.scd.time,fps={fps:g}",
                str(Path(temp) / "frame-%06d.png"),
            ])
        except ffmpeg.FFmpegError as exc:
            raise VisionError(str(exc)) from exc
        boundaries = sorted({float(v) for v in re.findall(r"lavfi\.scd\.time=([0-9.]+)", proc.stderr)
                             if 0 < float(v) < end_s-start_s})
        shots = [0.0, *boundaries]
        checked = set()
        for index, path in enumerate(sorted(Path(temp).glob("frame-*.png"))):
            relative = index / fps
            if relative >= end_s-start_s:
                break
            shot = bisect.bisect_right(shots, relative)-1
            png = path.read_bytes()
            import objc
            with objc.autorelease_pool():
                boxes = _detect(png)
            frames.append({"frame": index, "time_s": start_s+relative,
                           "shot_index": shot, "boxes": boxes})
            # First sample and then one per five seconds per shot, including no-face shots.
            check_key = (shot, int(relative // 5))
            if check_key not in checked:
                check = output / f"shot-{shot:03d}-frame-{index:06d}.jpg"
                _check_image(png, boxes, shot, start_s+relative, check)
                images.append(str(check)); checked.add(check_key)
        if not frames:
            raise VisionError("no sampled frames decoded from candidate window")
    tracks = associate_frames(frames, fps)
    return {"tracks": [asdict(track) for track in tracks],
            "shots": [{"shot_index": i, "start_s": start_s+t,
                       "end_s": start_s+(shots[i+1] if i+1 < len(shots) else end_s-start_s),
                       "relative_start_s": t} for i, t in enumerate(shots)],
            "coordinate_origin": "top-left", "time_basis": "absolute source seconds",
            "frames_processed": len(frames), "wall_s": time.perf_counter()-begun,
            "fps": fps, "check_images": images}


def detect_face_tracks(video_path: str, start_s: float, end_s: float,
                       fps: float = 5.0) -> list[FaceTrack]:
    with tempfile.TemporaryDirectory(prefix="maclips-face-checks-") as folder:
        result = analyze_window(video_path, start_s, end_s, folder, fps)
    return [FaceTrack(**track) for track in result["tracks"]]
