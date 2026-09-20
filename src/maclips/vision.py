"""Apple Vision face detection — stub only this session.

Build step 6 implements this. It replaces the fork's Haar cascade, which was
frontal-only and picked the largest face regardless of who was speaking.

The shape build step 6 fills in (PLAN.md §4.4):
  * `VNDetectFaceLandmarksRequest` at ~5 fps inside candidate windows only —
    ~4,500 frames for 15 clips, against ~36,000 for a full 2-hour source;
  * associate detections across frames by IoU *within a shot* (`scdet` marks
    the boundaries), dropping tracks under ~1 s;
  * per face per frame, inner-lip vertical opening normalised by face-box
    height — the mouth signal S7 correlates against diarized turns.

Nothing here loads or ships weights: Vision is an OS framework.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceTrack:
    """One face followed across frames within a single shot."""

    track_id: int
    shot_index: int
    # (x, y, w, h) in normalised source coordinates, median over the track.
    median_box: tuple[float, float, float, float]
    first_frame: int
    last_frame: int
    # Inner-lip opening per sampled frame, normalised by face-box height.
    mouth_signal: tuple[float, ...] = ()


def detect_face_tracks(
    video_path: str,
    start_s: float,
    end_s: float,
    fps: float = 5.0,
) -> list[FaceTrack]:
    """Not implemented until build step 6."""
    raise NotImplementedError(
        "Apple Vision face tracking lands in build step 6 (PLAN.md §8). "
        "Until then S6 is a stub and every clip is letterbox-only."
    )
