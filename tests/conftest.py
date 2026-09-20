"""Shared fixtures.

Media fixtures are generated locally with the configured ffmpeg — short, and
never the benchmark file. No test in this suite touches the network or loads a
Whisper/pyannote model.
"""
from __future__ import annotations

import subprocess

import pytest

from maclips import config


@pytest.fixture(scope="session")
def tiny_av(tmp_path_factory):
    """A real 3-second A/V file: enough for ffprobe and ffmpeg to be genuine."""
    path = tmp_path_factory.mktemp("media") / "tiny.mp4"
    subprocess.run(
        [str(config.FFMPEG), "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         str(path)],
        check=True, capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def video_only(tmp_path_factory):
    """A file with no audio stream — S0 must gate on it."""
    path = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run(
        [str(config.FFMPEG), "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc=size=320x180:rate=15:duration=3",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return path
