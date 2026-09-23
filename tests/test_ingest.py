"""S0 ingest: URL detection, format selection, and download failure.

No network: yt-dlp's YoutubeDL is replaced wherever a test would reach out.
"""
from __future__ import annotations

import pytest

from maclips import ingest
from maclips.ingest import IngestError, SourceRecord


def test_url_detection():
    assert ingest.is_url("https://www.youtube.com/watch?v=abc")
    assert ingest.is_url("http://example.com/a.mp4")
    assert not ingest.is_url("/Users/me/video.mp4")
    assert not ingest.is_url("relative/path.mkv")
    assert not ingest.is_url("C:/videos/a.mp4")


def test_format_selector_honours_the_height_cap():
    sel = ingest._format_selector(1440)
    assert "height<=1440" in sel
    assert sel.count("height<=1440") == 3, "cap must apply to every fallback branch"
    assert sel.endswith("/b"), "a last-resort progressive fallback must exist"


def test_height_cap_is_a_run_argument_not_a_constant():
    assert "height<=720" in ingest._format_selector(720)
    assert "height<=2160" in ingest._format_selector(2160)


def test_local_path_resolves_without_network(tmp_path, tiny_av):
    record = ingest.resolve(str(tiny_av), tmp_path)
    assert record.path == tiny_av
    assert record.video_id == ""
    assert record.origin == str(tiny_av)


def test_missing_local_file_is_an_ingest_error(tmp_path):
    with pytest.raises(IngestError, match="no such file"):
        ingest.resolve(str(tmp_path / "nope.mp4"), tmp_path)


def test_download_failure_suggests_upgrading_yt_dlp(tmp_path, monkeypatch):
    """YouTube breakage almost always means yt-dlp is stale — say so, don't act."""
    from yt_dlp import DownloadError

    class Failing:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): raise DownloadError("nsig extraction failed")

    monkeypatch.setattr("yt_dlp.YoutubeDL", lambda *a, **k: Failing())
    with pytest.raises(IngestError) as excinfo:
        ingest.download("https://www.youtube.com/watch?v=x", tmp_path)

    message = str(excinfo.value)
    assert "uv lock --upgrade-package yt-dlp" in message
    assert "Not done automatically" in message, "must not auto-upgrade a pinned dep"


def test_probe_failure_also_suggests_the_upgrade(tmp_path, monkeypatch):
    from yt_dlp import DownloadError

    class Failing:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, *a, **k): raise DownloadError("HTTP 403")

    monkeypatch.setattr("yt_dlp.YoutubeDL", lambda *a, **k: Failing())
    with pytest.raises(IngestError, match="upgrade-package yt-dlp"):
        ingest.probe_url("https://www.youtube.com/watch?v=x")


def test_ffmpeg_location_is_the_configured_binary_never_path():
    """yt-dlp's merge step must not reach for the broken PATH ffmpeg (§2.3)."""
    from maclips import config

    opts = ingest._base_opts()
    assert opts["ffmpeg_location"] == str(config.FFMPEG)
    assert opts["ffmpeg_location"] != "ffmpeg"
    assert "/" in opts["ffmpeg_location"], "must be an absolute path"


def test_already_downloaded_url_is_a_cache_hit(tmp_path, monkeypatch):
    """Re-ingesting the same URL must not re-download; the video id is the key."""
    audio = tmp_path / "abc123.audio.m4a"
    video = tmp_path / "abc123.video.mp4"
    audio.write_bytes(b"a")
    video.write_bytes(b"v")

    monkeypatch.setattr(
        ingest, "probe_url",
        lambda url: SourceRecord(path=ingest.Path(), origin=url, video_id="abc123"),
    )

    def explode(*a, **k):
        raise AssertionError("download_split() must not run on a cache hit")

    monkeypatch.setattr(ingest, "download_split", explode)
    record = ingest.resolve("https://www.youtube.com/watch?v=abc123", tmp_path)
    assert record.audio_path == audio
    assert record.video_path == video
    assert record.audio_source == audio


def test_cache_hit_requires_both_streams(tmp_path):
    """Audio alone must not count: S6 would start on a video that never arrived."""
    (tmp_path / "abc123.audio.m4a").write_bytes(b"a")
    assert ingest.cached_streams("abc123", tmp_path) is None

    (tmp_path / "abc123.video.mp4").write_bytes(b"v")
    assert ingest.cached_streams("abc123", tmp_path) is not None


@pytest.mark.parametrize("leftover", [
    "abc123.video.mp4.part", "abc123.video.mp4.ytdl", "abc123.video.temp.mp4",
    "abc123.video.mp4.part-Frag3",
])
def test_cache_hit_ignores_yt_dlp_resume_and_temp_files(tmp_path, leftover):
    """§2.5: a resume file is not a stream, so audio plus a leftover is a miss."""
    (tmp_path / "abc123.audio.m4a").write_bytes(b"a")
    (tmp_path / leftover).write_bytes(b"partial")
    assert ingest.cached_streams("abc123", tmp_path) is None


def test_cache_hit_returns_the_complete_stream_beside_a_leftover(tmp_path):
    (tmp_path / "abc123.audio.m4a").write_bytes(b"a")
    (tmp_path / "abc123.audio.m4a.part").write_bytes(b"partial")
    (tmp_path / "abc123.video.mp4.ytdl").write_bytes(b"state")
    (tmp_path / "abc123.video.mp4").write_bytes(b"v")
    audio, video = ingest.cached_streams("abc123", tmp_path)
    assert (audio.name, video.name) == ("abc123.audio.m4a", "abc123.video.mp4")


def test_local_file_needs_no_split_and_never_waits(tmp_path, tiny_av):
    record = ingest.resolve(str(tiny_av), tmp_path)
    assert record.audio_path is None
    assert record.audio_source == tiny_av, "S2 decodes the file itself"
    assert record.video_ready, "a local file has nothing to wait for"


def test_audio_ready_callback_fires_for_local_files(tmp_path, tiny_av):
    """S2's start signal must work on both paths, or the caller needs a branch."""
    fired = []
    ingest.resolve(str(tiny_av), tmp_path, on_audio_ready=fired.append)
    assert fired == [tiny_av]


def test_await_video_rejects_a_source_that_was_never_split(tmp_path, tiny_av):
    record = ingest.resolve(str(tiny_av), tmp_path)
    with pytest.raises(IngestError, match="not downloaded as split streams"):
        ingest.await_video(record)


def test_source_record_as_dict_drops_the_formats_blob():
    record = SourceRecord(
        path=ingest.Path("/tmp/a.mp4"), origin="u", video_id="v",
        title="T", channel="C", duration_s=1.0, formats=[{"height": 1080}],
    )
    assert "formats" not in record.as_dict()
    assert record.as_dict()["title"] == "T"


# --------------------------------------------------------------------------- #
# Split-stream download
# --------------------------------------------------------------------------- #

def test_audio_selector_asks_for_audio_only():
    """S2 must not wait on pixels, so the first fetch carries no video."""
    assert "bv" not in ingest.AUDIO_SELECTOR
    assert ingest.AUDIO_SELECTOR.startswith("ba")


def test_video_selector_honours_the_height_cap():
    sel = ingest._video_selector(1440)
    assert "height<=1440" in sel
    assert "+ba" not in sel, "video stream must not drag audio along again"


def test_split_download_signals_audio_before_video_completes(tmp_path, monkeypatch):
    """The whole point: S2 starts on audio while video is still arriving."""
    import threading

    video_release = threading.Event()
    events = []

    def fake_stream(url, dest_dir, selector, tag):
        if tag == "video":
            video_release.wait(timeout=5)
            events.append("video-done")
            p = dest_dir / "vid.video.mp4"
        else:
            events.append("audio-done")
            p = dest_dir / "vid.audio.m4a"
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(ingest, "_download_stream", fake_stream)
    monkeypatch.setattr(
        ingest, "probe_url",
        lambda url: SourceRecord(path=ingest.Path(), origin=url, video_id="vid"),
    )

    record = ingest.download_split("https://x/y", tmp_path,
                                   on_audio_ready=lambda p: events.append("signalled"))

    assert events[:2] == ["audio-done", "signalled"], (
        "the audio signal must fire before the video download finishes"
    )
    assert "video-done" not in events
    assert record.audio_source == record.audio_path

    video_release.set()
    video = ingest.await_video(record, timeout=5)
    assert video.name == "vid.video.mp4"
    assert record.video_ready


def test_await_video_surfaces_a_failed_video_download(tmp_path, monkeypatch):
    def fake_stream(url, dest_dir, selector, tag):
        if tag == "video":
            raise IngestError("video download failed: 403")
        p = dest_dir / "vid.audio.m4a"
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(ingest, "_download_stream", fake_stream)
    monkeypatch.setattr(
        ingest, "probe_url",
        lambda url: SourceRecord(path=ingest.Path(), origin=url, video_id="vid"),
    )
    record = ingest.download_split("https://x/y", tmp_path)
    with pytest.raises(IngestError, match="403"):
        ingest.await_video(record, timeout=5)


def test_streams_are_not_muxed(tmp_path, monkeypatch):
    """Separate files by design; S12 takes them as two ffmpeg inputs."""
    def fake_stream(url, dest_dir, selector, tag):
        p = dest_dir / f"vid.{tag}.{'m4a' if tag == 'audio' else 'mp4'}"
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(ingest, "_download_stream", fake_stream)
    monkeypatch.setattr(
        ingest, "probe_url",
        lambda url: SourceRecord(path=ingest.Path(), origin=url, video_id="vid"),
    )
    record = ingest.download_split("https://x/y", tmp_path)
    ingest.await_video(record, timeout=5)
    assert record.audio_path != record.video_path
    assert record.audio_path.is_file() and record.video_path.is_file()
    d = record.as_dict()
    assert d["audio_path"] and d["video_path"]
