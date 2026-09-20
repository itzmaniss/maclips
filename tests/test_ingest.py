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
    existing = tmp_path / "abc123.mp4"
    existing.write_bytes(b"x")

    monkeypatch.setattr(
        ingest, "probe_url",
        lambda url: SourceRecord(path=ingest.Path(), origin=url, video_id="abc123"),
    )

    def explode(*a, **k):
        raise AssertionError("download() must not run on a cache hit")

    monkeypatch.setattr(ingest, "download", explode)
    record = ingest.resolve("https://www.youtube.com/watch?v=abc123", tmp_path)
    assert record.path == existing


def test_source_record_as_dict_drops_the_formats_blob():
    record = SourceRecord(
        path=ingest.Path("/tmp/a.mp4"), origin="u", video_id="v",
        title="T", channel="C", duration_s=1.0, formats=[{"height": 1080}],
    )
    assert "formats" not in record.as_dict()
    assert record.as_dict()["title"] == "T"
