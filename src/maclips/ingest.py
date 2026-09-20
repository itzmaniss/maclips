"""S0 source ingest: local path or URL.

yt-dlp is used through its **Python API**, not as a subprocess of a PATH
binary. Two reasons: the PATH `ffmpeg` on this machine is a broken slim build,
and yt-dlp must be handed `ffmpeg_location` explicitly so its merge step uses
the same configured binary as every other stage (§2.3).

YouTube breaks extractors regularly. When a download fails the gate says so
and names the remedy, but never upgrades on its own — a silent dependency bump
mid-run would invalidate the lockfile the rest of the pipeline is pinned to.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config

# 9:16 is cut from the source's full *height* (§4.4), so height is what limits
# vertical quality — not the horizontal resolution people quote. A 9:16 crop
# from a source of height H is H*9/16 wide, so a 1080x1920 final needs H>=1920
# to avoid upscaling. 1440 is a middle default: better than 1080, far smaller
# than 2160. Override per run with --max-height.
DEFAULT_MAX_HEIGHT = 1440

URL_PATTERN = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


class IngestError(RuntimeError):
    """Ingest failed. Carries a remedy, because the usual cause is a stale yt-dlp."""


@dataclass
class SourceRecord:
    """What S0 registers about a source, local or remote.

    Audio and video are kept as **separate files** and never muxed. S2-S5 need
    only the audio, so merging would make the whole pipeline wait on the video
    download for no benefit; the final render (S12) takes the two as separate
    ffmpeg inputs, which costs nothing because it is already a filter_complex
    pass over both streams.

    For a local file, `path` is that file and `audio_path` is None — there is
    nothing to split and nothing to wait for.
    """

    path: Path
    origin: str  # the URL, or the local path
    audio_path: Path | None = None
    video_path: Path | None = None
    video_id: str = ""
    title: str = ""
    channel: str = ""
    duration_s: float = 0.0
    upload_date: str = ""
    width: int = 0
    height: int = 0
    formats: list[dict[str, Any]] = field(default_factory=list)

    @property
    def audio_source(self) -> Path:
        """What S2 decodes: the audio-only file when split, else the source."""
        return self.audio_path or self.path

    @property
    def video_ready(self) -> bool:
        """S6 needs pixels. S2-S5 do not, and must not wait for them."""
        return self.video_path is None or self.video_path.is_file()

    def as_dict(self) -> dict[str, Any]:
        """Checkpoint-safe view. `formats` is dropped: it is large and noisy."""
        return {
            "path": str(self.path),
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "video_path": str(self.video_path) if self.video_path else None,
            "origin": self.origin,
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "duration_s": self.duration_s,
            "upload_date": self.upload_date,
            "width": self.width,
            "height": self.height,
        }


def is_url(target: str) -> bool:
    return bool(URL_PATTERN.match(target.strip()))


def _format_selector(max_height: int) -> str:
    """Best video up to `max_height`, plus best audio, preferring an mp4 pair.

    Falls back to a progressive stream if no separate video/audio pair fits,
    which is what `/b` covers. Used only for the unsplit path.
    """
    return (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[height<={max_height}]+ba/"
        f"b[height<={max_height}]/b"
    )


AUDIO_SELECTOR = "ba[ext=m4a]/ba/b"
"""Audio only. Small and quick, which is the whole point: S2 can start on it
while the video is still downloading."""


def _video_selector(max_height: int) -> str:
    return f"bv*[height<={max_height}]/b[height<={max_height}]/b"


def _base_opts() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Never PATH: the merge/remux step must use the configured build.
        "ffmpeg_location": str(config.FFMPEG),
    }


def probe_url(url: str) -> SourceRecord:
    """Fetch metadata only — no media bytes.

    Run before downloading so a source can be judged (duration, speakers,
    available resolutions) before spending bandwidth on it.
    """
    from yt_dlp import DownloadError, YoutubeDL

    opts = {**_base_opts(), "skip_download": True}
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise IngestError(_stale_hint(f"metadata fetch failed for {url}", exc)) from exc

    formats = [
        {
            "width": f.get("width"),
            "height": f.get("height"),
            "vcodec": f.get("vcodec"),
            "acodec": f.get("acodec"),
            "filesize": f.get("filesize") or f.get("filesize_approx"),
        }
        for f in info.get("formats", [])
        if f.get("height")
    ]
    return SourceRecord(
        path=Path(),  # not downloaded yet
        origin=url,
        video_id=str(info.get("id") or ""),
        title=str(info.get("title") or ""),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        duration_s=float(info.get("duration") or 0.0),
        upload_date=str(info.get("upload_date") or ""),
        formats=formats,
    )


def _stale_hint(what: str, exc: Exception) -> str:
    return (
        f"{what}: {exc}\n"
        "  YouTube breakage almost always means yt-dlp is stale. Try:\n"
        "      uv lock --upgrade-package yt-dlp && uv sync\n"
        "  Not done automatically: it would move a pinned dependency mid-run."
    )


def download(url: str, dest_dir: Path, max_height: int = DEFAULT_MAX_HEIGHT) -> SourceRecord:
    """Download `url` into `dest_dir`, keyed by video id so re-ingest is a hit.

    The file is named by video id, not title: titles change and contain
    characters that make paths miserable, while the id is the stable identity
    the cache key is built from.
    """
    from yt_dlp import DownloadError, YoutubeDL

    dest_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        **_base_opts(),
        "format": _format_selector(max_height),
        "merge_output_format": "mp4/mkv",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except DownloadError as exc:
        raise IngestError(_stale_hint(f"download failed for {url}", exc)) from exc

    if not path.exists():
        # The merge step rewrites the extension; find what actually landed.
        candidates = sorted(dest_dir.glob(f"{info.get('id')}.*"))
        media = [c for c in candidates if c.suffix.lower() in {".mp4", ".mkv", ".webm"}]
        if not media:
            raise IngestError(
                f"download reported success but no media file for {info.get('id')!r} "
                f"in {dest_dir}"
            )
        path = media[0]

    return SourceRecord(
        path=path,
        origin=url,
        video_id=str(info.get("id") or ""),
        title=str(info.get("title") or ""),
        channel=str(info.get("channel") or info.get("uploader") or ""),
        duration_s=float(info.get("duration") or 0.0),
        upload_date=str(info.get("upload_date") or ""),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
    )


def _download_stream(url: str, dest_dir: Path, selector: str, tag: str) -> Path:
    """Fetch one stream into `<id>.<tag>.<ext>`."""
    from yt_dlp import DownloadError, YoutubeDL

    opts = {
        **_base_opts(),
        "format": selector,
        "outtmpl": str(dest_dir / f"%(id)s.{tag}.%(ext)s"),
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except DownloadError as exc:
        raise IngestError(_stale_hint(f"{tag} download failed for {url}", exc)) from exc

    if not path.exists():
        found = sorted(dest_dir.glob(f"{info.get('id')}.{tag}.*"))
        if not found:
            raise IngestError(f"{tag} download reported success but produced no file")
        path = found[0]
    return path


def download_split(
    url: str,
    dest_dir: Path,
    max_height: int = DEFAULT_MAX_HEIGHT,
    on_audio_ready: Callable[[Path], None] | None = None,
) -> SourceRecord:
    """Download audio and video as separate files, audio first.

    Audio is fetched on this thread and `on_audio_ready` fires the moment it
    lands, so S2 can start while the video is still arriving. Video downloads
    on a background thread; `record.video_path` is only guaranteed once
    `await_video()` returns.

    Nothing is muxed. S12 takes the two files as separate ffmpeg inputs.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    meta = probe_url(url)

    video_result: dict[str, Any] = {}

    def fetch_video() -> None:
        try:
            video_result["path"] = _download_stream(
                url, dest_dir, _video_selector(max_height), "video"
            )
        except IngestError as exc:  # re-raised by await_video on the caller's thread
            video_result["error"] = exc

    video_thread = threading.Thread(target=fetch_video, daemon=True)
    video_thread.start()

    audio_path = _download_stream(url, dest_dir, AUDIO_SELECTOR, "audio")
    if on_audio_ready is not None:
        on_audio_ready(audio_path)

    meta.audio_path = audio_path
    meta.path = audio_path  # until the video lands, the audio *is* the source
    meta._video_thread = video_thread  # type: ignore[attr-defined]
    meta._video_result = video_result  # type: ignore[attr-defined]
    return meta


def await_video(record: SourceRecord, timeout: float | None = None) -> Path:
    """Block until the video stream has landed. The S6 gate calls this.

    S2-S5 must never call it: the point of splitting is that they do not wait.
    """
    thread = getattr(record, "_video_thread", None)
    if thread is None:
        if record.video_path is not None:
            return record.video_path
        raise IngestError("this source was not downloaded as split streams")

    thread.join(timeout=timeout)
    if thread.is_alive():
        raise IngestError(f"video download still running after {timeout}s")

    result = getattr(record, "_video_result", {})
    if "error" in result:
        raise result["error"]
    if "path" not in result:
        raise IngestError("video download produced no file")
    record.video_path = result["path"]
    return record.video_path


def resolve(
    target: str,
    dest_dir: Path,
    max_height: int = DEFAULT_MAX_HEIGHT,
    on_audio_ready: Callable[[Path], None] | None = None,
) -> SourceRecord:
    """Accept a local path or a URL and return a registered source either way.

    URLs are fetched as split streams (audio first, video concurrently). A URL
    whose streams are already on disk is a cache hit keyed by video id.

    A local file is used as-is: there is nothing to split, `audio_path` stays
    None, and `audio_source` falls back to the file itself.
    """
    if not is_url(target):
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise IngestError(f"no such file: {path}")
        record = SourceRecord(path=path, origin=str(path))
        if on_audio_ready is not None:
            on_audio_ready(path)
        return record

    meta = probe_url(target)
    if meta.video_id:
        cached = cached_streams(meta.video_id, dest_dir)
        if cached is not None:
            audio, video = cached
            meta.audio_path, meta.video_path = audio, video
            meta.path = audio
            return meta
    return download_split(target, dest_dir, max_height=max_height,
                          on_audio_ready=on_audio_ready)


def cached_streams(video_id: str, dest_dir: Path) -> tuple[Path, Path] | None:
    """Both streams already on disk? Then ingest is a no-op.

    Requires *both*: an audio file without its video would let S6 start on a
    source whose pixels never arrived.
    """
    audio = [p for p in dest_dir.glob(f"{video_id}.audio.*") if p.is_file()]
    video = [p for p in dest_dir.glob(f"{video_id}.video.*") if p.is_file()]
    if audio and video:
        return audio[0], video[0]
    return None
