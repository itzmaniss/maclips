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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """What S0 registers about a source, local or remote."""

    path: Path
    origin: str  # the URL, or the local path
    video_id: str = ""
    title: str = ""
    channel: str = ""
    duration_s: float = 0.0
    upload_date: str = ""
    width: int = 0
    height: int = 0
    formats: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Checkpoint-safe view. `formats` is dropped: it is large and noisy."""
        return {
            "path": str(self.path),
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
    which is what `/b` covers.
    """
    return (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[height<={max_height}]+ba/"
        f"b[height<={max_height}]/b"
    )


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


def resolve(target: str, dest_dir: Path, max_height: int = DEFAULT_MAX_HEIGHT) -> SourceRecord:
    """Accept a local path or a URL and return a registered source either way.

    A URL whose media is already on disk is a cache hit: the video id names the
    file, so re-ingesting the same URL skips the download.
    """
    if not is_url(target):
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise IngestError(f"no such file: {path}")
        return SourceRecord(path=path, origin=str(path))

    meta = probe_url(target)
    if meta.video_id:
        existing = [
            p for p in dest_dir.glob(f"{meta.video_id}.*")
            if p.suffix.lower() in {".mp4", ".mkv", ".webm"}
        ]
        if existing:
            meta.path = existing[0]
            return meta
    return download(target, dest_dir, max_height=max_height)
