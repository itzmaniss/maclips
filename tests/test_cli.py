"""`maclips run`: the CLI's cache keys stay stable across real re-runs.

No network and no models: `resolve` and the environment gate are replaced, and
S0's body is a stub. S0's params, and therefore its cache key, are the real ones.
"""
from __future__ import annotations

import dataclasses
import sys

from maclips import cli, config
from maclips.ingest import SourceRecord
from maclips.stages import STAGES_BY_ID

URL = "https://www.youtube.com/watch?v=vid"


def _run(monkeypatch, capsys, record, *extra):
    monkeypatch.setattr(cli, "resolve", lambda *a, **k: record)
    monkeypatch.setattr(sys, "argv", ["maclips", "run", URL, *extra])
    assert cli.main() == 0
    return capsys.readouterr().out


def test_url_rerun_after_the_video_landed_loads_s0_s1_from_cache(
        tmp_path, tiny_av, monkeypatch, capsys):
    """First run: resolve returns before the video lands (`video_path` None).
    Second run: both streams are on disk, so resolve sets `video_path`. The
    source and every parameter are the same, so nothing may recompute (§5.2g)."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "work" / "maclips.db")
    monkeypatch.setattr(cli, "check_environment", lambda: None)
    s0 = dataclasses.replace(STAGES_BY_ID["S0"],
                             run=lambda ctx: {"clip_class": "general-own"})
    monkeypatch.setattr(cli, "STAGES", (s0, STAGES_BY_ID["S1"]))

    downloading = SourceRecord(path=tiny_av, origin=URL, audio_path=tiny_av, video_id="vid")
    landed = dataclasses.replace(downloading, video_path=tmp_path / "vid.video.mp4")

    first = _run(monkeypatch, capsys, downloading)
    assert "S0  ingest                 ran" in first
    second = _run(monkeypatch, capsys, landed, "--from", "S5")
    assert "S0  ingest                 cached" in second, second
    assert "S1  brief                  cached" in second, second


def test_a_changed_source_record_still_recomputes(tmp_path, tiny_av, monkeypatch, capsys):
    """Only `video_path` left the key; the rest of the record still keys S0."""
    monkeypatch.setattr(config, "WORK_DIR", tmp_path / "work")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "work" / "maclips.db")
    monkeypatch.setattr(cli, "check_environment", lambda: None)
    s0 = dataclasses.replace(STAGES_BY_ID["S0"],
                             run=lambda ctx: {"clip_class": "general-own"})
    monkeypatch.setattr(cli, "STAGES", (s0,))
    record = SourceRecord(path=tiny_av, origin=URL, audio_path=tiny_av, video_id="vid")

    _run(monkeypatch, capsys, record)
    changed = _run(monkeypatch, capsys, dataclasses.replace(record, title="renamed"))
    assert "S0  ingest                 ran" in changed, changed
