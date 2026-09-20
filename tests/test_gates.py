"""Startup gates. Each one must stop the process, not warn."""
from __future__ import annotations

import pytest

from maclips import gates
from maclips.gates import EnvironmentGate, GateReport, check_ffmpeg, check_platform, check_python


def test_platform_gate_rejects_non_darwin(monkeypatch):
    monkeypatch.setattr(gates.sys, "platform", "linux")
    with pytest.raises(EnvironmentGate, match="macOS-only"):
        check_platform(GateReport())


def test_platform_gate_rejects_intel(monkeypatch):
    monkeypatch.setattr(gates.sys, "platform", "darwin")
    monkeypatch.setattr(gates.platform, "machine", lambda: "x86_64")
    with pytest.raises(EnvironmentGate, match="Apple Silicon"):
        check_platform(GateReport())


def test_platform_gate_passes_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(gates.sys, "platform", "darwin")
    monkeypatch.setattr(gates.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(gates.platform, "mac_ver", lambda: ("26.5", ("", "", ""), "arm64"))
    report = GateReport()
    check_platform(report)
    assert any("darwin/arm64" in line for line in report.passed)


def test_python_gate_rejects_other_versions(monkeypatch):
    monkeypatch.setattr(gates.sys, "version_info", (3, 13, 1))
    with pytest.raises(EnvironmentGate, match="3.12 required"):
        check_python(GateReport())


def test_ffmpeg_gate_names_every_missing_filter(monkeypatch):
    """The failure must say which filters are absent, not just that one is."""
    monkeypatch.setattr(gates, "_ffmpeg_list", lambda kind: " crop  scale  loudnorm  scdet ")
    with pytest.raises(EnvironmentGate) as excinfo:
        check_ffmpeg(GateReport())
    message = str(excinfo.value)
    for missing in ("ass", "subtitles", "drawtext"):
        assert missing in message
    assert "ffmpeg-full" in message, "the message must say how to fix it"


def test_ffmpeg_gate_catches_missing_encoder(monkeypatch):
    filters = " ".join(f" {f} " for f in gates.REQUIRED_FFMPEG_FILTERS)

    def fake(kind):
        return filters if kind == "filters" else "libx264"

    monkeypatch.setattr(gates, "_ffmpeg_list", fake)
    with pytest.raises(EnvironmentGate, match="h264_videotoolbox"):
        check_ffmpeg(GateReport())


def test_ffmpeg_gate_passes_when_everything_present(monkeypatch):
    filters = " ".join(f" {f} " for f in gates.REQUIRED_FFMPEG_FILTERS)
    encoders = " ".join(gates.REQUIRED_FFMPEG_ENCODERS)

    monkeypatch.setattr(gates, "_ffmpeg_list", lambda kind: filters if kind == "filters" else encoders)
    monkeypatch.setattr(gates.shutil, "which", lambda name: f"/usr/bin/{name}")
    report = GateReport()
    check_ffmpeg(report)
    assert report.passed


def test_this_machine_currently_fails_the_ffmpeg_gate():
    """Documents a real, outstanding defect: Homebrew's ffmpeg has no libass.

    Delete this test once `ffmpeg-full` is installed — at which point
    `test_ffmpeg_gate_passes_when_everything_present` is the live check.
    """
    with pytest.raises(EnvironmentGate, match="missing required filter"):
        check_ffmpeg(GateReport())
