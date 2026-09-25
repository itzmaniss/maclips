"""Run-level length preset: only the S5 prompt's numbers change, and a brief wins."""
import argparse

import pytest

from maclips import cli, llm, ranking
from maclips.orchestrator import GateFailure, RunContext
from maclips.stages import STAGES_BY_ID, s5_rank


def args(preset="default", low=None, high=None):
    return argparse.Namespace(length_preset=preset, clip_min_duration=low, clip_max_duration=high)


def test_preset_resolution_and_precedence():
    assert cli.clip_duration_config(args(), {}) == (None, None)          # default: S5 key unchanged
    assert cli.clip_duration_config(args("shorts-dense"), {}) == (22.0, 45.0)
    assert cli.clip_duration_config(args("shorts-dense", high=60), {}) == (22.0, 60)
    # A brief stating either duration switches the preset off entirely.
    assert cli.clip_duration_config(args("shorts-dense"), {"max_duration_s": 90}) == (None, None)
    with pytest.raises(ValueError, match="unknown length preset"):
        cli.clip_duration_config(args("tiktok"), {})
    assert "clip_min_duration" in STAGES_BY_ID["S5"].params and "length_preset" not in STAGES_BY_ID["S5"].params


def prompt_for(tmp_path, monkeypatch, config, brief=None):
    seen = []
    def fake(prompt, usage_sink=None):
        seen.append(prompt)
        raise ranking.RankingError("stop after capturing the prompt")
    monkeypatch.setattr(llm, "rank_fn", fake)
    words = [{"word": f"w{i}.", "start": i, "end": i + .5, "score": .9} for i in range(20)]
    ctx = RunContext("r", tmp_path / "s", "h", tmp_path, config, outputs={
        "S0": {"clip_class": "general-own"}, "S1": {"brief": brief or {}}, "S3": {"words": words}})
    with pytest.raises(GateFailure):
        s5_rank(ctx)
    return seen[0]


def test_preset_reaches_prompt_numbers_and_wording_is_byte_identical(tmp_path, monkeypatch):
    low, high = cli.clip_duration_config(args("shorts-dense"), {})
    dense = prompt_for(tmp_path, monkeypatch, {"clip_min_duration": low, "clip_max_duration": high})
    default = prompt_for(tmp_path, monkeypatch, {})
    assert "55-112 words (~22-45s at" in dense
    assert "25-450 words (~10-180s at" in default
    assert dense.replace("~22-45s", "~10-180s").replace("55-112 words", "25-450 words") == default


def test_brief_durations_override_the_preset_in_the_prompt(tmp_path, monkeypatch):
    brief = {"min_duration_s": 30, "max_duration_s": 60}
    low, high = cli.clip_duration_config(args("shorts-dense"), brief)
    prompt = prompt_for(tmp_path, monkeypatch, {"clip_min_duration": low, "clip_max_duration": high}, brief)
    assert "75-150 words (~30-60s at" in prompt
