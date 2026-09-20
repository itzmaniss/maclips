"""S1 brief intake: schema, extraction, and the two hard gates."""
from __future__ import annotations

import json

import pytest

from maclips import brief as b
from maclips.brief import BriefInvalid, BriefRejected, CampaignConfig


def test_required_fields_are_the_three_from_the_schema():
    assert b.REQUIRED_FIELDS == ("brand", "campaign_name", "platforms_allowed")
    assert CampaignConfig().validate() == list(b.REQUIRED_FIELDS)


def test_a_complete_config_validates():
    cfg = CampaignConfig(brand="Acme", campaign_name="Spring", platforms_allowed=["tiktok"])
    assert cfg.validate() == []


def test_inverted_duration_range_is_caught():
    cfg = CampaignConfig(brand="A", campaign_name="B", platforms_allowed=["tiktok"],
                         min_duration_s=90, max_duration_s=30)
    assert "min_duration_s > max_duration_s" in cfg.validate()


@pytest.mark.parametrize("category", ["crypto", "Cryptocurrency", "gambling", "casino", "betting"])
def test_rejected_categories_are_gated(category):
    with pytest.raises(BriefRejected, match="rejected"):
        CampaignConfig(category=category).check_category()


def test_rejection_also_catches_the_category_hiding_in_the_brand():
    """A brief that never names its sector still has to be caught."""
    with pytest.raises(BriefRejected):
        CampaignConfig(brand="LuckySpin Casino", campaign_name="x").check_category()


def test_ordinary_categories_pass():
    CampaignConfig(brand="Acme", campaign_name="Spring", category="saas").check_category()


def test_extraction_rejects_before_a_human_ever_sees_it(monkeypatch):
    """The category gate runs inside extract(), not at confirm time."""
    payload = json.dumps({
        "brand": "CoinThing", "campaign_name": "Launch", "category": "crypto",
        "platforms_allowed": ["tiktok"], "missing": [],
    })
    monkeypatch.setattr(b, "complete", lambda *a, **k: payload, raising=False)
    monkeypatch.setattr("maclips.llm.complete", lambda *a, **k: payload)
    with pytest.raises(BriefRejected):
        b.extract("Promote our new coin")


def test_extraction_parses_a_fenced_response(monkeypatch):
    payload = "```json\n" + json.dumps({
        "brand": "Acme", "campaign_name": "Spring", "category": "saas",
        "platforms_allowed": ["tiktok", "instagram"],
        "min_duration_s": 30, "max_duration_s": 60,
        "missing": ["deadline", "rate_per_1k"],
    }) + "\n```"
    monkeypatch.setattr("maclips.llm.complete", lambda *a, **k: payload)
    cfg = b.extract("some brief text")
    assert cfg.brand == "Acme"
    assert cfg.platforms_allowed == ["tiktok", "instagram"]
    assert cfg.missing == ["deadline", "rate_per_1k"]
    assert cfg.raw_brief == "some brief text"
    assert not cfg.confirmed, "extraction must never self-confirm"


def test_extraction_rejects_non_json(monkeypatch):
    monkeypatch.setattr("maclips.llm.complete", lambda *a, **k: "I could not parse that brief.")
    with pytest.raises(BriefInvalid, match="valid JSON"):
        b.extract("x")


def test_extraction_rejects_a_json_array(monkeypatch):
    monkeypatch.setattr("maclips.llm.complete", lambda *a, **k: "[1, 2, 3]")
    with pytest.raises(BriefInvalid, match="expected an object"):
        b.extract("x")


def test_empty_brief_is_refused_without_calling_the_model(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("must not call the model on an empty brief")

    monkeypatch.setattr("maclips.llm.complete", explode)
    with pytest.raises(BriefInvalid, match="empty"):
        b.extract("   ")


def test_unknown_fields_from_the_model_are_ignored(monkeypatch):
    payload = json.dumps({
        "brand": "Acme", "campaign_name": "S", "platforms_allowed": ["tiktok"],
        "missing": [], "invented_field": "should be dropped",
    })
    monkeypatch.setattr("maclips.llm.complete", lambda *a, **k: payload)
    cfg = b.extract("x")
    assert not hasattr(cfg, "invented_field")


def test_round_trip_through_disk(tmp_path, monkeypatch):
    cfg = CampaignConfig(brand="Acme", campaign_name="S", platforms_allowed=["tiktok"],
                         forbidden_topics=["politics"], confirmed=True)
    path = b.save(cfg, tmp_path / "c.json")
    back = b.load(path)
    assert back.brand == "Acme"
    assert back.forbidden_topics == ["politics"]
    assert back.confirmed


def test_prompt_tells_the_model_to_admit_gaps():
    """`missing` is the field that makes human review possible."""
    assert "missing" in b.EXTRACTION_PROMPT
    assert "Do not infer" in b.EXTRACTION_PROMPT
    assert "SECONDS" in b.EXTRACTION_PROMPT
