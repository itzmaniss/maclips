"""One fixture clip per compliance rule; no real approvals or attestations."""
import pytest

from maclips.compliance import ComplianceError, human_checklist, validate_clip, validate_post


@pytest.fixture
def brief():
    return {"confirmed": True, "platforms_allowed": ["instagram"],
            "required_tags": ["@brand"], "required_hashtags": ["#campaign"],
            "required_phrases": ["great product"], "disclosure_text": "Paid partnership",
            "min_duration_s": 10, "max_duration_s": 180}


@pytest.fixture
def clip():
    return {"start": 100, "end": 130, "account_class": "campaign", "platform": "instagram",
            "caption": "@Brand #Campaign Paid partnership", "transcript_text": "A great product.",
            "hook_text": "Watch this", "commentary_mode": "none"}


def test_valid_campaign(brief, clip):
    validate_clip(clip, brief, "campaign")


@pytest.mark.parametrize("field,value,reason", [
    ("end", 105, "duration"), ("end", 281, "duration"),
    ("caption", "@brandish #campaign Paid partnership", "required_tags"),
    ("caption", "@brand #campaignextra Paid partnership", "required_hashtags"),
    ("caption", "@brand #campaign", "disclosure"),
    ("platform", "tiktok", "platform"), ("cta", "Subscribe", "CTA"),
    ("commentary", "My view", "commentary"), ("commentary_mode", "auto", "commentary"),
    ("account_class", "general", "account class"), ("account_class", "ypp", "account class"),
    ("transcript_text", "unrelated", "required phrase"),
])
def test_one_clip_per_rule(brief, clip, field, value, reason):
    clip[field] = value
    with pytest.raises(ComplianceError, match=reason):
        validate_clip(clip, brief, "campaign")


def test_overlay_prohibition(brief, clip):
    brief["overlays_allowed"] = False
    with pytest.raises(ComplianceError, match="hook overlay"):
        validate_clip(clip, brief, "campaign")


def test_campaign_class_wins_over_brief_commentary_permission(brief, clip):
    brief["commentary_allowed"] = True
    clip["commentary"] = "Opinion"
    with pytest.raises(ComplianceError, match="commentary"):
        validate_clip(clip, brief, "campaign")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_disclosure_requires_human(brief, clip, value):
    brief["disclosure_text"] = value
    with pytest.raises(ComplianceError, match="human must specify"):
        validate_clip(clip, brief, "campaign")


def test_unconfirmed_brief(brief, clip):
    brief["confirmed"] = False
    with pytest.raises(ComplianceError, match="human-confirmed"):
        validate_clip(clip, brief, "campaign")


def test_deferred_class(clip):
    with pytest.raises(ComplianceError, match="deferred"):
        validate_clip(clip, {}, "general-3p")


@pytest.mark.parametrize("account", ["campaign", "general"])
def test_own_routing_and_default_duration(clip, account):
    clip["account_class"] = account
    validate_clip(clip, {}, "general-own")
    clip["end"] = clip["start"] + 181
    with pytest.raises(ComplianceError, match="duration"):
        validate_clip(clip, {}, "general-own")


@pytest.mark.parametrize("status", ["submitted", "approved"])
@pytest.mark.parametrize("ticked", [False, None, "true", 1])
def test_post_gate_requires_explicit_boolean(status, ticked):
    with pytest.raises(ComplianceError, match="checkbox"):
        validate_post("campaign", {"status": status, "disclosure_ticked": ticked})


def test_post_fixture_attestation():
    validate_post("campaign", {"status": "submitted", "disclosure_ticked": True})
    validate_post("campaign", {"status": "draft", "disclosure_ticked": False})
    validate_post("general-own", {"status": "submitted", "disclosure_ticked": False})


def test_semantic_rules_are_human_checklist(brief):
    brief.update(forbidden_topics=["politics"], forbidden_edits=["no music"])
    checklist = human_checklist(brief)
    assert any("politics" in item for item in checklist)
    assert any("no music" in item for item in checklist)
    assert all("Human review" in item for item in checklist)


def test_end_card_blocked_when_brief_forbids_overlays(brief, clip):
    clip.update(hook_text="", end_card=True, end_card_text="Send this to someone")
    validate_clip(clip, brief, "campaign")
    brief["overlays_allowed"] = False
    with pytest.raises(ComplianceError, match="end card overlay"):
        validate_clip(clip, brief, "campaign")


def test_end_card_is_not_classed_as_cta(brief, clip):
    # PLAN.md §6.6 [I]: a share prompt is not a §1.4 CTA, pending the user's call.
    clip.update(end_card=True, end_card_text="Send this to someone who needs to hear it")
    brief["overlays_allowed"] = True
    validate_clip(clip, brief, "campaign")
