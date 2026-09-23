"""Mechanical export rules; semantic brief judgement remains explicitly human."""
from __future__ import annotations

import math
import re
import unicodedata

from .ranking import DEFAULT_MAX_DURATION_S, DEFAULT_MIN_DURATION_S


class ComplianceError(RuntimeError):
    """A specific class or brief rule blocks this clip or post."""


def _tokens(text: object) -> list[str]:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.findall(r"[@#]?[\w]+(?:[.'’][\w]+)*", value, flags=re.UNICODE)


def _contains(text: object, phrase: object) -> bool:
    haystack, needle = _tokens(text), _tokens(phrase)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                               for i in range(len(haystack) - len(needle) + 1))


def _duration(value: object, name: str) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ComplianceError(f"invalid {name}") from exc
    if not math.isfinite(number) or number < 0:
        raise ComplianceError(f"invalid {name}")
    return number


def validate_clip(clip: dict, brief: dict, clip_class: str) -> None:
    """Raise on the first mechanical violation; never certify semantic content."""
    if clip_class not in {"campaign", "general-own"}:
        raise ComplianceError("general-3p is deferred; unsupported clip class")
    campaign = clip_class == "campaign"
    if campaign and (not brief or brief.get("confirmed") is not True):
        raise ComplianceError("campaign requires a human-confirmed brief")
    account_class = clip.get("account_class")
    permitted = {"campaign"} if campaign else {"campaign", "general"}
    if account_class not in permitted:
        raise ComplianceError(f"account class {account_class!r} is not allowed for {clip_class}; YPP routing is forbidden")
    start, end = _duration(clip.get("start"), "clip start"), _duration(clip.get("end"), "clip end")
    if end <= start:
        raise ComplianceError("duration: clip end must follow start")
    rules = brief if campaign else {}
    low = rules.get("min_duration_s")
    high = rules.get("max_duration_s")
    low = _duration(DEFAULT_MIN_DURATION_S if low is None else low, "minimum duration")
    high = _duration(DEFAULT_MAX_DURATION_S if high is None else high, "maximum duration")
    if low > high or not low <= end - start <= high:
        raise ComplianceError(f"duration {end-start:.3f}s outside {low:g}–{high:g}s")
    platform = str(clip.get("platform", "")).strip().casefold()
    if not platform:
        raise ComplianceError("platform is required")
    if campaign and platform not in {str(p).strip().casefold() for p in rules.get("platforms_allowed", [])}:
        raise ComplianceError(f"platform {platform!r} is disallowed by brief")
    if campaign:
        if clip.get("cta") or clip.get("cta_text"):
            raise ComplianceError("CTA is forbidden on campaign clips")
        if clip.get("commentary") or clip.get("commentary_text") or clip.get("commentary_mode", "none") not in (None, "", "none"):
            raise ComplianceError("commentary is forbidden on campaign clips")
    caption = clip.get("caption", "")
    for field in ("required_tags", "required_hashtags"):
        for required in rules.get(field, []) or []:
            if not _contains(caption, required):
                raise ComplianceError(f"missing {field}: {required}")
    if campaign:
        disclosure = rules.get("disclosure_text")
        if not _tokens(disclosure):
            raise ComplianceError("human must specify campaign disclosure_text before export")
        if not _contains(caption, disclosure):
            raise ComplianceError("missing required disclosure in caption")
    combined = f"{caption}\n{clip.get('transcript_text', '')}"
    for phrase in rules.get("required_phrases", []) or []:
        if not _contains(combined, phrase):
            raise ComplianceError(f"missing required phrase: {phrase}")
    if rules.get("overlays_allowed") is False and str(clip.get("hook_text", "")).strip():
        raise ComplianceError("hook overlay is forbidden by brief")


def human_checklist(brief: dict) -> list[str]:
    """Items needing human judgement, deliberately not mechanically marked passed."""
    items = [f"Human review: no forbidden topic — {topic}" for topic in brief.get("forbidden_topics", []) or []]
    items += [f"Human review: obey edit restriction — {edit}" for edit in brief.get("forbidden_edits", []) or []]
    items.append("Human review: compare against original brief for platform-specific tags, alternative disclosure hashtags, required sounds, source allowlists, account, audience, payout and retention requirements; the schema cannot express these.")
    return items


def validate_post(clip_class: str, row: dict) -> None:
    if clip_class not in {"campaign", "general-own"}:
        raise ComplianceError("general-3p is deferred; unsupported clip class")
    if (clip_class == "campaign" and row.get("status") in {"submitted", "approved"}
            and row.get("disclosure_ticked") is not True):
        raise ComplianceError("campaign post cannot be submitted without paid-promotion checkbox attestation")
