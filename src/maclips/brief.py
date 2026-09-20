"""S1: campaign brief intake.

The schema is PLAN.md §7.2. Haiku extracts fields from free text; **you confirm
them**. Extraction is never trusted on its own — §5.5's rule is that the LLM is
never the only check on anything mechanically checkable, and every field here
feeds a mechanical check later (duration filters in S5, compliance in S11).

Two hard gates:
  * category crypto/gambling -> rejected at extraction, before confirmation;
  * an unconfirmed brief blocks S5.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config

REJECTED_CATEGORIES = ("crypto", "cryptocurrency", "gambling", "betting", "casino")

# Fields the schema requires. §7.2 marks everything else optional because real
# briefs vary and a missing optional field is information, not an error.
REQUIRED_FIELDS = ("brand", "campaign_name", "platforms_allowed")


class BriefRejected(RuntimeError):
    """The brief is for a category the tool does not serve (§7.2)."""


class BriefInvalid(RuntimeError):
    """Extraction produced something that is not a usable config."""


@dataclass
class CampaignConfig:
    """PLAN.md §7.2. Every field optional except the three marked required."""

    brand: str = ""
    campaign_name: str = ""
    category: str = ""

    rate_per_1k: float | None = None
    pool_total: float | None = None
    pool_used_pct_at_join: float | None = None
    deadline: str = ""

    platforms_allowed: list[str] = field(default_factory=list)

    min_duration_s: float | None = None
    max_duration_s: float | None = None

    required_tags: list[str] = field(default_factory=list)
    required_hashtags: list[str] = field(default_factory=list)
    required_phrases: list[str] = field(default_factory=list)

    disclosure_text: str = ""
    forbidden_topics: list[str] = field(default_factory=list)
    forbidden_edits: list[str] = field(default_factory=list)

    overlays_allowed: bool | None = None
    commentary_allowed: bool = False
    submission_window_min: int | None = None

    raw_brief: str = ""
    confirmed: bool = False

    # Fields Haiku could not find, so the confirm step can highlight them
    # rather than letting a silent default pass as an extracted value.
    missing: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Return the required fields that are still empty."""
        problems = []
        for name in REQUIRED_FIELDS:
            value = getattr(self, name)
            if not value:
                problems.append(name)
        if (self.min_duration_s is not None and self.max_duration_s is not None
                and self.min_duration_s > self.max_duration_s):
            problems.append("min_duration_s > max_duration_s")
        return problems

    def check_category(self) -> None:
        """Hard gate: crypto and gambling are refused (§7.2, §1.3)."""
        haystack = f"{self.category} {self.brand} {self.campaign_name}".lower()
        for banned in REJECTED_CATEGORIES:
            if banned in haystack:
                raise BriefRejected(
                    f"brief category {self.category or banned!r} is rejected: "
                    "crypto and gambling briefs are out of scope (PLAN.md §7.2)."
                )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "brand": {"type": "string"},
        "campaign_name": {"type": "string"},
        "category": {"type": "string"},
        "rate_per_1k": {"type": ["number", "null"]},
        "pool_total": {"type": ["number", "null"]},
        "deadline": {"type": "string"},
        "platforms_allowed": {"type": "array", "items": {"type": "string"}},
        "min_duration_s": {"type": ["number", "null"]},
        "max_duration_s": {"type": ["number", "null"]},
        "required_tags": {"type": "array", "items": {"type": "string"}},
        "required_hashtags": {"type": "array", "items": {"type": "string"}},
        "required_phrases": {"type": "array", "items": {"type": "string"}},
        "disclosure_text": {"type": "string"},
        "forbidden_topics": {"type": "array", "items": {"type": "string"}},
        "forbidden_edits": {"type": "array", "items": {"type": "string"}},
        "overlays_allowed": {"type": ["boolean", "null"]},
        "commentary_allowed": {"type": ["boolean", "null"]},
        "submission_window_min": {"type": ["integer", "null"]},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["brand", "campaign_name", "platforms_allowed", "missing"],
}

EXTRACTION_PROMPT = """You are extracting a structured campaign config from a \
clipping brief. Return JSON only, matching the schema exactly.

Rules:
- Extract only what the brief actually states. Do not infer, complete or \
normalise values the brief does not give.
- Durations are in SECONDS. Convert if the brief uses minutes.
- `platforms_allowed`: lowercase platform names, e.g. ["tiktok", "instagram"].
- `required_tags` are accounts to @-mention. `required_hashtags` include the #.
- `missing`: list every schema field the brief does not specify. This is the \
most important field — a wrong guess is worse than an admitted gap, because a \
human reviews `missing` and cannot review a confident invention.
- `category`: the brand's sector in one or two words.

Brief:
---
{brief}
---"""


def extract(raw_brief: str, model: str | None = None) -> CampaignConfig:
    """Haiku extracts the fields; the caller confirms them.

    Raises BriefRejected before anything else if the category is out of scope.
    """
    from .llm import complete

    if not raw_brief.strip():
        raise BriefInvalid("the brief is empty")

    model = model or config.BRIEF_MODEL
    raw = complete(
        EXTRACTION_PROMPT.format(brief=raw_brief),
        model=model,
        json_only=True,
        max_tokens=2048,
    )
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as exc:
        raise BriefInvalid(f"extraction did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BriefInvalid(f"extraction returned {type(data).__name__}, expected an object")

    data["raw_brief"] = raw_brief
    cfg = CampaignConfig.from_dict(data)
    cfg.check_category()   # hard gate, before any human sees it
    return cfg


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1]
        t = t.rsplit("```", 1)[0]
    return t.strip()


def save(cfg: CampaignConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.as_dict(), indent=2))
    return path


def load(path: Path) -> CampaignConfig:
    return CampaignConfig.from_dict(json.loads(path.read_text()))
