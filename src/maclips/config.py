"""Configuration: model slots, paths, and secrets.

Keys come from `.env`, which is gitignored. Nothing here holds a literal key,
and no key is ever written to a checkpoint or a log line.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------- #
# Model slots
# --------------------------------------------------------------------------- #
# PLAN.md §5.2 assigns the tiers deliberately: ranking is the quality-critical
# judgement and gets Sonnet, which breaks the usual "ranking -> Haiku" rule on
# purpose. Brief extraction and commentary drafts are Haiku because a human
# confirms both before anything renders.
#
# These are LiteLLM model strings: "<provider>/<model-id>". The Anthropic ids
# carry no date suffix.
RANKING_MODEL = os.getenv("MACLIPS_RANKING_MODEL", "anthropic/claude-sonnet-5")
BRIEF_MODEL = os.getenv("MACLIPS_BRIEF_MODEL", "anthropic/claude-haiku-4-5")
COMMENTARY_MODEL = os.getenv("MACLIPS_COMMENTARY_MODEL", "anthropic/claude-haiku-4-5")

# Per-million-token rates, for the cost line the UI shows per source.
# Verified against Anthropic's model table (2026-06-24 snapshot). PLAN.md §5.2
# quotes $3/$15 for Sonnet, which is Sonnet 4.6's rate, not Sonnet 5's.
TOKEN_RATES_USD_PER_MTOK = {
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
}

# --------------------------------------------------------------------------- #
# Optional local MLX LLM (PLAN.md "MLX: where and where not")
# --------------------------------------------------------------------------- #
# A config slot only. mlx-lm serves an OpenAI-compatible endpoint, so LiteLLM
# reaches it as an "openai/..." model with a custom api_base. No model is
# installed and nothing routes here until it is switched on deliberately.
LOCAL_LLM_ENABLED = os.getenv("MACLIPS_LOCAL_LLM_ENABLED", "false").lower() == "true"
LOCAL_LLM_MODEL = os.getenv("MACLIPS_LOCAL_LLM_MODEL", "")
LOCAL_LLM_API_BASE = os.getenv("MACLIPS_LOCAL_LLM_API_BASE", "http://127.0.0.1:8080/v1")

# --------------------------------------------------------------------------- #
# Diarization
# --------------------------------------------------------------------------- #
DIARIZATION_MODEL = os.getenv(
    "MACLIPS_DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
)
HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"  # community-1 is a gated download

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
WORK_DIR = Path(os.getenv("MACLIPS_WORK_DIR", PROJECT_ROOT / "work")).expanduser()
DB_PATH = Path(os.getenv("MACLIPS_DB_PATH", WORK_DIR / "maclips.db")).expanduser()


@dataclass(frozen=True)
class Secrets:
    """Resolved at call time, never at import time, never logged."""

    anthropic_api_key: str = ""
    huggingface_token: str = ""

    @classmethod
    def load(cls) -> Secrets:
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            huggingface_token=os.getenv(HF_TOKEN_ENV, "").strip(),
        )

    def require_anthropic(self) -> str:
        if not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to .env (gitignored). "
                "Ranking, brief extraction and commentary all need it."
            )
        return self.anthropic_api_key

    def require_huggingface(self) -> str:
        if not self.huggingface_token:
            raise RuntimeError(
                f"{HF_TOKEN_ENV} is not set. {DIARIZATION_MODEL} is a gated "
                "download; accept the terms on Hugging Face, then put the token "
                "in .env."
            )
        return self.huggingface_token


def estimated_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of one call, for the per-source cost line."""
    rate_in, rate_out = TOKEN_RATES_USD_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000
