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

# Per-million-token rates (input, output).
#
# Sonnet 5 is $2/$10: the introductory price became the standard price and the
# scheduled rise to $3/$15 will not occur (Anthropic pricing page, fetched
# 2026-09-23; §5.2).
TOKEN_RATES_USD_PER_MTOK = {
    "anthropic/claude-sonnet-5": (2.00, 10.00),
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
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
# ffmpeg binaries
# --------------------------------------------------------------------------- #
# Homebrew's default `ffmpeg` formula is built without libass, freetype or
# fontconfig, so `ass`, `subtitles` and `drawtext` are all absent from it —
# which kills burned captions and the hook/commentary overlays. `ffmpeg-full`
# has them but is keg-only, so it is never on PATH.
#
# We therefore address it by absolute path rather than touching PATH. Every
# ffmpeg and ffprobe call in this project resolves through these two settings,
# and the startup gate probes these same binaries, so the gate can never pass
# against a different build than the pipeline runs.
FFMPEG_FULL_PREFIX = Path("/opt/homebrew/opt/ffmpeg-full/bin")

FFMPEG = Path(
    os.getenv("MACLIPS_FFMPEG", FFMPEG_FULL_PREFIX / "ffmpeg")
).expanduser()
FFPROBE = Path(
    os.getenv("MACLIPS_FFPROBE", FFMPEG_FULL_PREFIX / "ffprobe")
).expanduser()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
WORK_DIR = Path(os.getenv("MACLIPS_WORK_DIR", PROJECT_ROOT / "work")).expanduser()
DB_PATH = Path(os.getenv("MACLIPS_DB_PATH", WORK_DIR / "maclips.db")).expanduser()

# S1 capture (src/maclips/capture.py). BRIEFS_RAW_DIR is *not* under WORK_DIR:
# it holds other people's campaign documents and is gitignored at the repo
# root (briefs/raw/), never inside work/ which has its own ignore rules.
# BROWSER_PROFILE_DIR is a persistent Chromium profile so a login (Content
# Rewards / Whop) survives across `capture` invocations instead of asking for
# it every run.
BRIEFS_RAW_DIR = Path(
    os.getenv("MACLIPS_BRIEFS_RAW_DIR", PROJECT_ROOT / "briefs" / "raw")
).expanduser()
BROWSER_PROFILE_DIR = Path(
    os.getenv("MACLIPS_BROWSER_PROFILE_DIR", WORK_DIR / "browser_profile")
).expanduser()


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
    """Dollar cost of one call. Thinking tokens are billed as output."""
    rate_in, rate_out = TOKEN_RATES_USD_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000

# Installed macOS system font used by libass.
CAPTION_FONT = os.getenv("MACLIPS_CAPTION_FONT", "Helvetica")

# Share-prompt end card (PLAN.md §6.6). Off by default per clip; the text is
# this constant unless the reviewer edits it. Never model-generated.
END_CARD_TEXT = "Send this to someone who needs to hear it"
END_CARD_SECONDS = 3.0
END_CARD_MAX_CHARS = 60
