"""LiteLLM call seam.

One function. Every model call in the pipeline goes through it, which is what
makes the local-MLX slot in config.py a config change rather than a code
change. `complete()` matches the `llm_fn` signature that highlights.py's
`get_highlights()` expects, so the fork's seam plugs straight in.
"""
from __future__ import annotations

from . import config


def complete(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.2,
    json_only: bool = False,
) -> str:
    """Send one prompt, return the text. Raises on an empty completion.

    Imports litellm lazily: it is a heavy import and the startup gates should
    be able to fail before paying for it.
    """
    import litellm

    model = model or config.RANKING_MODEL
    secrets = config.Secrets.load()

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_only:
        kwargs["response_format"] = {"type": "json_object"}

    if model.startswith("openai/") and config.LOCAL_LLM_ENABLED:
        # The mlx-lm server is OpenAI-compatible and needs no real key.
        kwargs["api_base"] = config.LOCAL_LLM_API_BASE
        kwargs["api_key"] = "not-needed"
    else:
        kwargs["api_key"] = secrets.require_anthropic()

    response = litellm.completion(**kwargs)
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError(f"{model} returned an empty completion.")
    return text


def rank_fn(prompt: str) -> str:
    """The ranking tier (Sonnet). PLAN.md §5.2."""
    return complete(prompt, model=config.RANKING_MODEL, json_only=True)


def brief_fn(prompt: str) -> str:
    """Brief field extraction (Haiku), confirmed by a human at S1."""
    return complete(prompt, model=config.BRIEF_MODEL, json_only=True)


def commentary_fn(prompt: str) -> str:
    """Commentary drafts (Haiku), accepted by a human at S10."""
    return complete(prompt, model=config.COMMENTARY_MODEL, max_tokens=1024)
