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
    usage_sink: list | None = None,
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

    try:
        response = litellm.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001 - re-raised with a usable hint
        raise _with_model_hint(exc, model) from exc

    text = response.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError(f"{model} returned an empty completion.")

    if usage_sink is not None:
        usage = getattr(response, "usage", None)
        usage_sink.append({
            "model": model,
            "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cost_usd_range": config.cost_range_usd(
                model,
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            ),
        })
    return text


def _with_model_hint(exc: Exception, model: str) -> Exception:
    """A bad model id is the most likely first failure; say so precisely."""
    text = str(exc).lower()
    if "not_found" in text or "404" in text or "does not exist" in text:
        base = model.split("/")[-1]
        undated = base.rsplit("-20", 1)[0] if "-20" in base else base
        return RuntimeError(
            f"model {model!r} was rejected as unknown.\n"
            f"  Current Anthropic model ids carry no date suffix — try "
            f"'anthropic/{undated}'.\n"
            f"  Original error: {exc}"
        )
    return exc


def rank_fn(prompt: str, usage_sink: list | None = None) -> str:
    """The ranking tier (Sonnet). PLAN.md §5.2."""
    return complete(prompt, model=config.RANKING_MODEL, json_only=True,
                    max_tokens=16000, usage_sink=usage_sink)


def brief_fn(prompt: str) -> str:
    """Brief field extraction (Haiku), confirmed by a human at S1."""
    return complete(prompt, model=config.BRIEF_MODEL, json_only=True)


def commentary_fn(prompt: str) -> str:
    """Commentary drafts (Haiku), accepted by a human at S10."""
    return complete(prompt, model=config.COMMENTARY_MODEL, max_tokens=1024)
