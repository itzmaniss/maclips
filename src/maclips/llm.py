"""LiteLLM call seam.

One function. Every model call in the pipeline goes through it, which is what
makes the local-MLX slot in config.py a config change rather than a code
change. `complete()` matches the `llm_fn` signature that highlights.py's
`get_highlights()` expects, so the fork's seam plugs straight in.
"""
from __future__ import annotations

from . import config

RANK_MAX_TOKENS = 32000
"""Caps thinking and JSON together (§5.2). 16,000 was spent entirely on
default-effort thinking (§5.2c); 32,000 leaves low-effort thinking room beside
~2k tokens of JSON."""


def complete(
    prompt: str,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float | None = 0.2,
    json_only: bool = False,
    usage_sink: list | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Send one prompt, return the text. Raises on a truncated or empty completion.

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
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        # LiteLLM 1.102.0 maps "low" on Sonnet 5 to adaptive thinking plus
        # output_config.effort (§5.2c). "none" sends nothing at all, which
        # leaves Sonnet 5 thinking at its default high effort.
        kwargs["reasoning_effort"] = reasoning_effort
    if json_only:
        # LiteLLM drops this for Anthropic when no schema is attached (§5.3):
        # JSON-only is enforced by the prompt and the caller's parser.
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

    choice = response.choices[0]
    finish_reason = choice.finish_reason
    if usage_sink is not None:
        # Before any raise: a failed call is still a billed call.
        usage_sink.append(_usage_record(model, response, finish_reason))

    text = choice.message.content or ""
    if finish_reason == "length" and (json_only or not text.strip()):
        # Ahead of the empty-text check: a cap hit while thinking leaves no
        # text, and "empty completion" would hide the real cause (§5.2c).
        # Truncated JSON is never valid either.
        what = "the JSON output is truncated" if json_only else "no text was produced"
        raise RuntimeError(f"{model} stopped at max_tokens={max_tokens}: {what}.")
    if not text.strip():
        raise RuntimeError(f"{model} returned an empty completion.")
    return text


def _usage_record(model: str, response, finish_reason: str | None) -> dict:
    """Usage of one call. `input_tokens` already includes any cache tokens."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "finish_reason": finish_reason,
        "cost_usd": config.estimated_cost_usd(model, input_tokens, output_tokens),
    }


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
    """The ranking tier (Sonnet), adaptive thinking at low effort. PLAN.md §5.2.

    Temperature is omitted: Sonnet 5 rejects any non-default sampling value.
    """
    return complete(prompt, model=config.RANKING_MODEL, json_only=True,
                    max_tokens=RANK_MAX_TOKENS, temperature=None,
                    reasoning_effort="low", usage_sink=usage_sink)


def brief_fn(prompt: str) -> str:
    """Brief field extraction (Haiku), confirmed by a human at S1."""
    return complete(prompt, model=config.BRIEF_MODEL, json_only=True)


def commentary_fn(prompt: str) -> str:
    """Commentary drafts (Haiku), accepted by a human at S10."""
    return complete(prompt, model=config.COMMENTARY_MODEL, max_tokens=1024)
