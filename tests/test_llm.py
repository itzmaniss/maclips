"""The LiteLLM seam: what ranking sends, and what a failed call leaves behind."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from maclips import config, llm, ranking
from maclips.ranking import RANKING_SCHEMA


def _response(text, finish_reason, prompt=44_014, completion=16_000, reasoning=16_000,
              cache_read=0, cache_creation=0):
    usage = SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_creation,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason,
                                 message=SimpleNamespace(content=text))],
        usage=usage,
    )


@pytest.fixture
def fake_completion(monkeypatch):
    import litellm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def install(resp):
        monkeypatch.setattr(litellm, "completion", lambda **k: resp)
    return install


def test_rank_fn_sends_low_effort_adaptive_thinking_and_no_temperature(monkeypatch):
    """§5.2c: the body the installed LiteLLM actually POSTs, captured offline.

    Without `thinking`, Sonnet 5 thinks at high effort and spent the whole
    16,000-token cap. `reasoning_effort="none"` would send nothing at all.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    sent = {}

    class Captured(Exception):
        pass

    def fake_post(self, url, data=None, json=None, **kwargs):
        sent["body"] = data if data is not None else json
        raise Captured()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(HTTPHandler, "post", fake_post)
    with pytest.raises(Exception):
        llm.rank_fn("rank this")

    body = sent["body"]
    body = json.loads(body) if isinstance(body, (str, bytes)) else body
    assert body["thinking"]["type"] == "adaptive"
    assert body["max_tokens"] == llm.RANK_MAX_TOKENS
    assert "temperature" not in body
    # §5.3: the schema goes in output_config.format beside the effort, not in
    # the deprecated top-level output_format LiteLLM's response_format sends.
    assert body["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": RANKING_SCHEMA},
    }
    assert "output_format" not in body and "response_format" not in body


def test_ranking_schema_is_valid_for_structured_outputs():
    """Every object closed; no keyword structured outputs rejects (§5.3)."""
    unsupported = {"minimum", "maximum", "multipleOf", "minLength", "maxLength", "maxItems"}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            assert not unsupported & node.keys()
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(RANKING_SCHEMA)


def test_a_schema_violating_reply_still_gates_client_side(monkeypatch):
    """The API constraint is not trusted alone: a reply that breaks the schema
    stops at the malformed-output gate after the one retry, as before (§5.4)."""
    import litellm

    from test_ranking import words

    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _response('{"candidates": [{"start_word": "seven"}]}', "stop",
                         completion=50, reasoning=0)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(litellm, "completion", fake)
    with pytest.raises(ranking.RankingError, match="no usable word indices"):
        ranking.rank(words(), llm_fn=llm.rank_fn)
    assert len(calls) == 2
    assert all(c["output_config"]["format"]["schema"] is RANKING_SCHEMA for c in calls)


def test_truncation_while_thinking_is_reported_as_truncation(fake_completion):
    """The A1 failure: cap hit during thinking, no text. Not 'empty completion'."""
    fake_completion(_response("", "length"))
    with pytest.raises(RuntimeError, match="max_tokens=32000: the JSON output is truncated"):
        llm.complete("p", model="anthropic/claude-sonnet-5", json_only=True,
                     max_tokens=32000)


def test_truncation_without_json_and_without_text_is_still_truncation(fake_completion):
    fake_completion(_response("", "length"))
    with pytest.raises(RuntimeError, match="no text was produced"):
        llm.complete("p", model="anthropic/claude-sonnet-5", max_tokens=100)


def test_usage_is_recorded_before_a_truncation_raise(fake_completion):
    """A failed call is still billed, so it must leave a record."""
    fake_completion(_response("", "length"))
    sink: list = []
    with pytest.raises(RuntimeError):
        llm.complete("p", model="anthropic/claude-sonnet-5", json_only=True,
                     max_tokens=16000, usage_sink=sink)
    assert sink == [{
        "model": "anthropic/claude-sonnet-5",
        "input_tokens": 44_014, "output_tokens": 16_000, "reasoning_tokens": 16_000,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "finish_reason": "length",
        "cost_usd": pytest.approx(0.248028),
    }]


def test_usage_is_recorded_before_an_empty_completion_raise(fake_completion):
    fake_completion(_response("  ", "stop", completion=10, reasoning=0))
    sink: list = []
    with pytest.raises(RuntimeError, match="empty completion"):
        llm.complete("p", model="anthropic/claude-sonnet-5", usage_sink=sink)
    assert len(sink) == 1 and sink[0]["finish_reason"] == "stop"


def test_usage_records_reasoning_and_cache_tokens_on_success(fake_completion):
    fake_completion(_response('{"candidates": []}', "stop", prompt=50_000,
                              completion=5_000, reasoning=3_000,
                              cache_read=1_000, cache_creation=2_000))
    sink: list = []
    assert llm.complete("p", model="anthropic/claude-sonnet-5", usage_sink=sink)
    record = sink[0]
    assert record["reasoning_tokens"] == 3_000
    assert record["cache_read_input_tokens"] == 1_000
    assert record["cache_creation_input_tokens"] == 2_000
    assert record["finish_reason"] == "stop"


def test_sonnet_5_cost_is_the_single_settled_rate():
    """$2/$10 per MTok (§5.2); thinking tokens are billed as output."""
    assert config.estimated_cost_usd("anthropic/claude-sonnet-5", 1_000_000, 0) == 2.0
    assert config.estimated_cost_usd("anthropic/claude-sonnet-5", 0, 1_000_000) == 10.0
    assert not hasattr(config, "DISPUTED_RATES_USD_PER_MTOK")
