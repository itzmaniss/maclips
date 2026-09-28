"""Prompt v3's hint block (§5.2i): no model, [n] references only, capped at 12."""
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from maclips import config, ranking, signals


def w(text, a, b):
    return {"word": text, "start": a, "end": b, "speaker": None}


WORDS = [w("Hello", 0.0, 0.4), w("there.", 0.5, 0.9),           # [0] nothing
         w("Why", 2.5, 2.8), w("not?", 2.9, 3.2),               # [2] trigger, question, pause 1.6 s
         w("We", 3.3, 3.5), w("lost", 3.6, 3.9), w("everything!", 4.0, 4.6),  # [4] story turn, exclamation
         w("Fine.", 4.7, 5.0)]                                   # [7] loud only


def test_lines_are_referenced_by_the_first_word_index_the_model_sees():
    block = signals.build(WORDS, peaks=[4.8])
    lines = block.splitlines()
    assert "Hints only" in lines[0] and "no speaker signals" in lines[0].lower()
    assert lines[1:] == ["[2] trigger word, question, pause of 1.6s before it",
                         "[4] exclamation, story turn",
                         "[7] one of the loudest moments"]
    transcript = ranking.build_transcript(WORDS)
    for line in lines[1:]:
        assert f"\n{line.split()[0]} " in "\n" + transcript, "every [n] opens a transcript line"
    assert not any(ch.isdigit() and ":" in line for line in lines[1:] for ch in line)  # no timestamps


def test_at_most_twelve_lines_strongest_first_then_in_transcript_order():
    words = []
    for i in range(30):
        text = "Why?" if i % 2 else "ok."
        words.append(w(text, i * 2.0, i * 2.0 + 0.5))  # 1.5 s pause before every line
    lines = signals.build(words).splitlines()[1:]
    assert len(lines) == signals.MAX_LINES
    assert all("question" in line for line in lines), "the questions outrank bare pauses"
    assert [int(l[1:l.index("]")]) for l in lines] == sorted(int(l[1:l.index("]")]) for l in lines)


def test_no_hint_gives_an_empty_block():
    assert signals.build([w("ok.", 0, .4), w("fine.", .5, .9)]) == ""


def test_loud_peaks_are_four_seconds_apart():
    level = {5: -5.0, 6: -6.0, 12: -8.0, 14: -9.0}
    seconds = [(float(t), level.get(t, -30.0 - t / 100)) for t in range(20)]
    assert signals.loud_peaks(seconds, count=3) == [0.0, 5.0, 12.0], "6 and 14 are within 4 s of a louder peak"
    peaks = signals.loud_peaks(seconds)
    assert all(b - a >= signals.PEAK_SPACING_S for a, b in zip(peaks, peaks[1:]))


def test_loud_seconds_measures_each_second_with_the_configured_ffmpeg(tmp_path):
    rate = 16000
    tone = np.concatenate([np.zeros(rate * 2), 0.5 * np.sin(np.arange(rate) * 0.3), np.full(rate, 0.01)])
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(rate)
        f.writeframes((tone * 32767).astype(np.int16).tobytes())
    if not Path(config.FFMPEG).exists():
        pytest.skip("configured ffmpeg is not installed")
    seconds = signals.loud_seconds(path)
    assert [round(t) for t, _ in seconds] == [2, 3], "the silent seconds report -inf and are skipped"
    assert signals.loud_peaks(seconds, count=1) == [pytest.approx(2.0)]


def test_v3_schema_adds_subscores_and_hook_type_with_safe_keywords_only():
    items = ranking.RANKING_SCHEMA_V3["properties"]["candidates"]["items"]
    assert items["additionalProperties"] is False
    assert set(items["required"]) == set(items["properties"])
    assert set(ranking.SUBSCORES) | {"hook_type"} <= set(items["required"])
    assert items["properties"]["hook_type"]["enum"] == list(ranking.HOOK_TYPES)
    text = json.dumps(ranking.RANKING_SCHEMA_V3)
    for keyword in ("minimum", "maximum", "minLength", "maxLength", "maxItems", "multipleOf"):
        assert keyword not in text
    assert ranking.RANKING_SCHEMA["properties"]["candidates"]["items"]["required"][-1] == "payoff_end_word", \
        "the current schema is untouched"


def test_v3_prompt_keeps_the_approved_prompt_and_adds_the_new_sections():
    for line in ranking.PROMPT.splitlines():
        if line.strip():
            assert line in ranking.PROMPT_V3
    for section in ("WHAT A CLIP NEEDS", "PREFER", "AVOID", "SCORES", "SIGNALS", "{signals}"):
        assert section in ranking.PROMPT_V3
    assert "{signals}" not in ranking.PROMPT


def test_subscores_are_parsed_for_display_and_never_reorder():
    items = [{"start_word": 0, "end_word": 5, "hook_score": 25, "engagement_score": 26,
              "value_score": 3.0, "shareability_score": "9", "hook_type": "story"},
             {"start_word": 10, "end_word": 15, "hook_score": 0, "hook_type": "rant"},
             {"start_word": 20, "end_word": 25}]
    parsed = ranking.parse_candidates(json.dumps({"candidates": items}))
    assert [c.rank for c in parsed] == [1, 2, 3]
    assert parsed[0].subscores == {"hook_score": 25, "engagement_score": None, "value_score": 3,
                                   "shareability_score": None}
    assert parsed[0].hook_type == "story" and parsed[1].hook_type is None
    assert parsed[2].subscores is None and parsed[2].as_dict()["subscores"] is None


def test_prompt_version_changes_the_s5_key_only_when_set(tmp_path):
    from maclips.orchestrator import RunContext, cache_key
    from maclips.stages import STAGES_BY_ID
    s5 = STAGES_BY_ID["S5"]
    key = lambda cfg: cache_key(s5, RunContext("r", tmp_path, "hash", tmp_path, cfg), {"S1": "a", "S3": "b"})
    assert key({"candidate_count": 12}) == key({"candidate_count": 12, "prompt_version": None})
    assert key({"candidate_count": 12}) != key({"candidate_count": 12, "prompt_version": "v3"})
