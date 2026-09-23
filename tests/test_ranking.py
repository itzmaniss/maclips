"""S5 ranking: word-index contract, sentence snapping, filters, gates."""
from __future__ import annotations

import json

import pytest

from maclips import ranking
from maclips.ranking import Candidate, RankingError


def words(n=400, wps=2.5):
    """A transcript with a sentence every 10 words, at a steady pace."""
    out = []
    for i in range(n):
        text = f"w{i}" + ("." if i % 10 == 9 else "")
        out.append({"word": text, "start": i / wps, "end": (i + 1) / wps, "speaker": None})
    return out


def test_prompt_forbids_timestamps():
    """§2.2 item 3: hallucinated timestamps are impossible if none are requested."""
    assert "Never output a timestamp" in ranking.PROMPT
    assert "word index" in ranking.PROMPT


def test_sentence_boundaries_are_found():
    w = words(30)
    assert ranking.sentence_starts(w) == [0, 10, 20]
    assert ranking.sentence_ends(w) == [9, 19, 29]


def test_snapping_moves_outward_never_inward():
    """Snapping inward could cut the sentence that made the moment worth clipping."""
    w = words(40)
    starts, ends = ranking.sentence_starts(w), ranking.sentence_ends(w)
    c = ranking.snap(Candidate(1, 13, 26), starts, ends, len(w))
    assert c.start_word == 10, "snapped back to the sentence start"
    assert c.end_word == 29, "snapped forward to the sentence end"


def test_timestamps_are_looked_up_never_generated():
    w = words(40)
    c = ranking.resolve_times(Candidate(1, 10, 19), w)
    assert c.start == w[10]["start"]
    assert c.end == w[19]["end"]


def test_words_without_timing_do_not_produce_a_span():
    w = [{"word": "a.", "start": None, "end": None}]
    c = ranking.resolve_times(Candidate(1, 0, 0), w)
    assert c.start is None and c.end is None


def test_out_of_range_indices_are_dropped():
    w = words(100)
    kept, dropped = ranking.post_process([Candidate(1, 5000, 5100)], w)
    assert kept == []
    assert dropped["out_of_range"] == 1


def test_negative_indices_are_dropped():
    w = words(100)
    kept, dropped = ranking.post_process([Candidate(1, -5, 20)], w)
    assert dropped["out_of_range"] == 1 and kept == []


def test_duration_filter_applies_after_snapping():
    """A span inside the range can fall outside it once snapped outward."""
    w = words(400)
    # 10 words = 4s at 2.5 wps; ask for a very short span
    kept, dropped = ranking.post_process([Candidate(1, 10, 19)], w,
                                         min_duration_s=30, max_duration_s=60)
    assert kept == []
    assert dropped["duration"] == 1


def test_candidates_in_range_survive():
    w = words(400)
    # 100 words = 40s
    kept, _ = ranking.post_process([Candidate(1, 0, 99)], w,
                                   min_duration_s=30, max_duration_s=60)
    assert len(kept) == 1
    assert 30 <= kept[0].duration <= 60


def test_overlap_removal_keeps_the_higher_rank():
    w = words(400)
    a = Candidate(1, 0, 99)      # rank 1
    bb = Candidate(2, 50, 149)   # overlaps a, lower rank
    kept, dropped = ranking.post_process([a, bb], w, min_duration_s=30, max_duration_s=60)
    assert [c.rank for c in kept] == [1]
    assert dropped["overlap"] == 1


def test_non_overlapping_candidates_both_survive():
    w = words(400)
    kept, _ = ranking.post_process(
        [Candidate(1, 0, 99), Candidate(2, 200, 299)], w,
        min_duration_s=30, max_duration_s=60,
    )
    assert len(kept) == 2


def test_rank_order_is_preserved_and_never_rescored():
    """§5.4: the model's rank is the rank. No weighted blend."""
    w = words(400)
    kept, _ = ranking.post_process(
        [Candidate(3, 200, 299), Candidate(1, 0, 99)], w,
        min_duration_s=30, max_duration_s=60,
    )
    assert [c.rank for c in kept] == [1, 3]


# --------------------------------------------------------------------------- #
# Schema validation and the single retry
# --------------------------------------------------------------------------- #

def _payload(n, start=0, step=120):
    return json.dumps({"candidates": [
        {"start_word": start + i * step, "end_word": start + i * step + 99,
         "hook_text": f"hook {i}", "why": "because", "topic": "t",
         "brief_flags": []}
        for i in range(n)
    ]})


def test_malformed_json_retries_once_then_stops():
    calls = []

    def bad(prompt):
        calls.append(prompt)
        return "not json at all"

    with pytest.raises(RankingError, match="after one retry"):
        ranking.rank(words(400), llm_fn=bad, count=5)
    assert len(calls) == 2, "exactly one retry"
    assert "IMPORTANT" in calls[1], "the retry must tell the model what went wrong"


def test_a_valid_retry_is_accepted():
    calls = []

    def flaky(prompt):
        calls.append(prompt)
        return "garbage" if len(calls) == 1 else _payload(6)

    kept, meta = ranking.rank(words(1000), llm_fn=flaky, count=6)
    assert meta["attempts"] == 2
    assert len(kept) >= 5


def test_missing_candidates_array_is_invalid():
    with pytest.raises(RankingError, match="after one retry"):
        ranking.rank(words(400), llm_fn=lambda p: json.dumps({"items": []}), count=5)


def test_non_integer_indices_are_invalid():
    payload = json.dumps({"candidates": [
        {"start_word": "start", "end_word": 99, "hook_text": "h", "why": "w", "topic": "t"}
    ]})
    with pytest.raises(RankingError, match="after one retry"):
        ranking.rank(words(400), llm_fn=lambda p: payload, count=5)


def test_fewer_than_five_survivors_is_flagged_not_raised():
    """The gate belongs to the stage; rank() reports so the stage can phrase it."""
    kept, meta = ranking.rank(words(1000), llm_fn=lambda p: _payload(3), count=3)
    assert meta["insufficient"] is True
    assert len(kept) < ranking.MIN_SURVIVING_CANDIDATES


def test_five_survivors_is_enough():
    kept, meta = ranking.rank(words(1000), llm_fn=lambda p: _payload(5), count=5)
    assert "insufficient" not in meta
    assert len(kept) == 5


def test_speaker_labels_appear_only_when_asked():
    w = words(30)
    for x in w:
        x["speaker"] = "SPEAKER_00"
    assert "SPEAKER_00" not in ranking.build_transcript(w)
    assert "SPEAKER_00" in ranking.build_transcript(w, with_speakers=True)


def test_transcript_lines_carry_the_first_word_index_and_no_time():
    """Start times were removed from the lines (§5.1): the index and the words only."""
    lines = ranking.build_transcript(words(30)).splitlines()
    assert lines[0] == "[0] " + " ".join(f"w{i}" for i in range(9)) + " w9."
    assert lines[1].startswith("[10] w10 ")
    assert lines[2].startswith("[20] w20 ")


def test_labelled_lines_carry_the_speaker_after_the_index():
    w = words(30)
    for x in w:
        x["speaker"] = "SPEAKER_01"
    assert ranking.build_transcript(w, with_speakers=True).splitlines()[1].startswith(
        "[10] SPEAKER_01: w10 "
    )


def test_default_range_is_the_postable_range():
    """No brief range: 10-180 s (§6.1). 180 s is the Shorts maximum, the
    smallest of the three platforms'; 10 s is the minimum four briefs state."""
    assert (ranking.DEFAULT_MIN_DURATION_S, ranking.DEFAULT_MAX_DURATION_S) == (10.0, 180.0)


def test_default_range_reaches_the_prompt_with_its_word_guidance():
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return '{"candidates": []}'

    with pytest.raises(ranking.RankingError):
        ranking.rank(words(30), llm)
    assert "Target 10-180 seconds. Roughly 25-450 words" in seen[0]


def test_default_range_keeps_a_150_s_span_that_30_60_dropped():
    w = words(400)  # 2.5 words/s, so 375 words = 150 s
    kept, dropped = ranking.post_process([Candidate(1, 0, 374)], w)
    assert len(kept) == 1 and dropped["duration"] == 0


def test_prompt_describes_index_only_lines_and_asks_for_word_indices_only():
    assert "begins with the word index of its first word, in square brackets" in ranking.PROMPT
    assert "@" not in ranking.PROMPT
    assert "end time minus" not in ranking.PROMPT
    assert "Refer to moments only by word index" in ranking.PROMPT
