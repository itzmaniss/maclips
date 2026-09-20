"""The A/B ranking experiment: IoU matching, blinding, and unblinding."""
from __future__ import annotations

import pytest

from maclips.experiment import (
    IOU_SAME_MOMENT,
    Run,
    blind_sheet,
    compare,
    iou,
    match_spans,
    overlap_fraction,
    unblind,
)


def cand(start, end, rank=1, hook="h", word_start=0, word_end=10):
    return {"rank": rank, "start": start, "end": end, "duration": end - start,
            "hook_text": hook, "why": "w", "topic": "t",
            "start_word": word_start, "end_word": word_end}


def words(n=200):
    return [{"word": f"w{i}", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(n)]


def test_iou_is_symmetric_and_bounded():
    assert iou((0, 60), (30, 90)) == iou((30, 90), (0, 60))
    assert 0.0 <= iou((0, 60), (30, 90)) <= 1.0
    assert iou((0, 60), (0, 60)) == 1.0
    assert iou((0, 60), (60, 120)) == 0.0


def test_matching_is_greedy_on_best_overlap_not_position():
    """Runs rank independently, so candidate 1 here may be candidate 3 there."""
    left = [(200, 260), (0, 60)]
    right = [(0, 60), (200, 260)]
    matched = match_spans(left, right)
    assert len(matched) == 2
    assert (0, 1) in [(i, j) for i, j, _ in matched]


def test_a_span_cannot_match_twice():
    left = [(0, 60)]
    right = [(0, 60), (1, 61)]
    assert len(match_spans(left, right)) == 1


def test_below_threshold_is_not_a_match():
    assert match_spans([(0, 60)], [(50, 110)], threshold=IOU_SAME_MOMENT) == []


def test_overlap_fraction_uses_the_smaller_run():
    """Otherwise a run that returned fewer candidates looks artificially worse."""
    assert overlap_fraction([(0, 60)], [(0, 60), (200, 260), (400, 460)]) == 1.0


def test_empty_runs_do_not_crash():
    assert overlap_fraction([], [(0, 60)]) == 0.0


# --------------------------------------------------------------------------- #
# The comparison's central claim
# --------------------------------------------------------------------------- #

def _runs(a1, a2, b1, b2):
    return [
        Run("A1", "unlabelled", [cand(*s) for s in a1]),
        Run("A2", "unlabelled", [cand(*s) for s in a2]),
        Run("B1", "labelled", [cand(*s) for s in b1]),
        Run("B2", "labelled", [cand(*s) for s in b2]),
    ]


def test_identical_runs_show_no_label_effect():
    spans = [(0, 60), (200, 260), (400, 460)]
    result = compare(_runs(spans, spans, spans, spans))
    assert result["within_mean"] == 1.0
    assert result["across_mean"] == 1.0
    assert result["gap"] == 0.0
    assert "no detectable effect" in result["verdict"]


def test_noise_within_conditions_is_not_read_as_a_label_effect():
    """The trap this experiment exists to avoid."""
    a1 = [(0, 60), (200, 260), (400, 460)]
    a2 = [(0, 60), (600, 660), (800, 860)]     # A disagrees with itself
    b1 = [(0, 60), (200, 260), (1000, 1060)]
    b2 = [(0, 60), (600, 660), (1200, 1260)]
    result = compare(_runs(a1, a2, b1, b2))
    assert result["gap"] < 0.15
    assert "no clear effect" in result["verdict"] or "no detectable" in result["verdict"]


def test_a_real_label_effect_is_reported():
    a1 = a2 = [(0, 60), (200, 260), (400, 460)]
    b1 = b2 = [(2000, 2060), (2200, 2260), (2400, 2460)]
    result = compare(_runs(a1, a2, b1, b2))
    assert result["within_mean"] == 1.0
    assert result["across_mean"] == 0.0
    assert "labels appear to change selection" in result["verdict"]


def test_missing_runs_are_inconclusive_not_silently_zero():
    result = compare([Run("A1", "unlabelled", [cand(0, 60)])])
    assert "inconclusive" in result["verdict"]


# --------------------------------------------------------------------------- #
# Blinding
# --------------------------------------------------------------------------- #

def test_sheet_never_reveals_the_condition():
    runs = _runs([(0, 60)], [(200, 260)], [(400, 460)], [(600, 660)])
    sheet, key = blind_sheet(runs, words(400))
    for token in ("A1", "A2", "B1", "B2", "unlabelled", "labelled"):
        assert token not in sheet, f"{token} leaks the condition"
    assert len(key) == 4


def test_the_same_moment_found_twice_is_rated_once():
    runs = _runs([(0, 60)], [(1, 61)], [(2, 62)], [(3, 63)])
    sheet, key = blind_sheet(runs, words(400))
    assert len(key) == 1, "near-identical spans must merge"
    assert sorted(next(iter(key.values()))) == ["A1", "A2", "B1", "B2"]


def test_sheet_is_shuffled_deterministically():
    runs = _runs([(0, 60)], [(200, 260)], [(400, 460)], [(600, 660)])
    one, _ = blind_sheet(runs, words(400), seed=7)
    two, _ = blind_sheet(runs, words(400), seed=7)
    assert one == two
    three, _ = blind_sheet(runs, words(400), seed=8)
    assert isinstance(three, str)


def test_sheet_carries_what_a_rater_needs():
    runs = _runs([(0, 60)], [(200, 260)], [(400, 460)], [(600, 660)])
    sheet, _ = blind_sheet(runs, words(400))
    assert "worth reviewing?" in sheet
    assert "**Hook:**" in sheet
    assert "00:00" in sheet


def test_unblinding_reports_yes_rate_per_condition():
    runs = _runs([(0, 60)], [(200, 260)], [(400, 460)], [(600, 660)])
    _, key = blind_sheet(runs, words(400))
    ratings = {cid: (i % 2 == 0) for i, cid in enumerate(sorted(key))}
    out = unblind(key, ratings, runs)
    assert out["rated"] == 4
    assert set(out["per_condition"]) <= {"unlabelled", "labelled"}
    for stats in out["per_condition"].values():
        assert 0.0 <= stats["yes_rate"] <= 1.0
        assert stats["yes"] + stats["no"] == stats["total"]


def test_unrated_candidates_are_excluded_not_counted_as_no():
    runs = _runs([(0, 60)], [(200, 260)], [(400, 460)], [(600, 660)])
    _, key = blind_sheet(runs, words(400))
    first = sorted(key)[0]
    out = unblind(key, {first: True}, runs)
    assert out["rated"] == 1
    assert sum(s["total"] for s in out["per_condition"].values()) == 1


def test_a_merged_moment_counts_for_every_condition_that_found_it():
    runs = _runs([(0, 60)], [(0, 60)], [(0, 60)], [(0, 60)])
    _, key = blind_sheet(runs, words(400))
    cid = next(iter(key))
    out = unblind(key, {cid: True}, runs)
    assert out["per_condition"]["unlabelled"]["yes"] == 1
    assert out["per_condition"]["labelled"]["yes"] == 1
