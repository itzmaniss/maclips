"""Step-4 ranking experiment: do speaker labels change what gets picked?

Diarization moved behind ranking in step 2b, so S5 now ranks an unlabelled
transcript (§2.2 item 1). That was a reasoned trade, not a measured one. This
module measures it.

**The comparison that matters is A<->B against within-condition noise.** Two
runs of the same prompt do not agree perfectly — the model is sampling. So
"labelled and unlabelled runs disagree" proves nothing on its own; the question
is whether they disagree *more* than two runs of the same condition do. Labels
only matter if A<->B overlap is clearly below A1<->A2 and B1<->B2.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IOU_SAME_MOMENT = 0.5
"""Two spans count as the same moment at this IoU or above."""


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two time spans."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    overlap = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - overlap
    return overlap / union if union > 0 else 0.0


def match_spans(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    threshold: float = IOU_SAME_MOMENT,
) -> list[tuple[int, int, float]]:
    """Greedy best-first pairing of spans across two runs.

    Greedy on descending IoU rather than positional: the runs are independently
    ranked, so candidate 3 in one may be candidate 7 in the other.
    """
    pairs = sorted(
        ((i, j, iou(a, b)) for i, a in enumerate(left) for j, b in enumerate(right)),
        key=lambda p: -p[2],
    )
    used_l: set[int] = set()
    used_r: set[int] = set()
    matched = []
    for i, j, score in pairs:
        if score < threshold or i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        matched.append((i, j, score))
    return matched


def overlap_fraction(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    threshold: float = IOU_SAME_MOMENT,
) -> float:
    """Share of the smaller run's candidates that the other run also found."""
    if not left or not right:
        return 0.0
    return len(match_spans(left, right, threshold)) / min(len(left), len(right))


@dataclass
class Run:
    """One ranking pass under one condition."""

    label: str            # "A1", "B2", ...
    condition: str        # "unlabelled" | "labelled"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    usage: list[dict[str, Any]] = field(default_factory=list)

    def spans(self) -> list[tuple[float, float]]:
        return [
            (c["start"], c["end"]) for c in self.candidates
            if c.get("start") is not None and c.get("end") is not None
        ]


def compare(runs: list[Run], threshold: float = IOU_SAME_MOMENT) -> dict[str, Any]:
    """Within-condition noise versus the cross-condition difference."""
    by_label = {r.label: r for r in runs}

    def pair(a: str, b: str) -> float | None:
        if a not in by_label or b not in by_label:
            return None
        return overlap_fraction(by_label[a].spans(), by_label[b].spans(), threshold)

    within = {"A1<->A2": pair("A1", "A2"), "B1<->B2": pair("B1", "B2")}
    across = {
        f"{a}<->{b}": pair(a, b)
        for a in ("A1", "A2") for b in ("B1", "B2")
    }
    within_vals = [v for v in within.values() if v is not None]
    across_vals = [v for v in across.values() if v is not None]
    within_mean = sum(within_vals) / len(within_vals) if within_vals else 0.0
    across_mean = sum(across_vals) / len(across_vals) if across_vals else 0.0

    return {
        "threshold": threshold,
        "within_condition": within,
        "across_condition": across,
        "within_mean": within_mean,
        "across_mean": across_mean,
        "gap": within_mean - across_mean,
        "verdict": _verdict(within_mean, across_mean, within_vals, across_vals),
    }


def _verdict(within_mean: float, across_mean: float,
             within: list[float], across: list[float]) -> str:
    if not within or not across:
        return "inconclusive: not all four runs are present"
    gap = within_mean - across_mean
    if gap <= 0:
        return (
            "no detectable effect — labelled and unlabelled runs agree as much as "
            "two runs of the same condition do. Run-to-run sampling noise "
            "dominates, so nothing here argues for restoring pre-ranking diarization."
        )
    if gap < 0.15:
        return (
            f"no clear effect — the cross-condition drop ({gap:.0%}) is within the "
            f"run-to-run noise already present ({within_mean:.0%} within-condition "
            "agreement). Not evidence that labels change selection."
        )
    return (
        f"labels appear to change selection: cross-condition agreement is "
        f"{gap:.0%} below within-condition. Worth weighing against the 8.8x "
        "diarization cost before reverting the ordering."
    )


def blind_sheet(runs: list[Run], words: list[dict[str, Any]], seed: int = 0) -> tuple[str, dict]:
    """Shuffled markdown review sheet plus the key that unblinds it.

    Candidates from every run are pooled and shuffled, with nothing in the
    sheet indicating which run produced which — not the order, not an id that
    encodes it, not the excerpt formatting. Identical moments found by several
    runs are merged into one row so the same clip is not rated twice.
    """
    pooled: list[dict[str, Any]] = []
    for run in runs:
        for c in run.candidates:
            if c.get("start") is None or c.get("end") is None:
                continue
            pooled.append({**c, "_run": run.label, "_condition": run.condition})

    merged: list[dict[str, Any]] = []
    for item in pooled:
        span = (item["start"], item["end"])
        for existing in merged:
            if iou(span, (existing["start"], existing["end"])) >= IOU_SAME_MOMENT:
                existing["_runs"].append(item["_run"])
                break
        else:
            merged.append({**item, "_runs": [item["_run"]]})

    rng = random.Random(seed)
    rng.shuffle(merged)

    lines = [
        "# Blind candidate review",
        "",
        f"{len(merged)} distinct moments pooled from {len(runs)} ranking runs.",
        "Which run produced which is not shown, and the order is shuffled.",
        "",
        "Mark each `worth reviewing?` as **yes** or **no**, then send the file back.",
        "",
        "---",
        "",
    ]
    key: dict[str, list[str]] = {}
    for n, item in enumerate(merged, 1):
        cid = f"C{n:02d}"
        key[cid] = item["_runs"]
        mm, ss = divmod(int(item["start"]), 60)
        mm2, ss2 = divmod(int(item["end"]), 60)
        excerpt = _excerpt(words, item["start_word"], item["end_word"])
        lines += [
            f"## {cid}  [{mm:02d}:{ss:02d} – {mm2:02d}:{ss2:02d}]  ({item['duration']:.0f}s)",
            "",
            f"**Hook:** {item.get('hook_text') or '(none)'}",
            "",
            f"**Why:** {item.get('why') or '(none)'}",
            "",
            f"**Topic:** {item.get('topic') or '(none)'}",
            "",
            "> " + excerpt.replace("\n", "\n> "),
            "",
            "**worth reviewing?** ",
            "",
            "---",
            "",
        ]
    return "\n".join(lines), key


def _excerpt(words: list[dict[str, Any]], start: int, end: int, limit: int = 900) -> str:
    text = " ".join(
        w["word"] for w in words[max(0, start):min(len(words), end + 1)]
    )
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def unblind(key: dict[str, list[str]], ratings: dict[str, bool],
            runs: list[Run]) -> dict[str, Any]:
    """Yes-rate per condition, once ratings come back."""
    condition_of = {r.label: r.condition for r in runs}
    tally: dict[str, dict[str, int]] = {}
    for cid, run_labels in key.items():
        if cid not in ratings:
            continue
        for condition in {condition_of.get(r, "?") for r in run_labels}:
            bucket = tally.setdefault(condition, {"yes": 0, "no": 0})
            bucket["yes" if ratings[cid] else "no"] += 1

    out = {}
    for condition, counts in tally.items():
        total = counts["yes"] + counts["no"]
        out[condition] = {
            **counts, "total": total,
            "yes_rate": counts["yes"] / total if total else 0.0,
        }
    return {"per_condition": out, "rated": len(ratings), "pooled": len(key)}


def save_key(key: dict[str, list[str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, indent=1))
    return path


def content_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]
