"""Step-4 ranking experiment on the cached benchmark transcript.

    uv run python scripts/ranking_experiment.py

Four ranking runs over work/bench/transcript.json:
  A1, A2 — unlabelled transcript (what S5 actually sees since step 2b)
  B1, B2 — speaker-labelled transcript (from the cached full diarization)
  C1, C2 — unlabelled, on Haiku 4.5 (experiment only; S5 stays on Sonnet)

Reports run-to-run overlap within each condition and across them, logs tokens
and cost per call at the $2/$10 Sonnet 5 rate, re-measures window
diarization on the real candidate windows, and writes a blind review sheet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from maclips import config, ranking
from maclips.bench import measure
from maclips.experiment import Run, blind_sheet, compare, save_key
from maclips.llm import complete, rank_fn

TRANSCRIPT = config.WORK_DIR / "bench" / "transcript.json"
OUT = config.WORK_DIR / "experiment"
COUNT = 12
HAIKU_MODEL = "anthropic/claude-haiku-4-5"


def haiku_fn(prompt: str, usage_sink: list | None = None) -> str:
    """Haiku 4.5 takes no effort parameter and no adaptive thinking: none is sent."""
    return complete(prompt, model=HAIKU_MODEL, json_only=True, max_tokens=16000,
                    temperature=None, usage_sink=usage_sink)


def load_words() -> list[dict]:
    if not TRANSCRIPT.is_file():
        print(f"missing {TRANSCRIPT}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(TRANSCRIPT.read_text())["words"]


def one_run(label: str, condition: str, words: list[dict]) -> Run | None:
    usage: list = []

    fn = haiku_fn if condition == "haiku" else rank_fn

    def call(prompt: str) -> str:
        return fn(prompt, usage_sink=usage)

    with measure(f"rank {label}") as m:
        candidates, meta = ranking.rank(
            words, llm_fn=call, count=COUNT,
            with_speakers=(condition == "labelled"),
        )
    tokens = sum(u["input_tokens"] for u in usage), sum(u["output_tokens"] for u in usage)
    reasoning = sum(u["reasoning_tokens"] for u in usage)
    cost = sum(u["cost_usd"] for u in usage)
    print(f"-> {label} ({condition}): {len(candidates)} candidates in {m.wall_s:.1f}s | "
          f"in={tokens[0]} out={tokens[1]} reasoning={reasoning} | cost ${cost:.3f} | "
          f"attempts={meta['attempts']} dropped={meta['dropped']}", flush=True)
    if meta.get("insufficient") and condition == "haiku":
        # Gated per model: a Haiku failure does not stop the Sonnet runs.
        print(f"GATED: {label} kept {len(candidates)}; left out of the comparison.",
              file=sys.stderr)
        return None
    if meta.get("insufficient"):
        # The S5 five-survivor gate (§5.4 item 5): stop before any further call.
        print(f"STOPPED: {label} kept {len(candidates)}, fewer than "
              f"{ranking.MIN_SURVIVING_CANDIDATES}; no further runs.", file=sys.stderr)
        raise SystemExit(2)
    return Run(label=label, condition=condition,
               candidates=[c.as_dict() for c in candidates], usage=usage)


def main() -> int:
    words = load_words()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"transcript: {len(words)} words", flush=True)

    unlabelled = [{**w, "speaker": None} for w in words]
    runs = [
        one_run("A1", "unlabelled", unlabelled),
        one_run("A2", "unlabelled", unlabelled),
        one_run("C1", "haiku", unlabelled),
        one_run("C2", "haiku", unlabelled),
        one_run("B1", "labelled", words),
        one_run("B2", "labelled", words),
    ]
    runs = [r for r in runs if r is not None]

    total_in = sum(u["input_tokens"] for r in runs for u in r.usage)
    total_out = sum(u["output_tokens"] for r in runs for u in r.usage)
    total = sum(u["cost_usd"] for r in runs for u in r.usage)
    print(f"\ntotal: in={total_in} out={total_out} cost ${total:.3f} across {len(runs)} runs",
          flush=True)

    print("\n=== span overlap (IoU >= 0.5 counts as the same moment) ===", flush=True)
    result = compare(runs)
    for name, value in result["within_condition"].items():
        print(f"  within  {name}: {value:.0%}" if value is not None else f"  within  {name}: n/a")
    for name, value in result["across_condition"].items():
        print(f"  across  {name}: {value:.0%}" if value is not None else f"  across  {name}: n/a")
    print(f"\n  within-condition mean : {result['within_mean']:.0%}")
    print(f"  across-condition mean : {result['across_mean']:.0%}")
    print(f"  gap                   : {result['gap']:+.0%}")
    print(f"\n  {result['verdict']}", flush=True)

    # ---- window diarization on the REAL candidate windows -----------------
    print("\n=== window diarization on real candidate windows ===", flush=True)
    from maclips import diarize as s4
    from maclips.audio import SAMPLE_RATE, load_wav

    wav = config.WORK_DIR / "bench" / "audio.wav"
    if wav.is_file():
        waveform = load_wav(wav)
        spans = [(c["start"], c["end"]) for c in runs[0].candidates]
        with measure("S4 real windows") as m:
            windows = s4.diarize_windows(waveform, spans)
        covered = sum(e - s + 20.0 for s, e in spans)
        print(f"-> {len(windows)} windows, {covered:.0f}s of audio "
              f"({covered / (len(waveform) / SAMPLE_RATE):.1%}) in {m.wall_s:.1f}s", flush=True)
        for w in windows:
            sig = w.result.significant_speakers()
            print(f"     window {w.index} [{w.start:.0f}-{w.end:.0f}s]: "
                  f"{len(sig)} significant of {len(w.result.speakers)} labels", flush=True)
    else:
        print(f"   skipped: {wav} not present", flush=True)

    # ---- blind sheet -------------------------------------------------------
    sheet, key = blind_sheet(runs, words, seed=20260920)
    sheet_path = OUT / "blind-review.md"
    sheet_path.write_text(sheet)
    save_key(key, OUT / "blind-key.json")
    (OUT / "runs.json").write_text(json.dumps(
        [{"label": r.label, "condition": r.condition, "candidates": r.candidates,
          "usage": r.usage} for r in runs], indent=1))
    (OUT / "comparison.json").write_text(json.dumps(result, indent=1))

    print(f"\nblind sheet: {sheet_path}  ({len(key)} distinct moments)", flush=True)
    print(f"key (do not read before rating): {OUT / 'blind-key.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
