"""Build step 2 benchmark: S0 download, S2 extract, S3 transcribe/align, S4 diarize.

Run with:  uv run python scripts/benchmark_step2.py <url-or-path>

Measures wall time, peak RSS, swap delta and macOS memory pressure per stage.
Models are loaded one at a time and released between stages, and the decoded
waveform is loaded once and shared, so the figures reflect the pipeline rather
than accumulated allocator slack.

MPS-vs-CPU is measured on a short slice rather than the whole source: a CPU
alignment pass over two hours would dominate the run for a ratio a few minutes
of audio establishes just as well.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from maclips import config, diarize as s4, ingest, transcribe as s3
from maclips.audio import SAMPLE_RATE, load_wav
from maclips.bench import measure, memory_pressure, render_table, swap_used_mb
from maclips.ffmpeg import extract_audio, probe
from maclips.gates import check_environment
from maclips.resources import release

SLICE_S = 300.0  # 5 minutes, for the MPS-vs-CPU comparison


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def record(rows: list, m, audio_s: float = 0.0) -> None:
    """Append a stage's metrics and print them immediately.

    Printed per stage rather than only in the closing table: build step 2's
    first run died in S4 and took every earlier timing with it, because nothing
    had been emitted yet.
    """
    rows.append(m)
    xrt = f"  ({audio_s / m.wall_s:.1f}x realtime)" if audio_s and m.wall_s else ""
    flag = "   ** SWAP GREW — this timing measures paging **" if m.swapped else ""
    print(f"-> {m.name}: {m.wall_s:.1f}s{xrt}  peak RSS {m.peak_rss_mb:.0f}M  "
          f"swap {m.swap_delta_mb:+.0f}M  [{m.pressure_at_peak}]{flag}", flush=True)


def main(target: str) -> int:
    check_environment()
    print(f"baseline: swap {swap_used_mb():.0f} MB, pressure {memory_pressure()}", flush=True)

    rows = []
    work = config.WORK_DIR / "bench"
    work.mkdir(parents=True, exist_ok=True)

    # ---- S0 ---------------------------------------------------------------
    # S4's gate runs first: it costs a second, and discovering a 403 only after
    # eight minutes of transcription is how the first run was wasted.
    banner("pre-flight: gated model access")
    s4.check_access()
    print("community-1 files are fetchable", flush=True)

    banner("S0  ingest")
    with measure("S0 ingest") as m:
        rec = ingest.resolve(target, config.WORK_DIR / "sources")
    meta = probe(rec.path)
    record(rows, m)
    size_mb = rec.path.stat().st_size / 2**20
    print(f"file      : {rec.path.name}  ({size_mb:.0f} MiB)")
    print(f"title     : {rec.title or '(local file)'}")
    print(f"duration  : {meta['duration_s']:.0f}s")
    audio_s = meta["duration_s"]

    # ---- S2 ---------------------------------------------------------------
    banner("S2  audio extract")
    wav = work / "audio.wav"
    with measure("S2 extract") as m:
        extract_audio(rec.path, wav, sample_rate=SAMPLE_RATE)
    record(rows, m, audio_s)
    print(f"wav       : {wav.stat().st_size / 2**20:.0f} MiB")

    with measure("   load wav") as m:
        waveform = load_wav(wav)
    print(f"waveform  : {waveform.nbytes / 2**20:.0f} MiB in memory, "
          f"{len(waveform) / SAMPLE_RATE:.0f}s")

    # ---- S3 ---------------------------------------------------------------
    banner("S3  transcribe (Whisper on MLX)")
    with measure("S3 transcribe") as m:
        segments, language = s3.transcribe(waveform)
    record(rows, m, audio_s)
    print(f"language  : {language}   segments: {len(segments)}")

    banner("S3  align (wav2vec2 on torch/MPS)")
    with measure("S3 align") as m:
        words = s3.align(segments, waveform, language or "en", device="mps")
    record(rows, m, audio_s)
    transcript = s3.Transcript(words=words, language=language, segments=segments)
    print(f"words     : {len(words)}   coverage: {transcript.coverage:.2%}")

    # ---- S4 ---------------------------------------------------------------
    banner("S4  diarize (pyannote community-1 on MPS)")
    with measure("S4 diarize") as m:
        diar = s4.diarize(waveform, device="mps")
    record(rows, m, audio_s)
    print(f"speakers  : {len(diar.speakers)} {diar.speakers}   turns: {len(diar.turns)}")

    s4.assign_speakers(transcript.words, diar)
    labelled = sum(1 for w in transcript.words if w.speaker)
    print(f"labelled  : {labelled}/{len(transcript.words)} words "
          f"({labelled / max(len(transcript.words), 1):.1%})")

    # ---- MPS vs CPU on a slice -------------------------------------------
    banner(f"MPS vs CPU  (first {SLICE_S:.0f}s slice)")
    sl = waveform[: int(SLICE_S * SAMPLE_RATE)]
    sl_segments = [s for s in segments if s.get("end", 0) <= SLICE_S] or segments[:20]
    compare = []
    for device in ("mps", "cpu"):
        with measure(f"align {device}") as m:
            s3.align(sl_segments, sl, language or "en", device=device)
        record(compare, m, SLICE_S)
    for device in ("mps", "cpu"):
        with measure(f"diarize {device}") as m:
            s4.diarize(sl, device=device)
        compare.append(m)

    release(waveform)

    # ---- report -----------------------------------------------------------
    banner("RESULTS")
    print(render_table(rows, audio_duration_s=audio_s))
    print(f"\nMPS vs CPU on a {SLICE_S:.0f}s slice:")
    print(render_table(compare, audio_duration_s=SLICE_S))

    # Persist the labelled transcript: regenerating spot-checks should never
    # require re-running a nine-minute diarization pass.
    import json

    out_json = work / "transcript.json"
    out_json.write_text(json.dumps({
        "source": str(rec.path),
        "duration_s": audio_s,
        "language": transcript.language,
        "coverage": transcript.coverage,
        "speakers": diar.speakers,
        "turns": [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in diar.turns],
        "words": [w.as_dict() for w in transcript.words],
    }, indent=1))
    print(f"\nlabelled transcript written: {out_json} "
          f"({out_json.stat().st_size / 2**20:.0f} MiB)")

    print("\nspot-checks (verify by ear):")
    for s in s4.sample_labels(transcript.words, count=3):
        mm, ss = divmod(int(s["start"]), 60)
        print(f"  [{mm:02d}:{ss:02d}] {s['speaker']} ({s['word_count']} words total): "
              f"{s['text'][:110]}")

    swapped = [r.name for r in rows if r.swapped]
    if swapped:
        print(f"\n** SWAP GREW during: {', '.join(swapped)} — "
              f"those timings measure paging, not the pipeline. **")
    else:
        print("\nno swap growth during any stage; timings reflect compute.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
