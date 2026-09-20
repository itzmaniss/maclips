"""Step 2b re-benchmark: split-stream ingest and window-only diarization.

    uv run python scripts/benchmark_step2b.py <url> [--keep-cache]

Cold cache for S0 (the cached streams are removed first), warm models. Answers
three questions the restructure was meant to improve:

  * **time to S2 start** — how soon can transcription begin? Previously this
    was the whole merged download; now it is the audio stream alone.
  * **S2-S3 wall** — the work that must finish before ranking.
  * **projected time-to-candidates** — the above plus ranking, which is what
    the operator actually waits for.

Window-only diarization is measured separately against the same spans S5 would
produce, to check the projection in §2.6 that it costs ~1.2 min rather than 9.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from maclips import config, diarize as s4, ingest, transcribe as s3
from maclips.audio import SAMPLE_RATE, load_wav
from maclips.bench import measure, memory_pressure, render_table, swap_used_mb
from maclips.ffmpeg import extract_audio
from maclips.gates import check_environment

WORK = config.WORK_DIR / "bench2b"
CANDIDATE_COUNT = 12
CANDIDATE_LEN_S = 60.0


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def clear_cache(video_id: str, dest: Path) -> int:
    removed = 0
    for p in dest.glob(f"{video_id}.*"):
        if p.is_file():
            p.unlink()
            removed += 1
    return removed


def spread_spans(duration: float, count: int, length: float) -> list[tuple[float, float]]:
    """Candidate spans spread across the source, as §5.3 requires of the ranker."""
    step = max(length, (duration - length) / max(count, 1))
    return [(i * step, min(i * step + length, duration)) for i in range(count)]


def main(url: str, keep_cache: bool = False) -> int:
    check_environment()
    s4.check_access()
    WORK.mkdir(parents=True, exist_ok=True)
    sources = config.WORK_DIR / "sources"
    print(f"baseline: swap {swap_used_mb():.0f} MB, pressure {memory_pressure()}", flush=True)

    meta = ingest.probe_url(url)
    if not keep_cache:
        n = clear_cache(meta.video_id, sources)
        print(f"cleared {n} cached stream file(s) for {meta.video_id} — S0 is cold", flush=True)

    rows = []

    # ---- S0: split download, timing the audio-ready signal ----------------
    banner("S0  split-stream ingest (cold)")
    t0 = time.perf_counter()
    audio_ready_at: list[float] = []

    def on_audio(path: Path) -> None:
        audio_ready_at.append(time.perf_counter() - t0)
        print(f"   audio ready at {audio_ready_at[0]:.1f}s: {path.name}", flush=True)

    with measure("S0 audio") as m_audio:
        record = ingest.download_split(url, sources, on_audio_ready=on_audio)
    rows.append(m_audio)
    time_to_s2 = audio_ready_at[0] if audio_ready_at else m_audio.wall_s
    print(f"-> time to S2 start: {time_to_s2:.1f}s "
          f"({record.audio_path.stat().st_size / 2**20:.0f} MiB audio)", flush=True)

    # ---- S2 + S3 run while the video is still downloading -----------------
    banner("S2  audio extract  (video still downloading)")
    wav = WORK / "audio.wav"
    with measure("S2 extract") as m:
        extract_audio(record.audio_source, wav, sample_rate=SAMPLE_RATE)
    rows.append(m)
    waveform = load_wav(wav)
    audio_s = len(waveform) / SAMPLE_RATE
    print(f"   {audio_s:.0f}s audio, {waveform.nbytes / 2**20:.0f} MiB in memory", flush=True)

    banner("S3  transcribe + align")
    with measure("S3 transcribe") as m:
        segments, language = s3.transcribe(waveform)
    rows.append(m)
    with measure("S3 align") as m:
        words = s3.align(segments, waveform, language or "en", device="mps")
    rows.append(m)
    transcript = s3.Transcript(words=words, language=language, segments=segments)
    from maclips.stages import MAX_WEAK_WORD_FRACTION, MIN_ALIGNMENT_SCORE

    weak = transcript.weak_fraction(MIN_ALIGNMENT_SCORE)
    print(f"   {len(words)} words, weak {weak:.2%} at min_score "
          f"{MIN_ALIGNMENT_SCORE} (gate: {MAX_WEAK_WORD_FRACTION:.0%}), "
          f"coverage {transcript.coverage:.2%}", flush=True)

    s2_s3_wall = sum(r.wall_s for r in rows[1:])

    # ---- S4 on candidate windows -----------------------------------------
    banner(f"S4  window-only diarization ({CANDIDATE_COUNT} x {CANDIDATE_LEN_S:.0f}s + 10s pad)")
    spans = spread_spans(audio_s, CANDIDATE_COUNT, CANDIDATE_LEN_S)
    with measure("S4 windows") as m:
        windows = s4.diarize_windows(waveform, spans)
    rows.append(m)
    covered = sum(min(audio_s, e + 10.0) - max(0.0, s - 10.0) for s, e in spans)
    print(f"   {len(windows)} windows, {covered:.0f}s of audio "
          f"({covered / audio_s:.1%} of the source)", flush=True)
    for w in windows[:4]:
        sig = w.result.significant_speakers()
        shares = ", ".join(f"{k} {v:.0%}" for k, v in sorted(w.result.shares().items()))
        print(f"     window {w.index} [{w.start:.0f}-{w.end:.0f}s]: "
              f"{len(sig)} significant of {len(w.result.speakers)} labels ({shares})", flush=True)

    # ---- video stream ------------------------------------------------------
    banner("S0  video stream (was downloading throughout)")
    with measure("S0 video wait") as m:
        video = ingest.await_video(record, timeout=3600)
    rows.append(m)
    print(f"   {video.name} ({video.stat().st_size / 2**20:.0f} MiB); "
          f"waited {m.wall_s:.1f}s after S4", flush=True)

    # ---- report ------------------------------------------------------------
    banner("RESULTS")
    print(render_table(rows, audio_duration_s=audio_s))
    print(f"\ntime to S2 start        : {time_to_s2:>7.1f}s")
    print(f"S2-S3 wall              : {s2_s3_wall:>7.1f}s")
    print(f"projected time-to-candidates (S0 audio + S2 + S3 + ranking):")
    for rank_s in (20.0, 40.0, 60.0):
        total = time_to_s2 + s2_s3_wall + rank_s
        print(f"   ranking {rank_s:>4.0f}s -> {total / 60:.1f} min")
    print("\n   (S5 is still a stub; the ranking figures are assumptions, not "
          "measurements — a single Sonnet call over ~30k tokens.)")
    swapped = [r.name for r in rows if r.swapped]
    print(f"\n** SWAP GREW during: {', '.join(swapped)} **" if swapped
          else "\nno swap growth during any stage; timings reflect compute.")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0] if args else "", "--keep-cache" in sys.argv))
