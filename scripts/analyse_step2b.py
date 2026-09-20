"""One-off data run: alignment-score distribution and the SPEAKER_02 question.

Produces the two things step 2b needs before thresholds can be chosen rather
than invented:

1. the distribution of whispermlx alignment scores on the benchmark source, so
   the S3 weak-word gate can be set from data;
2. full-source diarization, so SPEAKER_02's share of speaking time can be
   measured and spot-checked by ear.

Writes work/bench/transcript.json so neither ever needs re-running.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from maclips import config, diarize as s4, transcribe as s3
from maclips.audio import SAMPLE_RATE, load_wav
from maclips.bench import measure
from maclips.gates import check_environment

WORK = config.WORK_DIR / "bench"


def main() -> int:
    check_environment()
    s4.check_access()
    wav = WORK / "audio.wav"
    if not wav.is_file():
        print(f"missing {wav}; run the benchmark first", file=sys.stderr)
        return 1

    waveform = load_wav(wav)
    print(f"audio: {len(waveform) / SAMPLE_RATE:.0f}s", flush=True)

    with measure("transcribe") as m:
        segments, language = s3.transcribe(waveform)
    print(f"-> transcribe {m.wall_s:.1f}s  language={language} segments={len(segments)}", flush=True)

    with measure("align") as m:
        words = s3.align(segments, waveform, language or "en", device="mps")
    print(f"-> align {m.wall_s:.1f}s  words={len(words)}", flush=True)

    transcript = s3.Transcript(words=words, language=language, segments=segments)
    stats = transcript.score_stats()
    print("\n=== alignment score distribution ===", flush=True)
    print(json.dumps(stats, indent=1), flush=True)
    print("\nweak fraction at candidate thresholds:")
    for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        print(f"  min_score {thr:<5} -> weak {transcript.weak_fraction(thr):.3%}", flush=True)

    print("\n=== full-source diarization ===", flush=True)
    with measure("diarize full") as m:
        diar = s4.diarize(waveform, max_speakers=None)
    print(f"-> diarize {m.wall_s:.1f}s  speakers={diar.speakers}", flush=True)

    s4.assign_speakers(transcript.words, diar)
    shares = diar.shares()
    speaking = diar.speaking_time()
    print("\nspeaker shares (of speaking time):")
    for spk in sorted(shares, key=lambda s: -shares[s]):
        words_for = sum(1 for w in transcript.words if w.speaker == spk)
        print(f"  {spk}: {shares[spk]:6.2%}  {speaking[spk]:7.1f}s  {words_for:6d} words", flush=True)
    print(f"significant (>=5%): {diar.significant_speakers()}", flush=True)

    out = WORK / "transcript.json"
    out.write_text(json.dumps({
        "duration_s": len(waveform) / SAMPLE_RATE,
        "language": transcript.language,
        "score_stats": stats,
        "speakers": diar.speakers,
        "shares": shares,
        "speaking_time_s": speaking,
        "turns": [{"start": t.start, "end": t.end, "speaker": t.speaker} for t in diar.turns],
        "words": [w.as_dict() for w in transcript.words],
    }, indent=1))
    print(f"\nwrote {out} ({out.stat().st_size / 2**20:.0f} MiB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
