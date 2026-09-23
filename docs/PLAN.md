# Clipping Funnel — Build Plan

Target: single Apple Silicon machine (M4 Pro, 32 GB unified memory), macOS only. Plan date: 2026-09-20.

**Epistemic labels used throughout**

- **[V]** Verified this session against the primary source (repo, model card, official docs).
- **[R]** Retrieved from memory. Stable and checkable, but not re-checked for this plan.
- **[I]** Inferred from the stated context or from how the pieces fit together.
- **[U]** Unverified. Check before relying on it; each [U] says what would settle it.

---

## 0. Summary

**What gets built.** A local, semi-automated clipping tool with a browser UI.

1. You give it a long-form source and, for campaigns, the brief.
2. Within roughly 15 minutes it presents 10–15 ranked, pre-framed, captioned candidate previews. **Measured after the step-2b restructure: ~8.2–8.9 minutes from a cold start to candidates** on a 2-hour source (§2.6b) — S0 audio 26 s, S2+S3 7 m 28 s, plus an assumed 20–60 s of ranking. That leaves roughly 6 minutes of the 15 for preview rendering, which is unmeasured. [V for everything up to ranking]
3. You trim, choose a layout, set commentary, and approve.
4. Approved clips get a single full-quality encode plus an export bundle.
5. You post manually and log the post URL. The tool tracks submission status.

**Reframing recommendation.** Do not start with a learned active-speaker detection (ASD) model. Instead:

- Detect faces with **Apple's Vision framework** (face rectangles plus mouth landmarks).
- Get "who speaks when" from **pyannote community-1** speaker diarization.
- Attribute each diarized turn to the face whose mouth moves most during it.
- Plan crops with hysteresis and hard cuts at speaker changes.
- Render split-screen for every two-face clip, as agreed.

Escalate to **Light-ASD** (MIT) only if this heuristic fails a measured threshold on your own labelled clips.

The reason is licensing. The commercially cleaner ASD weights (trained on AVA) generalise markedly worse to in-the-wild video. The weights that generalise well are fine-tuned on a set built partly from TED content under a non-commercial, no-derivatives licence. Details are in §4.

**Build order principle.** The tool becomes usable for real campaigns at step 6, with split-screen for two-shots and face-centred crops otherwise, before speaker attribution exists. Speed to campaign is one of the three scarce things, so an early usable tool beats a complete late one.

**Business gate.** The expensive half of the build (attribution, the eval set, the ASD escalation) is only built if a 4-week trial on the cheap half clears an earnings-per-hour floor (§1.5). Campaign clipping has a failure criterion like every other stage in this document.

**Class (b), third-party clipping, is deferred** until its legal line is decided (§9, item 2). The tool does not ingest it in v1.

---

## 1. Scope

### 1.1 What this funnel is

The funnel serves two uses from one tool:

- **Campaign clipping** on Content Rewards (Whop). Brands supply long-form source and a brief. You post clips to your own accounts, submit the post link, and earn per 1,000 verified views after human approval.
- **General clipping** for reach, split into:
  - **(a)** your own long-form;
  - **(b)** third-party content, carrying commentary. **Deferred; see §1.4.**

  General clips may carry calls to action (CTAs) that redirect viewers to your monetised videos.

### 1.2 Why semi-automated

The binding constraints are:

1. speed to a campaign while its pool is full;
2. brief compliance, since rejected clips are unpaid;
3. reach of the posting accounts.

Clip volume is not one of them. So the tool automates the mechanical middle (transcribe, find moments, cut, reframe, caption, render) and leaves selection, brief judgement, and publishing to you.

### 1.3 Non-goals

- Autonomous posting or batch clip farming.
- Any upload to a YouTube Partner Program channel. Clipping never touches YPP channels.
- Multi-OS or multi-backend support. Apple Silicon only, with no hardware abstraction.
- Modular plugin architecture before the funnel is profitable.
- Crypto and gambling briefs. The brief intake rejects them (§7.2).

### 1.4 Clip classes and routing (enforced by the tool)

| Class | Source | Commentary default | CTA allowed | Posts from |
|---|---|---|---|---|
| `campaign` | Brand-supplied | None | No | Campaign accounts |
| `general-own` (a) | Your long-form | None | Yes | Campaign or general accounts |
| `general-3p` (b) | Third-party | Auto-generate | Yes | **Deferred: not offered at Ingest in v1** |

Every clip carries its class from ingest onward. The compliance gate (S11) blocks export when class, account, CTA, or commentary rules are violated.

**Why (b) is deferred rather than half-built.** Its legal basis (which sources, and where the line is) is undecided. A class that is routed and rendered but not decided is how the riskiest content ends up shipping by default.

Un-deferring requires a written decision: a source allowlist, the line, whether commentary is mandatory, and the account plan. Once decided, (b) gets separate accounts that never post `campaign` clips, so a strike on (b) cannot cost campaign reach. [I] The commentary flow (S10) is built at un-defer time, not before.

### 1.5 Business gate: when to stop

The market is heavily skewed. Most participants earn near nothing. So the tool needs a stop condition as hard as its pipeline gates.

**Time model.** Your cost per campaign is:

`T = setup (find campaign + confirm brief) + review (per source) + posting and submission (per clip) + tracking`

All four are timed automatically (§6.4). Nothing depends on remembering how long things took.

**What an hour has to produce.** At the clustered rate of about $1 per 1,000 verified views, reaching an hourly floor of $H needs **H × 1,000 approved, verified views per hour of your time**. For example, a $20/h floor needs 20,000 paid views per hour spent. [derived] You set H. A $20 placeholder is used below until you do, and it is not derived from anything.

**Kill criteria**, evaluated at the end of the trial: the first 4 weeks or 5 campaigns after build step 6, whichever is later.

| Criterion | Threshold | If it trips |
|---|---|---|
| Net earnings ÷ total logged hours | Below $H | Stop campaign clipping. Do **not** build steps 7–9. |
| Approval rate (approved ÷ submitted) | Below 50% after ≥20 submissions | Stop submitting. Diagnose brief-compliance before any more campaigns. |
| Per-campaign projected payout at join | Below 1 hour × $H at your recent views-per-clip | Skip the campaign; don't ingest. |
| Review time per source | Above 30 min median | Fix ranking quality before running more campaigns (§6.4). |

The approval and projection thresholds are starting points [I]. Revise them after the trial with real data, not before.

**What survives if the campaign gate trips.** The `general-own` class still serves your own long-form. Steps 1–6 are the whole of what that needs. The gate kills campaign operations and the reframing escalation, not necessarily the tool.

---

## 2. Pipeline

### 2.1 Stages

Each stage checkpoints its output and is cached by content hash (source hash + stage parameters). "Gate" means the stage stops. It does not warn and continue.

**Execution order is `S0 S1 S2 S3 S5 S4 S6 …`** — S4 runs after S5 because diarization is window-only and needs the candidate spans (§2.2 item 1). The ids keep their original meaning; only the order changed.

| # | Stage | Output | Hard gate (stops the run or blocks the artifact) |
|---|---|---|---|
| S0 | **Ingest**: local path, or URL via yt-dlp's **Python API** as **two separate streams** — audio first (S2 starts on it), video concurrently, never muxed (§2.5) | Source record with `audio_path` + `video_path`, hash | No video or audio stream; unreadable container; duration below the floor (default 2 min); download failure |
| S1 | **Brief** (campaign only): paste brief → Haiku extracts config → you confirm in the form | Confirmed campaign config | Unconfirmed brief → S5 will not start. Brief category is crypto/gambling → rejected |
| S2 | **Audio extract**: ffmpeg to 16 kHz mono WAV | `audio.wav` | — |
| S3 | **Transcribe + align** (`whispermlx`) | Word-level timestamped transcript, each word carrying an alignment score | **Weakly-aligned words** (interpolated, unaligned, or scoring below the floor) above 12% at `min_score = 0.10` — see §2.7; detected language ≠ expected |
| S4 | **Diarize** (pyannote community-1, exclusive mode) — **runs after S5, on candidate windows only** (each span ± 10 s), one pipeline reused across windows. In-memory waveform, never a path (§2.4) | Speaker-labelled words **within each window**; labels do not carry across windows | **Speakers holding ≥5% of the window's speaking time** differ from the expected count → stop and ask you to confirm |
| S5 | **Rank** (Sonnet) with brief constraints | Candidate list: word-index spans, hook, rationale | Malformed JSON after one retry; fewer than 5 valid candidates after snapping and duration filtering |
| S6 | **Visual analysis** on candidate windows only: shot boundaries (ffmpeg `scdet`), Apple Vision faces and mouth landmarks at ~5 fps, per-shot IoU tracking. **First stage needing pixels, so the first that waits for the video stream.** | Face tracks per shot | Video stream did not finish downloading; a candidate with no face track at all → that candidate gets letterbox layout only |
| S7 | **Speaker attribution + layout planning** | Per-clip crop plans: follow-crop, split-screen | Attribution confidence below threshold on more than X% of clip duration → **follow-crop blocked for that clip** (split or letterbox only) |
| S8 | **Proxy render**: 540×960, `h264_videotoolbox`, captions burned, all available layouts | Preview files | — |
| S9 | **Review** (you) | Approved clips: in/out, layout, hook text, commentary choice | Human gate by definition |
| S10 | **Commentary resolve**: Write / Auto-generate (Haiku) / None | Final overlay text | Auto-generated text not yet accepted by you → no final render |
| S11 | **Compliance check** against the confirmed brief and class rules | Pass/fail per clip | Any fail blocks the clip: duration out of range, missing required tags or disclosure, disallowed platform, CTA on campaign clip, clip routed to an account not allowed for its class |
| S12 | **Final render**: one ffmpeg pass from the original source | 1080×1920 MP4 | Output duration differs from plan by more than 0.1 s; wrong resolution; no audio stream |
| S13 | **Export bundle** per clip per platform | Folder: MP4, `caption.txt`, `checklist.md` | — |
| S14 | **Post + track** (manual): paste post URL, tick the disclosure checkbox | Tracking row | Campaign clip logged without the paid-promotion checkbox ticked → cannot be marked submitted |

### 2.2 Orderings that are load-bearing

1. **S2–S3 start the moment the *audio* lands, in parallel with S1. S4 runs after S5.** [V, revised after step 2]

   Two changes, both forced by measurement.

   **Audio arrives before video.** S0 downloads the two streams separately and
   never muxes them: audio is small and lands first, S2 starts on it
   immediately, and the video downloads concurrently for S6. Waiting for a
   merged file would block transcription behind a gigabyte of pixels that
   nothing before S6 reads. S12 takes the two as separate ffmpeg inputs, which
   costs nothing — it is already a `filter_complex` pass over both.

   **Diarization moved behind ranking.** It was S2→S4 in parallel with S1, with
   S5 waiting on both. Step 2 measured full-source diarization at **9 m 04 s
   for a 2-hour source — 54% of the whole S2–S4 budget** (§2.6), to label audio
   that ranking mostly discards. It now runs only on the candidate windows S5
   selects, each padded by 10 s, which is ~15 minutes of audio instead of 118.

   **The cost, stated plainly: S5 ranks an *unlabelled* transcript.** §5.1's
   input is no longer speaker-labelled, so the ranker cannot see who said what
   and loses "who delivered the punchline" as a signal. §8 anticipated exactly
   this trade and judged it acceptable; step 4 should check that judgement
   against real candidates, which is what the `--full-diarization` flag exists
   for — it keeps whole-source diarization available for that comparison, and
   for anything needing speaker identity across the source.

   **Second cost: labels are per-window only.** pyannote numbers speakers
   independently per call, so `SPEAKER_00` in one window is not `SPEAKER_00` in
   another. Within a clip that is all S7 needs, since it maps labels to faces
   inside that clip. Nothing may assume cross-window identity.

2. **Ranking (S5) runs before the expensive per-window analysis — now both S4 and S6.** The original "light" decision, which step 2 extended to diarization:
   - Face detection and landmarks on a full 2-hour source at 5 fps is about 36,000 frames. On 15 candidates × ~60 s it is about 4,500 frames.
   - Diarization on a full 2-hour source is 9 m 04 s measured. On the same candidate windows it is roughly 1.2 min at the measured 13× realtime.

   The cost is that ranking sees neither visual cues nor speaker labels. For podcast material the transcript carries most of the signal, so this is accepted — but it is now a larger bet than when only vision was deferred, and step 4 should test it. [V for the timings, I for the judgement]

3. **The LLM returns word indices, not timestamps.** Boundaries are then snapped to sentence starts and ends using punctuation in the transcript. This makes mid-word cuts impossible and removes the risk of hallucinated timestamps. Timestamps are looked up from the alignment data, never generated. [I]

4. **Attribution is per shot (S6→S7).** A shot boundary resets face tracks. The speaker-to-face mapping is re-established per shot. This handles multicam cuts without special-casing them. [I]

5. **Commentary (S10) and compliance (S11) come before the final render (S12), and S12 is the only full-quality encode.** It reads the original source once:
   - trim;
   - crop or stack;
   - scale;
   - burn captions, hook, and commentary;
   - loudness-normalise;
   - encode.

   This fixes the fork's double re-encode, which went libx264 and then an OpenCV `mp4v` writer.

### 2.3 Encoding choices

- **Previews:** `h264_videotoolbox` for speed. Quality doesn't matter at 540p.
- **Final:** `libx264` at CRF ~18. A 60-second clip encodes in seconds on an M4 Pro. Platforms re-encode on upload, so start from high quality. [R]
- **Audio:** `loudnorm` in the same pass. Podcast source levels vary. A single-pass target around −14 LUFS is conventional for short-form. [R]
- **Captions:** ASS subtitles burned with libass.
  **Homebrew's default `ffmpeg` formula does NOT have libass. [V]** As of
  Homebrew 7.0 it is a slim build (11 dependencies, no libass, freetype or
  fontconfig), so the `ass`, `subtitles` **and** `drawtext` filters are all
  absent — that breaks burned captions *and* the hook/commentary overlays.
  Use `ffmpeg-full` (47 dependencies, includes libass/freetype/fontconfig/
  harfbuzz). It is **keg-only**, so it is never on PATH.
  Do not mutate PATH. The pipeline addresses the binaries by absolute path
  through two settings read from `.env`:

  ```
  MACLIPS_FFMPEG   default /opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
  MACLIPS_FFPROBE  default /opt/homebrew/opt/ffmpeg-full/bin/ffprobe
  ```

  The startup gate probes these *same* binaries, so the gate cannot pass
  against a different build than the pipeline runs. Verified present on
  ffmpeg-full 9.0.2: `ass`, `subtitles`, `drawtext`, `scdet`, `crop`, `scale`,
  `loudnorm`, `concat`, `hstack`, `vstack`, `sendcmd`, `libx264`,
  `h264_videotoolbox`. [V]

  Note: installing `ffmpeg-full` upgrades x265, which breaks an older slim
  `ffmpeg` still in the Cellar (it links `libx265.216`; x265 4.3 ships `.217`).
  Harmless here because nothing resolves ffmpeg via PATH, but `brew upgrade
  ffmpeg` or `brew uninstall ffmpeg` keeps the rest of the system working. [V]

  **Do not `brew pin ffmpeg-full`.** Pinning is the obvious reflex here and it
  does not buy what it looks like it buys. `brew pin` holds the *formula's own*
  version; it does nothing about its dependencies. x265, libass, freetype and
  the rest keep upgrading underneath a pinned build, and when a shared
  library's soname bumps, the pinned binary is left linking a dylib that no
  longer exists — exactly the failure the unpinned slim `ffmpeg` just hit
  (`libx265.216` → `.217`). Pinning makes that *more* likely, not less,
  because the pinned build drifts further from its dependencies over time
  while an unpinned one gets relinked on upgrade.

  **The guard is the startup gate, not a pin.** The gate executes the
  configured binary and reads back its real filter and encoder lists, so a
  broken library link surfaces as a failed `check` with the binary's path in
  the message — before any stage does work. A pin would give a false sense of
  stability and still fail at the same point. If a Homebrew upgrade ever does
  break the build, the gate says so and the fix is `brew reinstall
  ffmpeg-full` (relink against current dependencies) or pointing
  `MACLIPS_FFMPEG` at a working build. [I]

### 2.4 The torch / torchcodec version window (S4) [V]

**This is a one-version window, and it is load-bearing. Verified in build
step 1 by installing the stack and importing it.**

`whispermlx` 3.13.1 requires `torch~=2.8.0` — that is torch 2.8.x only, while
current torch is 2.14. `pyannote.audio` 4.0.7 in turn requires
`torchcodec>=0.7.0`. **`torchcodec` declares no torch dependency at all**, so a
resolver picks the newest (0.16.0) and the mismatch is invisible at resolve
time and fatal at import time:

```
OSError: Symbol not found: _torch_call_dispatcher
  Expected in: torch/lib/libtorch_cpu.dylib
```

Per torchcodec's own compatibility table, exactly one release satisfies both
pyannote's `>=0.7.0` floor and torch 2.8.x:

| torchcodec | requires torch | |
|---|---|---|
| 0.6 | 2.8 | below pyannote's floor |
| **0.7** | **2.8** | **the only viable version** |
| 0.8 – 0.9 | 2.9 | too new for whispermlx |
| 0.10 | 2.10 | " |
| 0.11+ | ≥2.11 | " |

So `torchcodec==0.7.0` is pinned explicitly. **Any dependency upgrade must
re-verify that `import torchcodec` actually loads** — `uv lock` succeeding
proves nothing here, because the incompatibility is not expressible in
metadata. Treat `uv run python -c "import torchcodec"` as part of the upgrade
checklist.

**Consequence for S4: diarization must receive an in-memory waveform, never a
file path.**

torchcodec 0.7 links against FFmpeg 4–7 (`libavutil.56`–`59`). This machine
runs ffmpeg 8/9 (`libavutil.60`+), so torchcodec's decode path cannot load its
dylib *at all* here. What makes this survivable: `import pyannote.audio`,
`Pipeline` and `whispermlx` all import fine regardless — torchcodec is only
touched when pyannote is asked to **decode a media file itself**. Therefore:

- S2 already produces a 16 kHz mono WAV. Load it and hand pyannote a tensor:
  `pipeline({"waveform": waveform, "sample_rate": 16000})`.
- Never pass a path into the diarization pipeline.

**Step 2 action item — SETTLED [V]. `DiarizationPipeline` is safe; no bypass
needed.** Its `__call__` does:

```python
if isinstance(audio, str):
    audio = load_audio(audio)          # its own ffmpeg decode, not torchcodec
audio_data = {"waveform": torch.from_numpy(audio[None, :]), "sample_rate": SAMPLE_RATE}
output = self.model(audio_data, ...)   # always the in-memory dict
```

It never hands pyannote a path — even a string path is decoded first and
wrapped. pyannote itself emits the matching advice when torchcodec will not
load ("use audio preloaded in-memory as a `{'waveform': ..., 'sample_rate': ...}`
dictionary"), which is exactly this design. We pass the array S2 produced, so
the string branch is never taken either.

**A different trap in the same library, found in step 2 [V]:
`whispermlx.load_audio` hardcodes a bare `"ffmpeg"` PATH lookup.** On this
machine that resolves to the broken slim build (§2.3) and fails inside a
library call. It is never used: `maclips.audio.load_wav` reads the S2 WAV with
the standard library instead. That is possible precisely because S2 already
emits 16 kHz mono 16-bit PCM — the format both whispermlx and pyannote want —
so no decoder is needed at load time and the audio is decoded exactly once per
run. One array is then shared by S3 and S4 through `RunContext.shared`, which
is deliberately not checkpointed (a 2-hour waveform is ~450 MB of float32).

Fallback if a decode path is ever genuinely needed: `brew install ffmpeg@7`
(keg-only) supplies `libavutil.59` without disturbing the ffmpeg-full build
used for encoding. Not needed under the in-memory-waveform design.

### 2.5 URL ingest (S0) [V]

`uv run maclips ingest <path-or-url>` takes either. URLs go through **yt-dlp's
Python API**, not a subprocess of a PATH binary — the PATH `ffmpeg` here is a
broken slim build (§2.3), and yt-dlp must be handed `ffmpeg_location` pointing
at the configured binary so its merge step uses the same ffmpeg as every other
stage.

**Dependency.** `yt-dlp[default]`, not plain `yt-dlp`. The extra pulls in
`yt_dlp_ejs`, which drives YouTube's JavaScript player challenges; those must
be *executed*, so a JS runtime is required. Deno is that runtime. Two startup
gates cover this: `deno` on PATH (version reported) and `yt_dlp_ejs`
importable. Without them extraction either fails or silently degrades to a
reduced format list.

**Format selection and why height is the number that matters.** A 9:16 crop is
taken from the source's full *height* (§4.4), so a source of height H yields a
crop H×9/16 wide. A 1080×1920 final therefore needs **H ≥ 1920** to avoid
upscaling — not "1080p". The cap is a run argument (`--max-height`, default
1440) because the right value depends on the source's aspect ratio:

| source aspect | "1080p" means | 9:16 crop width | upscale to 1080 wide? |
|---|---|---|---|
| 16:9 | 1920×1080 | 608 px | yes, 1.78× |
| 16:9 | 2560×1440 | 810 px | yes, 1.33× |
| 16:9 | 3840×2160 | 1215 px | no |
| **2:1** | 1920×960 | 540 px | yes, 2.0× |
| **2:1** | 2560×1280 | 720 px | yes, 1.5× |
| **2:1** | 3840×1920 | 1080 px | no |

Cinematic 2:1 is common in this material and is *worse* than 16:9 at the same
nominal label, because the frame is shorter. Check with `maclips probe <url>`,
which prints each resolution with its aspect and the crop width it would give,
before committing bandwidth.

**Split streams, never muxed [V].** Audio and video are downloaded as two
separate files: `<id>.audio.<ext>` and `<id>.video.<ext>`. Audio is fetched
first and an `on_audio_ready` callback fires the moment it lands, so S2 starts
on it while the video is still arriving on a background thread. Merging them
would block transcription behind a gigabyte of pixels that nothing before S6
reads.

Measured on the benchmark source: **time to S2 start fell from ~4 m 40 s
(merged download) to 26.3 s** — the audio stream is 109 MiB against 1,150 MiB
merged. S12 takes the two as separate ffmpeg inputs, which costs nothing
because it is already a `filter_complex` pass over both.

`await_video()` blocks until the video lands and is called only by S6, the
first stage that needs pixels. S2-S5 must never call it.

**Caching.** The YouTube video id is the key and names both files, so
re-ingesting the same URL is a cache hit and no bytes are fetched. **Both**
streams must be present to count as a hit — audio alone would let S6 start on a
video that never arrived. Title, channel, duration, upload date and the origin
URL go into the source record.

**Gate.** A download failure stops the run and the message suggests
`uv lock --upgrade-package yt-dlp` — YouTube breakage is almost always a stale
extractor. It is never upgraded automatically: moving a pinned dependency
mid-run would invalidate the lockfile everything else is pinned against.

### 2.6 Step 2 measurements [V]

Measured on the M4 Pro against a **1h57m46s (7,066 s)** two-person interview
(2560x1280, 2.00:1). One Claude Code session open; no other heavy process.
**No swap growth in any stage** — memory pressure stayed `normal` throughout,
so every timing below reflects compute, not paging.

| stage | wall | vs realtime | peak RSS |
|---|---|---|---|
| S0 ingest (cache hit) | 1.6 s | — | 277 MB |
| S2 extract -> 16 kHz mono WAV | 6.5 s | 1090x | 277 MB |
| S3 transcribe (Whisper large-v3-turbo, MLX) | 335.5 s | 21x | 3.1 GB |
| S3 align (wav2vec2, torch/MPS) | 123.5 s | 57x | 6.8 GB |
| **S4 diarize (pyannote community-1, MPS)** | **543.7 s** | **13x** | **8.2 GB** |
| **S2-S4 total** | **16 m 49 s** | **7.0x** | 8.2 GB peak |

A cold S0 download of this source took ~4 m 40 s for 1,150 MiB.

**The §8 ten-minute target is missed. [V]** S2-S4 cost **16 m 49 s** for a
2-hour source — 68% over. Diarization alone is **9 m 04 s**, against the
"flag it above ~5 min per 2 h" threshold: it is **1.8x over**, and it is
**54% of the whole S2-S4 budget**. Transcription and alignment together
(7 m 39 s) would fit the target on their own.

This is §10 risk 2 landing, and it makes the §8 fallback live: diarize only the
candidate windows after ranking, rather than the whole source. **Implemented
in step 2b; see §2.6b.** The arithmetic that favours it: 15
candidates x ~60 s is ~15 minutes of audio against 118, so diarization would
drop from ~9 min to roughly 1.2 min at the measured 13x, putting S2-S4 near
9 minutes. The cost is that ranking then works from an unlabelled transcript.

**MPS vs CPU, measured on a 300 s slice [V].** MPS wins both, decisively, and
does **not** silently fall back — a fallback would show near-identical times:

| | MPS | CPU | MPS advantage |
|---|---|---|---|
| align | 4.9 s | 10.7 s | 2.2x |
| diarize | 23.8 s | 362.5 s | **15.2x** |

Diarization on CPU runs at 0.8x realtime — slower than the audio itself, i.e.
~2.5 hours for this source. MPS is not optional here. The slice's diarization
rate (12.6x) matches the full run's (13.0x), so the cost scales linearly and
there is no surprise at length.

**Transcript quality [V].** 278 segments, language auto-detected as `en`,
**19,624 words at 100.00% alignment coverage**. Coverage is still the wrong
gate, but **not for the reason first recorded here** — see the correction in
§2.7. Measuring the score distribution showed **zero interpolated words** on
this source; coverage reached 100% because every word aligned, not because
interpolation papered over failures. What makes coverage useless is different
and worse: it counts a word whose acoustic score is **0.000** as successfully
aligned.

**Speaker count: 3 raw labels, 2 real speakers [V] — resolved in step 2b.**
Measuring each label's share of speaking time settles it:

| label | words | share | seconds |
|---|---|---|---|
| `SPEAKER_02` | 13,680 | **66.56%** | 3,711 |
| `SPEAKER_00` | 5,786 | **32.51%** | 1,812 |
| `SPEAKER_01` | 144 | 0.92% | 52 |

The two substantial labels are the guest and the host. `SPEAKER_01` is 52
seconds of short interjections ("No,", "my own eyes? Probably quite a bit.") —
ordinary over-segmentation of backchannels, not a third person.

**This is why the count gate compares speaking-time share, not raw labels.** At
a 5% floor the source resolves to exactly 2 speakers, matching the interview.
Gating on `len(speakers)` would have failed a correct diarization.

An earlier note here suspected `SPEAKER_02` of being the spurious label. That
was wrong: it is the dominant voice. The suspicion came from positional
spot-sampling, which surfaced it late in the file; per-speaker sampling with
word counts shows it immediately.

**Model weights, cached locally [V].** All inference is on-device; Hugging Face
is only a registry. Whisper large-v3-turbo 1,539 MB and pyannote community-1 +
segmentation-3.0 in `~/.cache/huggingface`; the wav2vec2 aligner is a
torchaudio pipeline and lands in `~/.cache/torch` (360 MB) instead — warm both
for an offline run. `check_access()` short-circuits on a cached copy so the
gate never forces a network round trip.

**The gated-model gate: `model_info()` is not an access check. [V]**

`pyannote/speaker-diarization-community-1` is `gated: auto`, which makes its
*metadata* world-readable while file downloads still require accepted terms.
The first benchmark run proved the difference the expensive way: `model_info()`
returned happily, the gate passed, S0-S3 ran for eight minutes, and the run
then died inside `Pipeline.from_pretrained` with a 403 on `config.yaml`. So
`check_access()` fetches that file — the one pyannote loads first — and the
benchmark runs the check as a pre-flight **before** S0. A missing authorization
now costs a second instead of eight minutes.

### 2.6b Step 2b measurements: the restructure worked [V]

Same source (1h57m46s, 7,066 s), cold S0 cache, warm models, one session.
**No swap growth in any stage.**

| stage | step 2 | step 2b | change |
|---|---|---|---|
| time to S2 start | ~4 m 40 s (merged download) | **26.3 s** (audio only) | **10.6x faster** |
| S2 extract | 6.5 s | 6.8 s | — |
| S3 transcribe | 335.5 s | 320.4 s | — |
| S3 align | 123.5 s | 121.1 s | — |
| S4 diarize | 543.7 s (whole source) | **61.5 s** (12 placeholder windows) | **8.8x faster** |
| **S2-S4 total** | **16 m 49 s** | **8 m 30 s** | **under the 10-minute target** |

**Time to candidates, which is what the operator actually waits for: ~8.2-8.9
minutes** from a cold start, including the download. (S5 is now implemented
but was a stub at this benchmark; the 20-60 s ranking allowance is an assumption -- a single Sonnet call over ~30k
tokens -- not a measurement.)

**The concurrent video download is entirely hidden. [V]** The video wait
measured **0.0 s**: all 1,040 MiB of video had already arrived while S2 and S3
were running. The split costs nothing and buys the 26-second start.

**Placeholder-window diarization covers 13.4% of the source** -- 12
evenly-spaced windows of 60 s plus clipped 10 s padding each side total 950 s
against 7,066 s. These were generated by `spread_spans`, not selected by S5.
**Correction verified 2026-09-23 [V]:** 950 / 61.5 = **15.45x realtime on
the processed audio**. The old 114.9x used the full source duration divided
by window wall time; it is not the processing rate. The 8.8x wall-time
saving versus the full pass is still valid for these placeholder spans.
Real candidate-window time remains unmeasured; do not infer a clustering
speedup from the old denominator.

**Windows legitimately contain one speaker, and this is why an exact speaker
count must never be forced. [V]** Of the first four windows measured:

| window | span | significant speakers | shares |
|---|---|---|---|
| 0 | 0-60 s | 2 | `SPEAKER_00` 40%, `SPEAKER_01` 60% |
| 1 | 584-644 s | **1** | `SPEAKER_00` 100% |
| 2 | 1168-1228 s | **1** | `SPEAKER_00` 100% |
| 3 | 1751-1811 s | 2 | `SPEAKER_00` 74%, `SPEAKER_01` 26% |

A 60-second stretch of one person talking is completely ordinary in an
interview. Passing an exact count of 2 would force pyannote to split that
single voice in two and invent a second speaker. The expected count is
therefore passed as `max_speakers` -- a ceiling -- and `num_speakers` is not an
accepted argument anywhere in the code.

**Alignment quality against the chosen gate:** 19,621 words, **3.59% weak** at
`min_score = 0.10`, against a 12% ceiling (§2.7). Coverage still reads 100.00%.

### 2.6c Second source, session 2026-09-23c: stopped at S0 [V]

Source: Double Coverage Podcast, `BcrjhdSUv4Y` ("Brez Scales On How He Fixed
His Hairline…"). Uploaded 2026-09-05, 33 m 36 s, English (`en-US`). The best
available resolution is 1920×1080 16:9, so the 9:16 crop is 607 px wide, a
1.78× upscale to 1080 (§2.5). It was the newest qualifying upload. The two
newer uploads were skipped as under 30 minutes: `XHpWQ14EQac` (29 m 08 s) and
`CUXbgqvVOrU` (27 m 09 s). It ran as `--clip-class general-own
--language en`. That class is an experiment label only: the Reach brief is
unconfirmed, and S5 does not use an unconfirmed brief. Nothing is exported
or posted.

**The run stopped at the S0 download-failure gate** after 7.4 s. The audio
stream returned "HTTP Error 403: Forbidden", no stage from S2 on ran, and
there was no API spend. **The failure did not come back:** three later
verbose audio downloads of the same video, one partial and two full,
succeeded with the same pinned yt-dlp 2026.08.19, yt-dlp-ejs 0.8.0 and Deno
2.9.7. yt-dlp logs a GVS PO-token binding experiment and SABR-forced web
formats for this video. [I] The likely cause is an intermittent 403 on a
format URL from a client that needs a PO token, not a stale extractor.
Nothing was upgraded and the gate was not loosened.

**The approved retry downloaded the audio and then stopped at the S0
video-stream gate [V].** The retry ran through the committed CLI code with
the stage list cut before S5 (see below). The audio (32.6 MB) landed after
7.4 s, so the first attempt's 403 was intermittent. S0 then stopped with
"source has no video stream". **The cause is a bug in the committed code,
not in the source.** For a URL, `cli._cmd_run` sets `ctx.source =
record.audio_source`, which is the audio-only `.m4a`. `s0_ingest` ffprobes
`ctx.source` and requires a video stream. So on the split-stream path
(§2.5), **`maclips run <url>` stops at S0 for every URL.** The §2.6b
benchmark never hit this because `scripts/benchmark_step2b.py` calls
`download_split` and the S2/S3 functions directly, bypassing `s0_ingest`. No
test covers S0 with a split URL source. The process exit also stopped the
background video download: `BcrjhdSUv4Y.video.mp4.part` (16.6 MB) remains in
`work/sources/`. Not fixed here: the fix is a separate decision. S2/S3
timings, the weak-word fraction and the S3 gate outcome for this source are
still unmeasured.

### 2.7 The S3 alignment gate, set from data [V]

**Correction to §2.6.** The 100.00% coverage figure was first explained here as
interpolation filling the gaps. Measuring the distribution disproved that:
**0 of 19,625 words were interpolated**, and 0 were unaligned. Every word
aligned acoustically. The mechanism described was real — whispermlx sets
`start`/`end`/`score` only for characters it can align, then interpolates
missing `start`/`end` across the sentence **without ever backfilling `score`**,
so a word with timings but no score has guessed timings — it simply did not
occur on this source.

**What actually makes coverage useless** is that it treats any word carrying
timings as aligned, including words whose acoustic confidence is **0.000**.
On the benchmark source:

| percentile | score |
|---|---|
| p0 (minimum) | **0.000** |
| p1 | 0.001 |
| p5 | 0.200 |
| p10 | 0.419 |
| p25 | 0.708 |
| p50 (median) | 0.828 |
| p90 | 0.958 |
| mean | 0.755 |

Coverage reports 100% across all of it. The bottom 1% of words are acoustically
unmatched, and coverage cannot see them.

**Weak-word fraction at candidate thresholds** (weak = interpolated, unaligned,
or scoring below the threshold):

| `min_score` | weak fraction |
|---|---|
| 0.05 | 2.90% |
| **0.10** | **3.59%** |
| 0.20 | 5.00% |
| 0.30 | 6.83% |
| 0.50 | 12.70% |

**Chosen: `min_score = 0.10`, `max_weak_word_fraction = 0.12`.**

The clean source sits at 3.59%, giving ~3.3x headroom. The looseness is
deliberate. A tighter pairing — say 0.30 and 5% — would *fail this correct
transcript* at 6.83%, and a gate that stops good audio is worse than no gate.
This one exists to catch alignment **collapse**: wrong language, a music bed,
the wrong audio track, where the weak fraction runs to tens of percent.

**A caveat that shapes the threshold [V].** Low scores are strongly biased by
word length, because `score` is a *per-character mean*. Of the 704 words below
0.10, the length distribution is 150 one-character, 211 two-character, 183
three-character — mean **2.7 characters against 4.2 overall** — and they are
overwhelmingly unstressed function words ("I", "a", "as", "is", "and", "it").
A one-character word's score is a single character's confidence, which is
inherently noisy. So the low-score tail measures English function-word density
as much as alignment quality, and a threshold tight enough to "clean it up"
would mostly be penalising natural speech.

This matters less than it sounds for clip quality: §2.2 item 3 snaps boundaries
to **sentence** starts and ends, so a poorly-scored "I" mid-sentence never
becomes a cut point. If a future source shows weak words clustering at sentence
boundaries specifically, that is the signal worth gating on, and it would need
a different measure than this one.

---

## 3. What's reused vs built

| Component | Origin | Notes |
|---|---|---|
| Transcription + wav2vec2 alignment | **Integrated, not reused** — see note below | `whispermlx` is a PyPI package (3.13.1). There was no pre-existing stage of yours in this repo to wire up. |
| Caption data | **Reused**: alignment output | Only the renderer is new (ASS word-highlight instead of SRT). |
| Orchestrator (checkpointing, hash cache, hard gates) | **Built in step 1** — see note below | Written from this section's stage list; there was no existing orchestrator in this repo. |
| Pipeline entry, `get_highlights(..., llm_fn=)` seam, ranking prompt, subclip logic | **From fork** (MIT, keep the upstream copyright notice) | The prompt is rewritten substantially (§5). |
| MuAPI `mode="api"` path | **Deleted** | Local only. |
| `faster-whisper`, Haar cascade, OpenCV video writer, two-pass encode | **Deleted** | — |
| Diarization + word merge | **Built** | pyannote community-1. |
| Brief extraction, schema, confirm form | **Built** | — |
| Vision face tracking (pyobjc → Apple Vision) | **Planned; stub only** | Step 6. |
| Speaker attribution, layout planner | **Planned; stub only** | Step 7, the core of §4. |
| One-pass final renderer | **Planned; stub only** | Step 5, ffmpeg `filter_complex`. |
| Web UI (Ingest / Review / Posted) | **Planned** | Step 5; brief confirmation currently uses the CLI. |
| SQLite tracking, export bundles | **Schema built; operational tracking/export pending** | Steps 5 and 8. |

**Correction after build step 1 [V].** This table originally listed the
orchestrator and the `whispermlx` transcription stage as *"Reused: your
existing…"*. Neither existed in this repository, and no other location was
identified. What actually happened:

- **The orchestrator was written from scratch in step 1**, to the §2.1 stage
  list — content-hash caching, per-stage checkpointing, hard gates. It is not
  a port of an earlier pattern.
- **`whispermlx` is a third-party PyPI package**, not your own stage. So
  **build step 2 is an integration task, not a wiring-up task**, and should be
  estimated as such: install, feed it audio, handle its output shape, and deal
  with the dependency constraint recorded under S4 below.

**Located after the fact [V]:** the transcription/alignment stage this table
meant is `~/coding/AMC/manim-funnel/src/align/` — `whisper.py` (mlx-whisper
transcription), `forced.py` (wav2vec2 forced alignment) and `qa.py`.
**Decision: copy the knowledge, do not share the module.** The contracts are
opposite. manim-funnel force-aligns a *known* script to TTS audio and its
aligner raises when the word count disagrees with `Section.words` — that check
is what validates TTS fidelity. maclips must *discover* unknown speech, so
there is no prior word list to check against and `whisper.py` discards the
per-word timings by joining segments to one string. `forced.py` is also bound
to that project's `schema.Section` / `timing.WordTiming`, so sharing it means
extracting those too, then adding a mode flag to serve both contracts — the
"abstraction for later" the Principles rule out. Roughly 15 of its 246 lines
carry over, as facts rather than code:

- `whispermlx.load_audio(path)` returns an array; pass the array, never the
  path. **manim-funnel independently hit the §2.4 torchcodec breakage and
  settled on exactly this workaround** — good corroboration.
- API surface: `load_model`, `load_align_model(language_code=, device=)`,
  `align(transcript=, model=, align_model_metadata=, audio=, device=)`, then
  `result["word_segments"]`.
- `whispermlx.align` tokenizes with a bare `text.split(" ")` — no punctuation
  handling of its own. Bears directly on §2.2 item 3, which snaps spans to
  sentence boundaries *using* punctuation.
- whispermlx omits `start`/`end` for a word it cannot place, even after
  interpolating across the sentence. That behaviour is what makes S3's 97%
  aligned-word-coverage gate measurable.
- Transcription runs on MLX; alignment stays on torch/MPS.
- Default model `mlx-community/whisper-large-v3-turbo`.

Note also that manim-funnel pins `torchcodec 0.16.0` against `torch 2.8.0` —
the broken pair in §2.4. Its call-site workaround masks the breakage rather
than fixing it; `torchcodec==0.7.0` is the fix if that project wants one.

**A note on the fork's value.** After these replacements, what survives from the fork is a skeleton, one seam, and a prompt draft. Expect it to be a small fraction of the final code. That is fine: the fork is a starting scaffold, not a dependency. Don't let "stay close to upstream" constrain design choices. [I]

---

## 4. The reframing problem

### 4.1 Decomposition

Reframing breaks into four independently checkable pieces:

1. **Where are the faces?** Detection plus tracking per shot. Checkable by overlaying boxes on a few frames.
2. **Who is speaking?** Attribution. Checkable against hand labels (§4.6).
3. **Where should the crop be?** Layout planning. Checkable as "does the crop contain the attributed face, with headroom?"
4. **Render.** Checkable by ffprobe plus eyeballing.

Nearly all the risk is in piece 2. Pieces 1, 3 and 4 are routine.

### 4.2 What each source type needs

| Source type | Needs attribution? | Approach |
|---|---|---|
| Multicam, already cut to the speaker | Rarely | Shot detection → one dominant face per shot → centre on it |
| Static wide two-shot | **Yes** | Attribution → follow-crop; split-screen always rendered too |
| Single speaker | No | Centre on the face |
| 3+ person panel | Yes | Follow-crop; split-screen uses the two most-speaking faces |

### 4.3 Options and licences (commercial use)

Code and weights are listed separately.

| Option | Code licence | Weights / data | Commercial verdict |
|---|---|---|---|
| **Apple Vision** (`VNDetectFaceRectanglesRequest`, `VNDetectFaceLandmarksRequest`) | macOS system framework, called via pyobjc | No weights shipped; built into the OS | **Clean.** Use of OS frameworks in your own software is standard under Apple's SDK terms [R]. The landmarks request detects eyes and mouth [V]. |
| **pyannote.audio** + `speaker-diarization-community-1` | MIT [R] | **CC-BY-4.0** [V]; gated Hugging Face download | **Clean with attribution.** |
| **Light-ASD** (CVPR 2023) | **MIT** [V]; weights ship in the repo's `weight/` folder under that licence [V] | Two checkpoints: AVA-trained (default) and TalkSet-finetuned [V] | **AVA weights: usable, with residual provenance risk. TalkSet weights: avoid.** See the notes below. |
| **TalkNet-ASD** | **MIT** [V] | AVA and TalkSet checkpoints, same caveats as Light-ASD | Heavier than Light-ASD with the same licence picture. No reason to prefer it. |
| **S3FD** face detector (used by the TalkNet and Light-ASD demo scripts) | Not checked | Not checked | **[U] — not needed.** Apple Vision replaces it. Do not import the demo pipeline wholesale. |
| **LR-ASD** (IJCV 2025 extension of Light-ASD) | Licence not confirmed from the repo page | Not checked | **[U] — excluded** until a licence file is confirmed. |
| **LoCoNet / LASER-ASD** | The LASER PyPI wrapper states MIT [V] | Underlying weights not checked | **[U] — excluded.** Heavier; no advantage for this use. |
| OpenCV Haar cascade (current fork) | Apache/BSD [R] | — | Clean but inadequate: frontal-only, picks the largest face. |

**Light-ASD's weight provenance.**

- **AVA-trained weights.** AVA-ActiveSpeaker is built from YouTube movies [V]. Annotations are commonly listed as CC BY 4.0, per a third-party dataset listing [R]. The underlying film copyright stays with the owners. Whether trained weights are encumbered by training-data copyright is legally unsettled. This is residual risk, not a known violation. [I]
- **TalkSet-finetuned weights.** TalkSet is derived from VoxCeleb2 and LRS3 [V]. LRS3's licence text is for research purposes. It states that the TED-sourced content must respect TED's terms and the CC BY-NC-ND 4.0 licence [V]. For a commercial pipeline, don't use weights derived from non-commercial, no-derivatives material. **Avoid.**
- **The performance gap.** On the out-of-domain Columbia benchmark, Light-ASD's AVA weights average **81.1% F1** and the TalkSet weights **95.5% F1** [V]. TalkNet's authors say explicitly that AVA-trained models transfer poorly to video outside AVA, which is why they built TalkSet [V].

So the licence-cleaner checkpoint is the weaker one on exactly the in-the-wild footage you'll be clipping. This is the main reason the recommendation below starts with a heuristic.

### 4.4 Recommendation: diarization plus mouth-motion association

This approach uses only clean components (Apple Vision, pyannote community-1, your own code). Podcast conditions suit it: few speakers, long turns, mostly seated, mostly frontal or three-quarter views.

**Mechanics**

1. **Face tracks (S6).**
   - Run Vision face landmarks at ~5 fps within each candidate window.
   - Associate detections across frames by IoU within a shot. Shot boundaries come from ffmpeg `scdet`.
   - Drop tracks shorter than ~1 s.
2. **Mouth signal.** Per face per frame, compute inner-lip vertical opening normalised by face-box height.
3. **Turn attribution.** For each diarized speaker turn inside a shot:
   - Compute each track's mouth-motion energy (standard deviation of the mouth signal over the turn).
   - Attribute the turn to the highest-energy track if it beats the runner-up by a margin.
4. **Shot-level mapping.** Majority-vote a mapping from speaker label to face track per shot, weighted by turn duration. Apply the mapping to every turn in the shot, including short turns where the visual signal is weak.
   - **Confidence** = agreement fraction of the vote.
   - This is what S7's gate checks.
5. **Layout plan: follow-crop.**
   - Segments follow the attributed speaker.
   - Turns under ~1.0 s (backchannels like "yeah" or "right") do not trigger a switch. This is the hysteresis.
   - A switch is a **hard cut** at the turn's first word boundary. No panning.
   - **This is a default under test, not settled.** Hard cuts versus eased pans between speakers is an aesthetic bet. Neither "clips should cut" nor "high-performing clips pan" has been verified here. [U]
     - Step 7 renders a subset of approved clips both ways; you pick blind in Review.
     - If pans win, the planner emits a short eased crop transition (ffmpeg `sendcmd`, ~0.3 s) at switches. Everything else in the static-segment design stays.
     - The 1.0 s hysteresis is a tunable parameter, checked in the same test.
   - Within a segment the crop is **static**: centred on the track's median position, face centre at ~⅓ from the top, 9:16 cut from full source height.
   - If the face drifts outside a dead-zone, the segment splits.
   - Static crops per segment render as ffmpeg `trim`+`crop`+`concat` in one pass. No per-frame crop expressions are needed.
6. **Layout plan: split-screen.**
   - Rendered whenever two tracks exist.
   - Two 1080×960 panels, each a crop around one face.
   - The left-in-source person goes on top, consistently.
   - Captions sit at the seam.
   - Split-screen needs pieces 1 and 3 but **not** piece 2, which is why it is the safe fallback.
7. **Speaker with no face track** (off-screen or looking away). Hold the previous segment. If that runs longer than ~3 s, fall back to split-screen or letterbox for that span.

**Known failure modes, and what catches them**

| Failure | Why | Caught by |
|---|---|---|
| Boom or handheld mic covering the mouth | Landmarks unreliable → weak signal | Low confidence → S7 gate → split-screen only |
| Profile views | Vision's landmark quality on strong yaw is **[U]**; measure it in the eval | Confidence gate |
| Overlapping speech | Exclusive diarization assigns one speaker | Minor; review catches it |
| People moving around (not seated) | Static-crop-per-segment assumption breaks | Dead-zone splits segments; if a clip becomes choppy, you choose split/letterbox in review |
| Diarization over- or under-counting speakers | Model error | S4 gate when you gave an expected count |

**Stated assumption.** Seated, mostly stationary speakers. That matches the dominant campaign type (podcasts and interviews). If you start taking vlog or IRL-stream briefs, the static-segment design needs dynamic crop paths (ffmpeg `sendcmd`), which is a v2 item.

### 4.5 Escalation path, used only if §4.6 fails

If the heuristic misses the threshold on two-shot sources:

1. **First, tune the heuristic.** Adjust the margin, hysteresis, landmark sampling rate, and the mouth metric. This is cheap.
2. **Then try Light-ASD with AVA weights** as a per-track scorer on ambiguous shots only. It has ~1.0M parameters [V], so it's cheap to run.
   - It must run on PyTorch MPS. Whether Conv3d and the GRU run on MPS without CPU fallback is **[U]**. Settle it with a 20-line smoke test.
   - Expect it to underperform its paper numbers out of domain (the 81.1% F1 above).
3. **Last resort: fine-tune Light-ASD on your own labelled footage.** Use only footage you have rights to, such as your own long-form. That produces weights whose provenance you control. It costs more labelling but removes the licence question. Treat this as an open decision (§9).

### 4.6 Evaluation (blocks shipping follow-crop)

- **Labelled set:** ~10 real sources of campaign type, 3 minutes each. Label the active speaker every 0.5 s using a simple labelling mode in the Review player (keys 1/2/3). Expect roughly 1–2 hours of labelling.
- **Metric:** percentage of clip duration where the crop shows the true speaker, plus median switch latency.
- **Ship threshold for follow-crop on two-shots:** ≥95% [I]. This is a starting target, not derived. Tighten or loosen it once you see what "wrong" looks like in approvals.
- **Gate calibration:** set S7's confidence threshold so that clips passing the gate hit the ship threshold. Clips below it get split-screen, which is still approvable.

---

## 5. Highlight detection and ranking

### 5.1 Input

- The full transcript, one line per sentence, prefixed with the first word index. **Not speaker-labelled**: diarization moved behind ranking in step 2b (§2.2 item 1), so speaker labels do not exist yet at S5. Use `--full-diarization` to produce a labelled transcript for the step-4 comparison.
- A 2-hour podcast measured **44,014 tokens** unlabelled and **62,833** speaker-labelled (`count_tokens`, `claude-sonnet-5`, full ranking prompt, §5.2b) [V] — not the 25–35k first estimated. It still fits in one call. No chunking, and no loss of cross-section context.

### 5.2 Model and cost

- **Model: Sonnet (`claude-sonnet-5`).** This deliberately breaks your "ranking → Haiku" tiering rule. This ranking is the quality-critical judgement that decides which moments ever reach approval.
- **Rate: $2 / $10 per million tokens (input/output) [V].** Anthropic's
  pricing page (fetched 2026-09-23) says the introductory price is now the
  standard price and the rise to $3/$15 will not occur (§5.2c).
  `config.TOKEN_RATES_USD_PER_MTOK` carries this one rate; the disputed-rate
  range and `cost_range_usd()` were removed on 2026-09-23.
- **Thinking: adaptive at `low` effort [decision, user].** Omitting `thinking`
  runs Sonnet 5 at its default high effort, which spent the whole
  16,000-token cap on the first live call (§5.2c). `rank_fn` sets
  `reasoning_effort="low"` and `max_tokens=32000`, which caps thinking and
  JSON together. Temperature is omitted.
- **Cost per source: $0.10 unlabelled, measured on 2 calls [V].** 44,014
  input tokens ($0.088) plus about 1,250 output tokens ($0.0125). The output
  count includes the thinking tokens, which are billed as output; low effort
  used only 21-32 of them. Wall time was 14-16 s. The worst case at the
  32,000 cap is $0.41. Both calls hit the survivor gate, so this is the cost
  of a call, not yet of a usable candidate list (§5.2c).
- **Tokenizer.** The current-generation tokenizer uses more tokens per
  character than older estimates assumed: the `chars / 4` estimate was 1.47x
  too low on the benchmark transcript (§5.2b) [V]. Budget from
  `count_tokens`, not character counts.
- **Usage is logged per call, including failed calls [V, tests].**
  `llm.complete` records input, output, reasoning, cache-read and
  cache-creation tokens, `finish_reason` and cost before any raise.

- **Haiku 4.5** ($1 / $5 per million tokens [R]; model id `claude-haiku-4-5`,
  no date suffix) is used for:
  - brief field extraction, with human confirmation;
  - commentary drafts, with human acceptance.

### 5.2b Implementation status and measured transcript size [V/blocked]

Steps 3 and 4 are **implemented**. Step 3 was exercised during capture
work, but not across the full brief set; step 4 has no completed live ranking
run. The original blocker was a missing `ANTHROPIC_API_KEY`; the current
credential status is recorded in §5.2c. The sizes below come from the cached
benchmark transcript, not model usage.

**Transcript size, measured on 19,625 real words (§2.6b's source):**

| form | sentences | characters | first estimate (`chars / 4`) | measured prompt tokens [V] |
|---|---|---|---|---|
| unlabelled (what S5 sees) | 2,095 | 118,410 | ~29.6k | **44,014** |
| speaker-labelled | 2,095 | 143,502 | ~35.9k | **62,833** |

Measured 2026-09-23 with the free `count_tokens` endpoint on `claude-sonnet-5`,
over the full prompt `ranking.rank` builds (119,783 / 144,875 characters
including instructions). The count is identical with and without
`thinking: disabled`. The unlabelled count equals the failed live call's
`prompt_tokens` exactly (§5.2c). The `chars / 4` estimate was **1.47x too low**
(about 2.7 characters per token on this transcript).

**Speaker labels add 21.2% more characters but 42.8% more tokens [V]**
(+18,819 tokens, about +$0.038 input per source at $2/MTok).

**Cost is a single number since 2026-09-23 [V].** The rate is settled at
$2/$10 (§5.2). The range reporting this paragraph described has been removed.

### 5.2c Session 2026-09-23: real briefs and ranking experiment [V]

Source initially inspected: `72f711a`; documentation corrections committed
as `7fff2d1`. The user authorized one subsequent client correction:
`rank_fn()` now explicitly uses **temperature=1**, because installed LiteLLM
rejects Sonnet 5 at the previous default 0.2 before sending a request. All
four conditions use 1. Haiku extraction, ranking prompts, models, filters,
dependencies, and raw briefs were not changed. Credential setup initially
blocked requests; the user's replacement key now works.

**Validation and inputs [V].** Initial `uv run pytest -q`: **169 passed in
5.62 s**; after the temperature fix: **169 passed in 6.01 s**. Environment
check passed (Python 3.12.13, macOS 26.5 arm64, torch 2.8.0/MPS, MLX GPU,
ffmpeg-full, Deno 2.9.7, yt-dlp-ejs 0.8.0, Vision). Source and wheel build
passed. Input: **19,625 words**, **19,610 labelled words**, **2,095 sentences**;
WAV **7,065.809 s**, mono 16 kHz / 16-bit PCM. Transcript SHA-256:
`2baffee949d5ebb3311d87ee13d02dceb9c3e6e598c16f0f1295b2b940b6f479`.
`work/session-20260923/source-receipt.json` records exact code, lockfile,
benchmark and brief hashes; no secrets are included.

**Real-brief extraction [V].** Eight campaign briefs attempted independently
with `claude-haiku-4-5`, `--no-confirm`: **7 saved configs, 1 JSON failure,
0 category rejections**. Lovable failed before the category check; the other
seven passed. `_open-tabs.txt` is a navigation list, not a ninth brief.
Every saved config preserves `raw_brief` exactly and remains unconfirmed.

| Brief | Observed errors in the saved config (full field audit kept locally) |
|---|---|
| Arabic / MW4 | Lost $26,300 pool, 10 s minimum, topic/edit exclusions and permitted voiceover; five alternative disclosure hashtags flattened to one required list; platform-scoped tags lost their scope. Rates conflict between page/doc and cannot fit one scalar. |
| Duetti | Lost $1.25 rate and 10 s minimum; null deadline violates declared string type; required sound has no schema field. |
| eFlow | Lost $2 rate and $5,000 pool; pre-publication approval and bio-link requirements have no dedicated fields. |
| GUNSMXKE | Lost $0.75 rate and explicit role-specific commentary permission; null deadline; logo/bio requirements unrepresented. |
| Jackelyne | Lost $1.50 rate, 10 s minimum, required “Jacky” hook mention and edit restrictions; commentary false and overlays null despite permission; support contacts misclassified as required tags. |
| Lovable | Invalid JSON: unterminated string, line 132 column 14 (char 4228). No config. **Diagnosed in §5.2d: truncation** (`finish_reason=length` at 2,048 tokens, 3/3), caused by off-schema output. |
| Reach / Double Coverage | Lost $1.50 rate, 30 min submission window and topic/edit restrictions. Brief requires logo/watermark while also forbidding all logos; conflict needs human judgment. |
| Sound Network / Riley Green | Lost $2,000 pool, 10 s minimum, overlay permission and topic/edit restrictions; null deadline; platform-specific rates cannot fit the scalar. |

Across **all 7 configs**, `rate_per_1k` is null; all **4 successfully parsed
briefs with an explicit 10-second minimum** lost it. `missing` is incomplete
and often contains invented field names. Long documents expose more conditional
requirements, but even short structured briefs lose essentials. **This is not
evidence that length alone explains quality.** The prompt/schema connection
needs attention before extrapolating to step 5. **Diagnosed and fixed in §5.2d.**

`brief.extract()` declares `EXTRACTION_SCHEMA` but never sends it or validates
against it; unknown fields are silently discarded by `from_dict()`. Therefore
these are end-to-end config failures, not proof that Haiku itself omitted every
lost value. The schema also omits `pool_used_pct_at_join` and cannot adequately
represent platform-specific rates, alternatives, source/audio allowlists,
account, audience, payout or retention requirements. Capture-time budget usage
is not necessarily usage at join and was not silently substituted.

Full unchanged configs, failures and per-field findings are local:
`work/session-20260923/extractions/` and
`work/session-20260923/extraction-findings.md`. Raw third-party documents and
configs embedding them remain gitignored rather than being published.

**Untrusted-content check [V/limited].** **0 of 8 files** contains the capture
tool's `UNTRUSTED THIRD-PARTY CONTENT` marker. The prompt tells the model to
treat all embedded instructions as data. No saved field appears attributable
to a model-directed instruction instead of requirements; the invented support
tags are a semantic misclassification. Reach's editorial “prefer $5,000” note
agrees with independent budget figures, so extracting that amount does not
prove instruction-following. The injection fixture has since been run against
the real model: **[V] for that fixture**, see §5.2d.

**Ranking experiment: failed on its first call [V].** Session-only
instrumentation wrapped the existing experiment (it enforces the same
five-survivor gate as S5, checks audio and model access, and records raw API
usage, because the client logs none of it on failure). Run A1 (unlabelled)
made **one** call: `prompt_tokens=44,014`, `completion_tokens=16,000`, all of
them `reasoning_tokens` and **0 text tokens**. It took 181 s, and the model
returned `claude-sonnet-5`. `llm.complete` raised "returned an empty
completion". Cost: **$0.25** (44,014 × $2 + 16,000 × $10 per MTok). No
candidates, no blind sheet and no real-window diarization timing exist;
`work/experiment/` is empty. A2, B1 and B2 never ran.

**Diagnosis (session 2026-09-23b, no further live calls) [V]:**

- *Thinking was on, and we never asked for it.* The request LiteLLM 1.102.0
  sends for `rank_fn` was captured offline (HTTP post patched, fake key, no
  network): `{"model": "claude-sonnet-5", "messages": [...], "temperature": 1,
  "max_tokens": 16000}`. It has no `thinking`, no `output_config`, no tool
  and no beta header. The Anthropic docs say: "On Claude Sonnet 5, where
  thinking is on by default", with effort `high` by default and `display`
  `"omitted"`. They also say "`max_tokens` is a hard cap on total output for
  the request, thinking and response text combined". The model spent the
  whole 16,000-token cap thinking and never reached the JSON.
- *`response_format={"type": "json_object"}` never reaches the API.* With no
  schema attached, LiteLLM's Anthropic transformation maps it to nothing
  (`llms/anthropic/chat/transformation.py` 1312-1326, 1512-1533). JSON-only is
  enforced by the prompt alone. This is the same class of bug as §5.2d.
- *The error message hid the truncation.* `llm.complete` checks for empty
  text before it checks `finish_reason == "length"`, so a max-tokens stop
  reads as "empty completion". The error is raised outside `ranking.rank`'s
  parse `try`, so the retry did not run: one call, not two.
- *`temperature=1` is harmless but unnecessary.* The docs say Sonnet 5
  rejects *non-default* sampling values, and LiteLLM passes 1 through only
  because it equals the default. Omitting temperature sends nothing.
- *LiteLLM trap [V]:* `reasoning_effort="none"` sends no thinking field at
  all, so thinking stays on. Only `thinking={"type": "disabled"}` reaches
  the API as "disabled". `reasoning_effort="low"` or `"medium"` maps to
  adaptive thinking plus `output_config.effort`.
- *Usage logging gap [V].* `llm.complete` appends usage only after both
  raise checks pass. It records no reasoning tokens, no cache tokens and no
  finish reason, so a failed call leaves no record.

**Pricing settled [V].** Anthropic's pricing page (fetched 2026-09-23) says:
"The $2/$10 … pricing for Claude Sonnet 5, announced at launch as
introductory pricing through August 31, 2026, is now the standard price. The
previously scheduled increase to $3/$15 … will not occur." The dispute in
§5.2 and the $3/$15 entry in `config.DISPUTED_RATES_USD_PER_MTOK` were stale.
Both were corrected in session 2026-09-23c.

**Open, for the user [decision]:** whether ranking should think at all. The
options are: thinking disabled (≈2k output, ≈$0.11/source unlabelled, tens of
seconds [I]); adaptive at `low`/`medium` effort with a larger cap (thinking
length [U]); or default `high` with a 64k cap and streaming (≥16k thinking
observed, up to ≈$0.73/source). No fix is applied yet. **Decided and applied (session 2026-09-23c) [V]:**
adaptive thinking at `low` effort. The body installed LiteLLM 1.102.0 now
sends for `rank_fn`, captured offline the same way: `{"model":
"claude-sonnet-5", "max_tokens": 32000, "thinking": {"type": "adaptive",
"display": "summarized"}, "output_config": {"effort": "low"}}`. There is no
temperature and no beta header. `display: "summarized"` is LiteLLM's own
addition and changes visibility only, not billing. `llm.complete` now reports
`finish_reason == "length"` as truncation before the empty-text check and
records usage before any raise. `tests/test_llm.py` pins each of these.

The built-in comparison verdict uses a fixed 15-percentage-point heuristic,
not a significance test. Report within-condition noise before cross-condition
overlap and do not infer a quality winner without blind ratings.

Corrections made earlier in this session: the S3 threshold is 12%, not 5%;
window-only S4 is already implemented; 61.5 s used placeholders; 114.9x used
the wrong realtime denominator; 21.2% is character overhead; §3's future
components are not built; and linked-doc capture is already built, not v2.

**Session 2026-09-23c: first working ranking calls, then the survivor gate
[V].** With low-effort thinking (§5.2), both calls finished normally, with
valid JSON on the first attempt:

| run | input | output | reasoning | finish | wall | cost | survivors |
|---|---|---|---|---|---|---|---|
| A1 unlabelled | 44,014 | 1,262 | 21 | stop | 16.1 s | $0.1006 | **4 / 12** |
| A2 unlabelled | 44,014 | 1,248 | 32 | stop | 13.8 s | $0.1005 | **3 / 12** |

The measured cost per source is **$0.10** unlabelled, and ranking takes
**14-16 s**, under the 20-60 s allowance in §2.6b. Whether `reasoning` is the
count the API reported or LiteLLM's estimate from the summarized thinking
text is not verified [U].

**Both runs hit the five-survivor gate (§5.4 item 5).** Every drop was a
duration drop (8 and 9), with no overlap and no out-of-range drops. The
model's own spans are too long. Before snapping, 6 of A1's 8 dropped spans
and 8 of A2's 9 were already over 60 s: 97-427 words and up to 142 s, against
the prompt's "roughly 75-150 words". Outward snapping (§2.2 item 3) pushed
the rest over 60 s: A1 56.7 to 60.9 s and 59.0 to 66.2 s, A2 59.0 to 66.2 s.
[I] With only about 20-30 thinking tokens, the model does not check span
length against the word guidance. The experiment was stopped before B1 and
B2. There are no blind sheet, no overlap figures and no real-window
diarization timing yet. Raw outputs are in `work/session-20260923c/raw/`
(gitignored). Not changed here: the prompt, the snapping direction and the
duration filter. Which of them to change is a decision.

`scripts/ranking_experiment.py` records `insufficient` but does not stop on
it. It continues to the next run, to diarization and to the blind sheet, so a
gated run can still reach the blind sheet. It was stopped by hand this time.

**The CLI has no stop-before-stage option [V].** `maclips run` has `--from`
but no `--to`/`--until`, and `orchestrator.run_pipeline` stops only at a
gate. So an S0-S3-only run (no API call) is impossible from the CLI. The
session 2026-09-23c retry (§2.6c) used a scratch driver that sets
`cli.STAGES` to the committed stages before S5. This is recorded as a
finding only; no option was added.

**Spend: $0.2012 recorded, $0.64 worst case [V/U].** The script was killed
just after the spend guard printed B1's projection, so the B1 request may
already have been sent. If it was, it is not in the ledger. It would cost
about $0.13 if it matched A1 and A2, and at most $0.436 at the 32,000 cap.
That puts the worst-case session total at $0.64, against a $3.00 cap.

### 5.2d Brief extraction fix, 2026-09-23 [V]

**Diagnosis first.** Each of the 8 captured files was checked for the
business fields before anything changed:

| Brief | Rate | Pool total / used | 10 s minimum | Deadline |
|---|---|---|---|---|
| Arabic / MW4 | present (per-platform) | present / present | present, Arabic: “أقل مدة للفيديو 10 ثواني” | absent |
| Duetti | present | present / present | present: “Minimum video length 10s” | absent |
| eFlow | present (per-platform) | present / present | not stated | absent |
| GUNSMXKE | present (per-platform) | present / present | not stated | absent |
| Jackelyne | present (per-platform) | present / present | present: “Minimum length: 10 seconds” | absent |
| Lovable | present (per-platform) | present / present | not stated | absent |
| Reach | present (per-platform) | present / present | not stated | absent |
| Sound Network | present (per-platform) | present / present | present: “Video must be AT LEAST 10 seconds long” | absent |

Every file records “Deadline: not shown on this page”, so deadline is absent
from source, not dropped. **Everything present was lost in extraction; nothing
was lost in capture.** No capture change was made. Caveat: these 8 files are
the manual captures that preceded `capture.py` (they carry hand-labelled
“Rate per 1K views:” headers and no untrusted-content marker). Whether
`capture.py`'s own page-text dump includes the rate/pool widgets has **not
been verified [U]**.

**Cause [V].** The prompt said “matching the schema exactly” but never sent
the schema. Raw Haiku output, recorded before `from_dict()`, used its own keys:
`rate_per_1k_views`, `cpm_usd`, `rates`, `pool_total_usd`,
`minimum_video_duration`, `submission_deadline_minutes`, and it echoed
`raw_brief`. `from_dict()` discarded all of them silently. Only the fields the
prompt named (`platforms_allowed`, tags, hashtags, `missing`) survived.

**Lovable [V]: truncation, not malformed JSON.** Three reruns of the unchanged
prompt all ended with `finish_reason=length` at exactly 2,048 output tokens,
inside an off-schema list of about 26 source URLs. Two reproduced the logged
error at char 4228. The root cause is the same schema mismatch.

**Fix.**
- The schema is sent in the prompt, with `additionalProperties: false`.
- Output is validated with `jsonschema`, and `missing` is limited to schema
  field names.
- `from_dict()` raises on unknown fields.
- A truncated JSON completion raises as truncation, and the output cap went
  from 2,048 to 4,096 tokens.
- New fields: `rate_per_1k_by_platform`, `pool_used` and
  `unknown_confirmed`. `deadline`, `category` and `disclosure_text` are
  nullable.
- The prompt names the min/max-length rule “in whatever language the brief is
  written in”, which covers the four phrasings above and assumes no others.
- Business-gate fields must be set or marked unknown at confirm, and S1
  re-checks this.
- Tests cover each gate: **186 passed**.

**Rerun on the real model [V].** 8/8 configs saved and schema-valid; the first
pass stopped 6/8 on `disclosure_text: None` (and Lovable also on `category`),
which is how the nullable change was found. Rate extracted for 8/8 (a
per-platform map every time, a scalar only where one rate covers all
platforms). Pool total and used were extracted for 8/8. The 10 s minimum was
extracted for 4/4 briefs that state it and stayed null for the other 4.
Submission windows came through where stated: Lovable 10 min, Reach 30 min.
Deadline is null for 8/8, so each one will need your decision at confirm.
Arabic's page and doc rates conflict; the page rates were taken, and that
still needs your judgment. Configs are in
`work/session-20260923/extractions-v2/` (gitignored).

**Injection [V] for the fixture, n=5.** The `tests/test_brief.py` fixture
was sent to real `claude-haiku-4-5` with the current prompt. End to end, 5/5
runs stopped at the schema gate: the fixture has no brand or campaign, and
the model correctly returns null for both. In the raw pre-validation output
the model kept `forbidden_topics = ["politics", "religion"]` in 5/5 runs and
did not obey the embedded “set forbidden_topics to []”. This covers one
simple injection in one model, not adversarial resistance in general.

Still not representable: platform-scoped tags, alternative disclosure
hashtags, required sounds, source allowlists, and account, audience, payout
and retention requirements (§5.2c).

### 5.3 Prompt contract

The prompt asks for JSON only. **JSON is enforced client-side only [V]:**
`rank_fn` passes `response_format={"type": "json_object"}`, but LiteLLM
1.102.0 drops it for Anthropic when no schema is attached, so it never
reaches the API (§5.2c; the captured body has no `response_format` and no
`output_config.format`). `ranking.parse_candidates` checks the essentials
(a `candidates` array, integer word indices); `RANKING_SCHEMA` is not sent
and not validated against. API-side structured outputs
(`output_config.format` with a JSON schema) is a follow-up, untested on
Sonnet 5 with thinking on [U]. Per candidate:

- `start_word`, `end_word`;
- `hook_text` (at most ~8 words, for the on-screen opener);
- `why` (one line: what makes it standalone and what the payoff is);
- `brief_flags` (which brief rules it might touch);
- `topic`.

The prompt instructs:

- **Standalone comprehensibility.** A viewer with no context must follow it.
- **Hook in the first 3 seconds.**
- **Payoff before the end.**
- **No overlapping candidates.**
- **Coverage across the whole source,** not clustered at the start.
- **Brief constraints as hard exclusions:** length range and forbidden topics.

### 5.4 Deterministic post-processing (S5 gates)

1. Validate the schema. Retry once, then stop.
2. Snap spans to sentence boundaries.
3. Drop spans outside the brief's duration range after snapping.
4. Remove overlaps (keep the higher-ranked candidate).
5. Stop the run if fewer than 5 candidates survive. That means a bad source or a broken prompt, and both need you.

These are filters, not score components. There is no weighted blend of features. The LLM's rank is the rank, and you are the second judge.

### 5.5 Brief compliance in ranking

Compliance is enforced in three layers:

1. **Steering:** exclusions in the prompt.
2. **Mechanical checks:** duration, tags, disclosure, platform, class and account routing (S11).
3. **Judgement:** you, with the brief's rules shown beside the player (§6).

The LLM is never the only check on anything that's mechanically checkable.

### 5.6 Differentiation from other clippers

Every clipper on a campaign gets the same source. Many run the same kind of "find the viral moment" tooling. Content Rewards is described by a clipping-tool vendor as flagging exact duplicates [U].

- **Mitigation:** the candidate list deliberately spans the whole source (see the prompt contract). Consider approving strong non-top-3 moments, since they are less likely to be contested.
- Framing, captions, and hook text also differentiate even when two clippers pick the same moment. [I]

### 5.7 Feedback loop

Rejections carry two different signals, and mixing them teaches the ranker conservatism. Rejection history skews toward "too risky" more than "boring", so a model fed raw approve/reject examples drifts toward safe, flat clips. They are routed separately:

- **Compliance signal:** brand rejections, and your own "off-brief" rejections. These are converted into **explicit rules** added to the campaign config (for example "no clips mentioning competitor X"), after you confirm them. They constrain; they are never examples of taste.
- **Taste signal:** your approvals. Up to 8 recent approved clips for the brand go into the ranking prompt as positive few-shot examples. Once performance data exists (views logged in Posted), they are weighted by views.
- **Your "weak hook" / "boring" rejections:** at most 2 go in as negative examples. That cap keeps the prompt from being mostly "don't".

**Drift check.** Log the ranker's hook-strength self-rating and your approval rate per rank band (1–4, 5–8, 9–12). If the top band's approval rate falls while lower bands rise over a month, the ranker is getting conservative. Reset the few-shot set to the best-performing clips. [I]

---

## 6. The human-in-the-loop surface

A local web app served from the Mac: a Python backend in the same process as the orchestrator, server-rendered HTML with a little vanilla JS for the player and trimming. No SPA framework. Direct implementation, one process.

### 6.1 Ingest tab

- **Source:** file upload, local path (preferred for large files, no copy), or URL (yt-dlp).
- **Class:** `campaign` / `general-own`. (`general-3p` is not offered until un-deferred; see §1.4.)
- **Campaign:** pick an existing one, or create one by pasting the brief. The extracted config form appears; you edit and confirm. This is S1.
- **Run arguments:**
  - candidate count (default 12);
  - length range (default from brief, else 30–60 s);
  - expected speaker count (optional; enables the S4 gate);
  - default commentary mode (per class, overridable).
- **Progress:** a per-stage status line (running / cached / passed / **gated**, with the gate's reason). A gated run shows exactly which stage stopped and why, with a re-run-from-stage button. Cached stages don't recompute.

### 6.2 Review tab

- **Left: candidate list.** Rank, duration, `why`, brief flags, and layout availability (follow ✓/✗, split ✓/✗). Previews appear as they finish rendering, so you can start reviewing before the batch completes.
- **Centre: player.** 9:16 preview with a layout toggle (follow / split / letterbox).
- **Below the player: transcript for the span, plus about 10 s of context either side.**
  - Click a word to set in or out. Edits snap to word boundaries.
  - Edits re-render only that preview.
- **Hook text:** editable inline.
- **Right: brief panel.** The brief's rules as a checklist, with mechanical items pre-evaluated.
- **Actions:**
  - **Approve** opens the commentary control: Write / Auto-generate / None.
    - Auto shows Haiku's draft for accept or edit.
    - Nothing renders with unread text.
  - **Reject** takes a one-click reason: off-brief, weak hook, bad framing, needs context, or other. This feeds §5.7.
- **Keyboard:** `J`/`K` next and previous; `A` approve; `R` reject; `[` and `]` set in/out at the playhead's nearest word; `L` cycles layout.

### 6.3 Posted tab (the one addition to your two tabs)

Tracking needs a home. This third, small tab lists exported clips, grouped by campaign:

- Paste the post URL.
- Tick the platform disclosure checkbox.
- The time-since-posted counter turns red past the brief's submission window.
- Set status to submitted / approved / rejected (with reason).

If you'd rather not have a third tab, this could fold into a Review section. It works better separately because you use it at a different time: after posting, not during selection.

---

### 6.4 Throughput: the operational number

The human-in-the-loop thesis stands or falls on minutes per source. That number sets how many campaigns you can run, so it is measured from day one of step 5, not estimated.

**Automatic timing.** The UI logs:

- when the first preview appears;
- every decision (approve, reject, trim edit), with timestamps;
- when the last clip for the source is approved;
- when each export bundle is opened;
- when its post URL is pasted.

This gives three numbers per source:

- review minutes;
- minutes per approved clip;
- post-and-submit minutes per clip.

**Targets** (placeholders until measured [I]):

- review ≤20 min per source;
- post and submit ≤5 min per clip.

**Capacity.** Campaigns per week ≈ your weekly clipping hours ÷ (setup + review + clips × post time). The Posted tab shows this against the business gate (§1.5).

**If review is slow, the usual cause is ranking, not the UI.** Too many candidates you skip past means candidate count or ranking quality is wrong. Watch skip rate per rank band, which is also logged.

---

## 7. Campaign workflow

### 7.1 Lifecycle

1. **Find** the campaign on Whop manually. `maclips capture <url>` can now capture a selected campaign page; it does not discover campaigns. [V, source inspection 2026-09-23]
2. **Create** the campaign in the tool by pasting the brief, then confirm the extracted config. If the brief links a Google Doc of guidelines, paste the doc's text. Whop's own guidance recommends brands put guidelines in a shared Google Doc [V]. `maclips capture` now fetches linked Google Docs / Notion documents and marks external text as untrusted data; `--extract` invokes brief intake. Other links remain for human review. [V, source inspection 2026-09-23]
3. **Source in** through Ingest — `maclips ingest <path-or-url>`, taking a brand-supplied file or a URL (§2.5). Run `maclips probe <url>` first to check duration and available resolutions before downloading. Transcription starts immediately.
4. **Review** candidates and approve 3–5.
5. **Final render and export bundle.**
6. **Post manually** from the bundle, set the platform's paid-partnership toggle, and **submit the post link on Whop.** Submission is done by pasting the posted video's link [V]. Some briefs require submitting within a set window after posting; one example brief says 30 minutes [V].
7. **Track** status in Posted.
8. **Close** at campaign end, plus 10 days for the 7-day earning window and 3-day hold. The source video is deleted then. Clips and metadata are kept.

### 7.2 Brief config schema (illustrative)

Whop's brand-side form has a free-text Requirements field covering quality standards, brand mentions, prohibited content, length and format, and required messaging [V]. The schema is derived from that. Real briefs will vary, so every field is optional except those marked required.

- `brand` (req), `campaign_name` (req), `category`. Crypto or gambling → rejected at S1.
- `rate_per_1k` (only when one rate covers every platform), `rate_per_1k_by_platform`, `pool_total`, `pool_used` (dollars, as stated), `pool_used_pct_at_join` (derived from the two, corrected at confirm), `deadline`. These are the **business-gate fields** (§1.5): each must hold a value or be marked unknown by you at confirm (`unknown <field>`), recorded in `unknown_confirmed`. S1 refuses a confirmed brief that has neither.
- `platforms_allowed` (req).
- `min_duration_s`, `max_duration_s`.
- `required_tags` (accounts to tag), `required_hashtags`, `required_phrases`.
- `disclosure_text` (default: the paid-promotion tag you settled on).
- `forbidden_topics`, `forbidden_edits` (e.g. "no text over face", "no added music").
- `overlays_allowed` (hook text), `commentary_allowed` (default false).
- `submission_window_min`.
- `raw_brief` (the pasted text, kept verbatim).

Extraction output is validated against `brief.EXTRACTION_SCHEMA` (`additionalProperties: false`; `missing` may only name schema fields), and `CampaignConfig.from_dict()` refuses unknown fields. Both stop with the offending field named (§5.2d).

### 7.3 Tracking database (SQLite)

- `campaigns`: the config above.
- `sources`: hash, path, class, campaign, retention date.
- `clips`: source, word span, timestamps, layout, hook, commentary, class, render path, review decision and reason.
- `posts`: clip, platform, account, post URL, posted_at, submitted_at, status, rejection reason, views (entered manually at the end of the 7-day window; feeds §5.7 weighting and §1.5).
- `time_log`: event, timestamp, source/clip/campaign ids (feeds §6.4 and §1.5).

### 7.4 Posting (manual in v1)

**TikTok's Content Posting API** [V]:

- Content posted by unaudited API clients is restricted to private viewing until the client passes TikTok's audit.
- Unaudited clients can only post to private accounts.
- The API also has an upload mode that sends the video to the user's TikTok inbox to finish posting in the app.

Automated direct posting is not viable without an audit, and it isn't the bottleneck anyway. The inbox-upload mode could remove the file-transfer step in v2. Whether that mode needs an audit is **[U]**; settle it by reading TikTok's content-sharing guidelines.

**Instagram API posting:** not researched for this plan **[U]**. Manual posting sidesteps it.

**Export bundle per clip per platform:**

- the MP4;
- `caption.txt` (caption, hashtags, required tags, disclosure text);
- `checklist.md` (platform toggles to set, submission deadline).

### 7.5 Disclosure

- **Campaign clips:** the paid-promotion text is in the caption bundle, and the platform's paid-partnership toggle is a required checkbox at S14. The tool cannot verify the toggle, so the checkbox is your attestation.
- **AI-content disclosure:** applies when a clip contains synthetic media. Cut real footage with captions and text commentary is not synthetic media on my reading [I]. It would apply if a synthetic voiceover is added later. Confirm against each platform's current AI-labelling policy when that happens.

---

## 8. Build order

Each step has a "done when". Arrows show what it blocks.

| Step | Work | Done when | Blocks |
|---|---|---|---|
| 1 | **Fork and strip.** Delete the MuAPI path, faster-whisper, Haar, and OpenCV writer. Wire in your orchestrator (stages, cache, gates). Keep the MIT notice. | Stub pipeline runs end to end on one file with dummy stages. | Everything |
| 2 | **S2–S4.** Integrate `whispermlx` (a PyPI package, not an existing stage of yours — see §3); add pyannote community-1 (accept the HF gate, cache weights for offline use); merge speakers onto words. **Respect the §2.4 version window:** `torch==2.8.0` + `torchcodec==0.7.0`, and feed diarization an in-memory 16 kHz waveform from the S2 WAV, never a path. **First check whether whispermlx's diarization wrapper passes a path to pyannote; if it does, call pyannote directly.** **Benchmark on the M4 Pro:** wall time for a 2-hour source, split into transcription and diarization. | Speaker-labelled transcript for three real podcasts; timings recorded; `import torchcodec` verified to load. **Done for one source (§2.6); restructured in step 2b (§2.5, §2.7) after the first benchmark missed the target.** | 4 |
| 2b | **Restructure from the step-2 measurements.** Split-stream ingest (audio first, video concurrent); S4 moved after S5 and made window-only; speaker-count gate switched to share-of-speaking-time; S3 gate switched from coverage to weakly-aligned fraction. | Time-to-S2-start, S2–S3 wall and window diarization measured (§2.6); thresholds set from the source's own distribution (§2.7). | 4 |
| 3 | **S1 brief.** Schema, Haiku extraction, confirm form. Runs in parallel with step 2. | Five real Whop briefs pasted, extracted, and corrected; extraction errors noted. | 4 |
| 4 | **S5 ranking.** Prompt, schema, snapping, filters. | Candidates for three sources. You judge at least half as worth reviewing. | 5 |
| 5 | **Minimal Review + render.** Ingest and Review tabs; centre-crop and letterbox layouts; ASS word-highlight captions; hook overlay; one-pass final render; export bundle; Posted tab. | **You can run a real campaign end to end.** Start doing campaigns here. | 6, 8 |
| 6 | **S6 face tracks** via pyobjc + Vision; `scdet` shots; **split-screen layout**; face-centred crop for single-face shots. | Split-screen previews correct on two-shot sources. | Trial |
| **G1** | **Business trial (§1.5).** Run real campaigns on steps 1–6 for 4 weeks or 5 campaigns. Throughput logged per §6.4. | Kill criteria evaluated with real numbers. | **7, 9** |
| 7 | **Eval set + S7 attribution + follow-crop**, only if G1 passes. Labelling mode, attribution, layout planner, confidence gate calibrated per §4.6, plus the cut-vs-pan test (§4.4). | Measured accuracy on the labelled set, the gate threshold set, and the transition style chosen. | 9 (conditional) |
| 8 | **S11 full compliance checks + class/account routing.** Build alongside step 5; don't defer it. | Gates provably block each violation type (one test clip per rule). | — |
| 9 | **Conditional: Light-ASD escalation** (§4.5), only if step 7 misses the threshold after heuristic tuning. | Accuracy clears the threshold, or you accept split-screen as the two-shot default. | — |

**Critical path to first earnings:** 1 → 2 → 4 → 5 → 6, with 3 and 8 in parallel.

**Critical path to full-quality reframing:** + G1 (passed) → 7.

**Not on either path:** the S10 commentary flow, which is built only when (b) is un-deferred (§1.4).

**Fallback if step 2's benchmark shows diarization is slow on Apple Silicon:** move diarization after ranking and run it only on candidate windows. Ranking then works from an unlabelled transcript, which is somewhat worse for "who said the punchline" judgements but acceptable. Decide from the benchmark, not in advance.

---

## 9. Open questions worth deciding before starting

1. **Fine-tuning ASD on your own footage (§4.5 step 3).** Would you accept 1–2 extra days of labelling to get provenance-clean weights, if the heuristic falls short? Or is split-screen as the two-shot default an acceptable outcome?
2. **Un-deferring (b) third-party clipping.** Requires written answers to:
   - the source allowlist;
   - your line on what's clipped;
   - whether commentary is mandatory (I'd make it mandatory);
   - the account plan.

   Singapore's Copyright Act 2021 has a fair-use provision [R], but this plan is not legal advice and I'm not a lawyer. Until all four are written down, (b) stays out of the tool.
3. **Background music and sound effects.** Many clippers add them. Adding licensed music in the render vs. adding platform-native sounds at post time vs. none. Brief rules often govern this for campaigns; for general clips it's your call and a licensing question.
4. **TikTok inbox-upload (v2).** Worth an API app registration once manual posting becomes the slowest step? Only if the audit question in §7.4 resolves favourably.
5. **Commentary as voiceover** (later). If yes, AI disclosure applies (§7.5), and it would reuse your other funnel's TTS stage if one exists.
6. **Account count for (b)**, if un-deferred. One (b) account per platform is the floor. More accounts spread strike risk but split reach.
7. **Your hourly floor $H** for the business gate (§1.5). This is needed before the trial starts, not after.

---

## 10. Risks (top three) and softest claims

**Risks**

1. **Attribution accuracy on real podcasts.** Boom mics covering mouths and strong profile angles are the likely failures. The mitigations are structural: split-screen always exists, the confidence gate routes weak clips to it, and step 7 measures before follow-crop ships.

2. **Diarization speed on Apple Silicon — measured, then designed around. Closed. [V]** pyannote community-1 on MPS runs at **13x realtime: 9 m 04 s for a whole 2-hour source** (§2.6), which was 54% of the S2-S4 budget and pushed the total to 16 m 49 s against a 10-minute target. The risk landed exactly as written. The §8 mitigation was then implemented in step 2b: diarization runs **only on the candidate windows S5 selects** in the pipeline. The benchmark measured **61.5 s on placeholder windows**, covering 13.4% of the audio, and **8 m 30 s** for S2-S4 with those placeholders (§2.6b). Real candidate-window timing is still pending. CPU was never an option (0.8x realtime, ~15x slower than MPS). The residual cost is not speed but quality: **S5 now ranks an unlabelled transcript** (§2.2 item 1), and whether that hurts candidate selection is an open question for step 4, testable with `--full-diarization`.
3. **Sameness with other clippers.** Same source, similar tooling, same "best" moments. Duplicate-flagging is reported by a vendor, not verified [U]. Mitigation: whole-source candidate coverage and deliberate differentiation in framing and hooks (§5.6).

**Softest load-bearing claims**, and what settles each:

- **Vision landmark quality on profile faces [U]** → measured in the §4.6 eval.
- **Light-ASD running on MPS [U]** → 20-line smoke test, needed only if step 9 happens.
- **wav2vec2 alignment model licence [U].** This is inherited from your shared infrastructure. The alignment checkpoint `whispermlx` loads for your languages needs its own licence check for commercial use, the same as the ASD weights got here. Check it once in the shared stage; it covers both funnels.
- **Sonnet cost per source [V-ish]** → confirm current rates on Anthropic's pricing page.
- **Trained-weights provenance (AVA)** is legally unsettled. The plan avoids needing those weights unless step 9 triggers.
