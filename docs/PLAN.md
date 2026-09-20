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
2. Within roughly 15 minutes it presents 10–15 ranked, pre-framed, captioned candidate previews. The 15-minute figure is a target, benchmarked in build step 2.
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

Each stage checkpoints its output and is cached by content hash (source hash + stage parameters), using your existing orchestrator pattern. "Gate" means the stage stops. It does not warn and continue.

| # | Stage | Output | Hard gate (stops the run or blocks the artifact) |
|---|---|---|---|
| S0 | **Ingest**: register source (upload, local path, or URL via yt-dlp), probe with ffprobe | Source record, hash | No video or audio stream; unreadable container; duration under 2 min |
| S1 | **Brief** (campaign only): paste brief → Haiku extracts config → you confirm in the form | Confirmed campaign config | Unconfirmed brief → S5 will not start. Brief category is crypto/gambling → rejected |
| S2 | **Audio extract**: ffmpeg to 16 kHz mono WAV | `audio.wav` | — |
| S3 | **Transcribe + align** (reused, `whispermlx`) | Word-level timestamped transcript | Aligned-word coverage below threshold (e.g. 97%); detected language ≠ expected |
| S4 | **Diarize** (pyannote community-1, exclusive mode) → merge speaker labels onto words | Speaker-labelled word transcript | You gave an expected speaker count and it differs → stop and ask you to confirm |
| S5 | **Rank** (Sonnet) with brief constraints | Candidate list: word-index spans, hook, rationale | Malformed JSON after one retry; fewer than 5 valid candidates after snapping and duration filtering |
| S6 | **Visual analysis** on candidate windows only: shot boundaries (ffmpeg `scdet`), Apple Vision faces and mouth landmarks at ~5 fps, per-shot IoU tracking | Face tracks per shot | A candidate with no face track at all → that candidate gets letterbox layout only |
| S7 | **Speaker attribution + layout planning** | Per-clip crop plans: follow-crop, split-screen | Attribution confidence below threshold on more than X% of clip duration → **follow-crop blocked for that clip** (split or letterbox only) |
| S8 | **Proxy render**: 540×960, `h264_videotoolbox`, captions burned, all available layouts | Preview files | — |
| S9 | **Review** (you) | Approved clips: in/out, layout, hook text, commentary choice | Human gate by definition |
| S10 | **Commentary resolve**: Write / Auto-generate (Haiku) / None | Final overlay text | Auto-generated text not yet accepted by you → no final render |
| S11 | **Compliance check** against the confirmed brief and class rules | Pass/fail per clip | Any fail blocks the clip: duration out of range, missing required tags or disclosure, disallowed platform, CTA on campaign clip, clip routed to an account not allowed for its class |
| S12 | **Final render**: one ffmpeg pass from the original source | 1080×1920 MP4 | Output duration differs from plan by more than 0.1 s; wrong resolution; no audio stream |
| S13 | **Export bundle** per clip per platform | Folder: MP4, `caption.txt`, `checklist.md` | — |
| S14 | **Post + track** (manual): paste post URL, tick the disclosure checkbox | Tracking row | Campaign clip logged without the paid-promotion checkbox ticked → cannot be marked submitted |

### 2.2 Orderings that are load-bearing

1. **S2–S4 start the moment a source lands, in parallel with S1.** Transcription does not depend on the brief. You confirm the brief while the Mac transcribes. S5 waits for both. Speed to campaign is the point, and this removes brief-reading from the critical path. [I]

2. **Ranking (S5) runs before visual analysis (S6), and S6 runs only on candidate windows.** This is the main "light" decision:
   - Face detection and landmarks on a full 2-hour source at 5 fps is about 36,000 frames.
   - On 15 candidates × ~60 s it is about 4,500 frames.

   The cost is that ranking cannot use visual cues such as reactions or laughter on camera. For podcast material, the transcript carries most of the signal, so this is accepted. [I]

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
- **Captions:** ASS subtitles burned with libass. Confirm your ffmpeg build has it: `ffmpeg -filters | grep ass`. Homebrew's ffmpeg does. [R]

---

## 3. What's reused vs built

| Component | Origin | Notes |
|---|---|---|
| Transcription + wav2vec2 alignment | **Reused**: your `whispermlx` stage | Runs on the Apple GPU via MLX. No reimplementation. |
| Caption data | **Reused**: alignment output | Only the renderer is new (ASS word-highlight instead of SRT). |
| Orchestrator (checkpointing, hash cache, hard gates) | **Reused** | Same pattern, new stage list. |
| Pipeline entry, `get_highlights(..., llm_fn=)` seam, ranking prompt, subclip logic | **From fork** (MIT, keep the upstream copyright notice) | The prompt is rewritten substantially (§5). |
| MuAPI `mode="api"` path | **Deleted** | Local only. |
| `faster-whisper`, Haar cascade, OpenCV video writer, two-pass encode | **Deleted** | — |
| Diarization + word merge | **Built** | pyannote community-1. |
| Brief extraction, schema, confirm form | **Built** | — |
| Vision face tracking (pyobjc → Apple Vision) | **Built** | — |
| Speaker attribution, layout planner | **Built** | The core of §4. |
| One-pass final renderer | **Built** | ffmpeg `filter_complex`. |
| Web UI (Ingest / Review / Posted) | **Built** | — |
| SQLite tracking, export bundles | **Built** | — |

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

- The full speaker-labelled transcript, one line per sentence, prefixed with the first word index and the speaker label.
- A 2-hour podcast is roughly 25–35k tokens [I], so it fits in one call. No chunking, and no loss of cross-section context.

### 5.2 Model and cost

- **Model: Sonnet (`claude-sonnet-5`).** This deliberately breaks your "ranking → Haiku" tiering rule. This ranking is the quality-critical judgement that decides which moments ever reach approval.
- **Cost:** ~30k input plus ~4k output tokens per source ≈ **$0.15 per source** at $3 / $15 per million tokens. The rate comes from third-party pricing pages; confirm on Anthropic's pricing page before budgeting. [V-ish]
- **Haiku 4.5** ($1 / $5 per million tokens) is used for:
  - brief field extraction, with human confirmation;
  - commentary drafts, with human acceptance.

### 5.3 Prompt contract

The prompt returns JSON only, validated against a schema. Per candidate:

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

1. **Find** the campaign on Whop. This stays manual; the tool doesn't scrape Whop.
2. **Create** the campaign in the tool by pasting the brief, then confirm the extracted config. If the brief links a Google Doc of guidelines, paste the doc's text. Whop's own guidance recommends brands put guidelines in a shared Google Doc [V]. Fetching it automatically is v2.
3. **Source in** through Ingest (brand-supplied link or file). Transcription starts immediately.
4. **Review** candidates and approve 3–5.
5. **Final render and export bundle.**
6. **Post manually** from the bundle, set the platform's paid-partnership toggle, and **submit the post link on Whop.** Submission is done by pasting the posted video's link [V]. Some briefs require submitting within a set window after posting; one example brief says 30 minutes [V].
7. **Track** status in Posted.
8. **Close** at campaign end, plus 10 days for the 7-day earning window and 3-day hold. The source video is deleted then. Clips and metadata are kept.

### 7.2 Brief config schema (illustrative)

Whop's brand-side form has a free-text Requirements field covering quality standards, brand mentions, prohibited content, length and format, and required messaging [V]. The schema is derived from that. Real briefs will vary, so every field is optional except those marked required.

- `brand` (req), `campaign_name` (req), `category`. Crypto or gambling → rejected at S1.
- `rate_per_1k`, `pool_total`, `pool_used_pct_at_join`, `deadline`.
- `platforms_allowed` (req).
- `min_duration_s`, `max_duration_s`.
- `required_tags` (accounts to tag), `required_hashtags`, `required_phrases`.
- `disclosure_text` (default: the paid-promotion tag you settled on).
- `forbidden_topics`, `forbidden_edits` (e.g. "no text over face", "no added music").
- `overlays_allowed` (hook text), `commentary_allowed` (default false).
- `submission_window_min`.
- `raw_brief` (the pasted text, kept verbatim).

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
| 2 | **S2–S4.** Plug in `whispermlx`; add pyannote community-1 (accept the HF gate, cache weights for offline use); merge speakers onto words. **Benchmark on the M4 Pro:** wall time for a 2-hour source, split into transcription and diarization. | Speaker-labelled transcript for three real podcasts; timings recorded. | 4 |
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

2. **Diarization speed on Apple Silicon.** pyannote on MPS for a 2-hour file is unmeasured [U]. If it's slow it breaks the 15-minute target. The mitigation is the candidate-window-only fallback in §8. Settled by step 2's benchmark.

3. **Sameness with other clippers.** Same source, similar tooling, same "best" moments. Duplicate-flagging is reported by a vendor, not verified [U]. Mitigation: whole-source candidate coverage and deliberate differentiation in framing and hooks (§5.6).

**Softest load-bearing claims**, and what settles each:

- **Vision landmark quality on profile faces [U]** → measured in the §4.6 eval.
- **Light-ASD running on MPS [U]** → 20-line smoke test, needed only if step 9 happens.
- **wav2vec2 alignment model licence [U].** This is inherited from your shared infrastructure. The alignment checkpoint `whispermlx` loads for your languages needs its own licence check for commercial use, the same as the ASD weights got here. Check it once in the shared stage; it covers both funnels.
- **Sonnet cost per source [V-ish]** → confirm current rates on Anthropic's pricing page.
- **Trained-weights provenance (AVA)** is legally unsettled. The plan avoids needing those weights unless step 9 triggers.
