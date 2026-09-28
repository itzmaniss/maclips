# maclips status (read this first)

Short, current state. `docs/PLAN.md` is the full reference (about 45k tokens). Read only the sections a task names. Labels: [V] verified, [R] retrieved from memory, [I] inferred, [U] unverified. Updated 2026-09-28 at `f374f50`, with 398 tests passing.

## What it is
A local, semi-automated clipping tool for one M4 Pro Mac (48 GB) with a browser UI. It takes a long source (and a campaign brief when there is one), produces ranked 9:16 captioned previews, and gives them to a human for review. Then it does a one-pass final render and builds an export bundle. Posting is manual.

## Pipeline (execution order S0 S1 S2 S3 S5 S4 S6 S7 S8 · S9 · S11 S12 S13 S14)
| Stage | What | Where |
|---|---|---|
| S0 | ingest; split-stream yt-dlp, audio first | `ingest.py`, `ffmpeg.py` (absolute ffmpeg paths from `config.py`) |
| S1 | brief extraction (Haiku) plus a confirm form | `brief.py` |
| S2/S3 | audio; whispermlx transcription and alignment; gate at 12% weak words | `audio.py`, `transcribe.py`, `gates.py` |
| S5 | ranking: LLM picks word-index spans, then deterministic post-processing | `ranking.py`, `llm.py`, `signals.py` |
| S4 | pyannote diarization, only on the S5 windows | `diarize.py` |
| S6 | Apple Vision face tracks | `vision.py` |
| S7 | layouts: split / face-centred / letterbox | `layouts.py` |
| S8 | 540×960 previews | `render.py`, `pacing.py`, `render_stages.py` |
| S9 | Review (human gate) | `web.py`, `static/app.js`, `templates/app.html` |
| S11–S13 | compliance, the 1080×1920 final, the export bundle | `compliance.py`, `render.py`, `export.py` |
| S14 | post tracking | `db.py`, `tracking.py` |

Stages are wired in `stages.py` and `orchestrator.py`, and the CLI is `cli.py`. The cache is keyed on the content hash plus the stage version: **any change to a stage's output must bump that stage's version.**

## Current settings (decided)
- **Ranking:**
  - `claude-sonnet-5`, low effort, structured output (`output_config`);
  - the model's order is the rank, and scores are logged only, never sorted on;
  - word indices only, never timestamps;
  - default clip range 10–180 s; S5 fails if fewer than 5 clips survive.
- **Prompts:** the approved PROMPT is the default. `--prompt-version v3` (supoclip ideas, 5 clips, subscores, signals) is experimental (PLAN §5.2i).
- **Local model:** `MACLIPS_LOCAL_LLM_ENABLED=true` plus `MACLIPS_RANKING_MODEL=openai/mlx-community/gemma-4-26B-A4B-it-qat-4bit` with `mlx_lm.server`. It is free but takes about 8–13 minutes per ranking, and the server ignores the JSON schema, so it usually needs one retry.
- **End boundaries (PLAN §6.7):**
  - clip edges padded 0.12 s into silence;
  - a sentence-start end is read as exclusive;
  - "..." doesn't count as a sentence end;
  - pause-aware extension of at most 2 sentences;
  - extension to the end of the diarization turn.
- **Render:**
  - continuous 6-word captions;
  - dead-air cuts over 0.8 s;
  - punch-in zoom 1.15×;
  - end card and cold open (off / tease / payoff), both off by default.

## Waiting on the user (human gates)
1. Rate blind sheet 1: `work/session-20260925-bakeoff/bakeoff-blind-sheet.md` (Sonnet low / Sonnet high / Gemma 26B / Gemma E4B on video 3).
2. Rate blind sheet 2: `work/session-20260928-promptv3/v3-blind-sheet.md` (prompt v3, Sonnet low and Gemma 26B).
3. Watch `previews-endfix/` for videos 2 and 3.
4. Set $H (the hourly floor; placeholder $20) and run one real campaign end to end.

## Open decisions
- **The ≥5 gate at a requested count of 5 (prompt v3):** scale the floor, or request 6–7.
- **S8 face-box crash:** one out-of-frame Vision box stops S8 for the whole source. Proposed: clamp the boxes and fail per clip.
- **Cold-open lines:** only 5 of 44 are valid. Proposed option A: trim in code to the shortest complete phrase of 1.5–4 s.
- **Bare-fragment endings** ("I think."): no rule yet.
- **Parked:** TRIBE v2 scorer (`docs/todo.md`); a Gemma local judge scored against the user's ratings.

## Data (gitignored, under `work/`)
- Video 2 `BcrjhdSUv4Y`: `work/7899e5b0d2733cfe/`.
- Video 3 `DZtGxNs9AVg`: `work/7f8b3b5213024774/`.
- Never open `*blind-key.json`; only the user unblinds.
