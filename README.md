# maclips

Semi-automated video clipping for a single Apple Silicon Mac. Fork of
[SamurAIGPT/AI-Youtube-Shorts-Generator](https://github.com/SamurAIGPT/AI-Youtube-Shorts-Generator)
(MIT — see `LICENSE` and `NOTICE`).

The build plan is `docs/PLAN.md`. **Build step 1 of 9** is done: the fork is
stripped, the orchestrator skeleton runs S0–S14 as stubs, and the startup
gates are live. No pipeline stage does real work yet.

## Requirements

macOS on Apple Silicon, and nothing else — there is no cross-platform path and
no hardware abstraction by design (`PLAN.md` §1.3). The startup gates enforce:

| Gate | Requires |
|---|---|
| `platform` | `darwin` / `arm64`, macOS 14+ |
| `python` | exactly 3.12 (whispermlx caps at <3.14, torchcodec 0.7 at ≤3.13) |
| `ffmpeg-filters` | `ass`, `subtitles`, `drawtext`, `scdet`, `crop`, `scale`, `loudnorm` |
| `ffmpeg-encoders` | `h264_videotoolbox` (previews), `libx264` (final) |
| `deno` | on PATH — yt-dlp runs YouTube's JS challenges on it |
| `yt-dlp-ejs` | importable (needs `yt-dlp[default]`, not plain `yt-dlp`) |
| `mlx` | MLX default device is the GPU |
| `torch-mps` | PyTorch can reach the MPS backend |
| `vision` | `pyobjc-framework-Vision` imports |

> **Homebrew's `ffmpeg` formula will not pass.** As of Homebrew 7.0 it is built
> without libass, freetype or fontconfig, so `ass`, `subtitles` and `drawtext`
> are all absent. Install the full build:
>
> ```sh
> brew install ffmpeg-full
> export PATH="$(brew --prefix ffmpeg-full)/bin:$PATH"   # keg-only
> ```

## Setup

```sh
uv sync                 # installs the pinned environment (Python 3.12)
uv run maclips check    # run the startup gates
```

Then create `.env` in the repo root (it is gitignored):

```sh
# Ranking (Sonnet), brief extraction and commentary drafts (Haiku), via LiteLLM.
ANTHROPIC_API_KEY=

# pyannote/speaker-diarization-community-1 is a gated download: accept the terms
# on its Hugging Face model page, then paste a read token here.
HUGGINGFACE_TOKEN=

# --- optional overrides ---
# MACLIPS_RANKING_MODEL=anthropic/claude-sonnet-5
# MACLIPS_BRIEF_MODEL=anthropic/claude-haiku-4-5
# MACLIPS_COMMENTARY_MODEL=anthropic/claude-haiku-4-5
# MACLIPS_WORK_DIR=./work

# --- optional local MLX LLM (config slot only; no model installed) ---
# Serve with:  mlx_lm.server --model <repo> --port 8080
# MACLIPS_LOCAL_LLM_ENABLED=true
# MACLIPS_LOCAL_LLM_MODEL=openai/<model-name-the-server-reports>
# MACLIPS_LOCAL_LLM_API_BASE=http://127.0.0.1:8080/v1
```

## Usage

```sh
uv run maclips check                       # gates only
uv run maclips probe <url>                 # metadata only, no download
uv run maclips ingest <path-or-url>        # register a source
uv run maclips ingest <url> --max-height 1920
uv run maclips run <path-or-url>           # run the pipeline
uv run maclips run source.mp4 --from S5    # recompute from one stage
uv run pytest                              # 71 tests
```

`probe` prints each available resolution with its aspect ratio and the width a
9:16 crop would have — run it before committing bandwidth. A 9:16 crop comes
from the source's **height**, so a 1080x1920 output needs source height >= 1920
to avoid upscaling; for 2:1 cinematic sources (common in podcasts) that means
the 3840x1920 rendition, not the one labelled "1080p".

Stages S0, S2, S3 and S4 are implemented. S1 and S5-S14 are still stubs.

## How the orchestrator works

Three properties, in `src/maclips/orchestrator.py`:

- **Content-hash caching.** A stage's key covers the source hash, its own id and
  version, the config keys it *declares* it reads, and its upstream stages'
  keys. Invalidation therefore cascades: re-transcribing changes S3's key,
  which changes S4's, and so on.
- **Per-stage checkpointing.** Completed output lands in
  `work/<hash>/checkpoints/S*.json`, written-then-renamed so an interrupted
  write cannot leave a half checkpoint. A gated run is fixed and resumed from
  the stage that stopped it.
- **Hard gates.** A stage raises `GateFailure` and the run stops with a reason;
  downstream stages report as `blocked`. Nothing warns and continues, and a
  gated stage writes no checkpoint.

## Layout

```
src/maclips/
  gates.py          startup environment gates (live)
  orchestrator.py   caching, checkpointing, gate runner (live)
  stages.py         S0–S14 stubs; gate conditions are real
  cli.py            `maclips check` / `maclips run`
  config.py         model slots, paths, secrets
  llm.py            the single LiteLLM seam
  highlights.py     get_highlights() + llm_fn seam, kept from the fork
  ffmpeg.py         subprocess-only ffmpeg; subclip kept from the fork
  ingest.py         S0: local path or URL via yt-dlp's Python API
  audio.py          load the S2 WAV into memory (stdlib; no decoder needed)
  transcribe.py     S3: whispermlx transcribe + wav2vec2 align
  diarize.py        S4: pyannote community-1, in-memory waveform only
  resources.py      release a stage's model before the next one loads
  bench.py          per-stage wall time, peak RSS, swap, memory pressure
  vision.py         Apple Vision face tracking — stub (build step 6)
  db.py             SQLite schema (PLAN.md §7.3), stdlib sqlite3, no ORM
docs/
  PLAN.md                            the build plan
  prompts/ranking-upstream-draft.md  the fork's prompt, kept as a draft
```
