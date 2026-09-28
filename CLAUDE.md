# maclips

Local semi-automated video clipping tool (Python, Apple Silicon). **Read `docs/STATUS.md` first.** `docs/PLAN.md` is the full reference, and it is large: read only the sections your task names.

## Rules
- All Python goes through uv: `uv run ...`, `uv add <pkg>`. Never use system python, and never hand-edit `pyproject.toml` or `uv.lock`. Add a dependency only with the user's approval.
- Tests: run `uv run pytest -q` before every commit, and keep it green.
- Commits: on `main` unless told otherwise, with a body saying what was verified. **No `Co-Authored-By` trailer.** Push after each phase.
- Never read, print or commit `.env` or secrets.
- Call ffmpeg/ffprobe only through the absolute paths in `config.py`.
- Any change to a stage's output bumps that stage's version (the cache keys on versions, not code).
- Gates are hard failures, not log lines. Diagnose before fixing, and report problems plainly.
- A feature is done only after a real CLI run on a real source; say what ran for real and what is only tested.
- Model IDs carry no date suffix (`claude-sonnet-5`, `claude-haiku-4-5`).
- Never run two MPS-heavy jobs at once (whisper, pyannote, mlx, renders).
- Label claims in `docs/PLAN.md` [V]/[R]/[I]/[U]. Update `docs/STATUS.md` when the state changes.
- Keep files under about 500 lines. Scratch work goes in `work/session-<date>-<topic>/` (gitignored).

## Human gates (only the user does these)
Rating blind sheets, confirming briefs, approving clips, exporting, posting, setting $H. **Never open `*blind-key.json`.**
No platform hash or duplicate-detection evasion.
