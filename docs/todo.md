# maclips todo (parked ideas)

## TRIBE v2 as a clip scorer (parked 2026-09-28)
- Model: https://huggingface.co/facebook/tribev2 (Meta). It predicts fMRI responses of an average subject (~20k cortical vertices per timestep) from video, audio and text. It does not predict engagement; any score would be a derived proxy [I].
- Idea: score rendered clips, check against the user's blind-sheet ratings, and add it to the pipeline as a second opinion only if it agrees. Later, combine it with post analytics (S14) to understand the audience.
- Open issues:
  - License is CC-BY-NC-4.0. The planner reads use inside a monetised workflow as commercial; the user reads it differently. The user decides.
  - Needs gated LLaMA 3.2-3B (the user logs in to HF), V-JEPA2 ViT-g and Wav2Vec-BERT. PyTorch; whether it runs on MPS is unverified [U].
  - If tried, run it in an isolated env under `work/`, not as a maclips dependency, on the 22 bake-off clips after ratings exist.
- Comparison baseline: a Gemma 4 26B local judge on the same clips.
