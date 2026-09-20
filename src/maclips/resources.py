"""Release a stage's model before the next stage loads its own.

The pipeline holds three different models across S3 and S4 (Whisper on MLX,
wav2vec2 on torch/MPS, pyannote on torch/MPS). On a 32 GB unified-memory
machine that is shared with whatever else is open, holding two at once is how a
run starts swapping — and a swapping benchmark measures paging, not the
pipeline.

Dropping the last Python reference is not enough on its own: both frameworks
keep their own allocator caches, so each needs an explicit purge.
"""
from __future__ import annotations

import gc
from typing import Any


def release(*objects: Any) -> None:
    """Drop references, collect, then purge both framework caches."""
    for obj in objects:
        del obj
    gc.collect()

    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001 - a cache purge must never break a run
        pass

    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:  # noqa: BLE001
        pass
