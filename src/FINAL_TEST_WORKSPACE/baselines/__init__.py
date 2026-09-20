"""Controlled reference baselines for the final ZhiYao-Graph experiments.

These models are *style reimplementations* evaluated on the exact same fixed
KIBA/Davis splits and train-only scalers as ZhiYao-Graph. They are not presented
as bit-for-bit reproductions of external repositories.
"""

from .registry import (
    BASELINE_NAMES,
    build_baseline_model,
    get_baseline_config,
    load_baseline_checkpoint,
)

__all__ = [
    "BASELINE_NAMES",
    "build_baseline_model",
    "get_baseline_config",
    "load_baseline_checkpoint",
]
