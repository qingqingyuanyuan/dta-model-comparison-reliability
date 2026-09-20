#!/usr/bin/env python3
"""[DEPRECATED] Unsafe cross-architecture checkpoint conversion.

The previous script attempted to map weights from a hand-written GAT
implementation into a PyG GATConv model and then saved the partially matched
state with ``strict=False``. That can silently leave important layers random.

For the final ZhiYao-Graph pipeline this operation is disabled.

Required rule:
  A paper checkpoint must be loaded by the exact architecture that produced it,
  preferably with ``strict=True`` and explicit dataset/setting/scaler metadata.
"""

raise RuntimeError(
    "scripts/convert_weights.py is disabled for the final pipeline. "
    "Do not convert checkpoints across incompatible GAT implementations. "
    "Retrain with the canonical model code or use the exact legacy architecture "
    "only for historical inspection."
)
