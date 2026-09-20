#!/usr/bin/env python3
"""[DEPRECATED] Historical KIBA checkpoint conversion entry.

The old version of this script loaded a hard-coded ``kiba_model.pt`` and
assumed that a checkpoint could be converted into the current app model merely
because some layer names/shapes matched. That is not safe enough for a final
paper pipeline.

No weight conversion is performed here anymore.

Use instead:
  - exact architecture + strict checkpoint loading for new final checkpoints;
  - ``scripts/evaluate_model.py`` to verify a checkpoint on its fixed test split;
  - legacy checkpoints only for historical/debug inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def audit_checkpoint(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Not a recognized project checkpoint")

    print("=" * 68)
    print("Legacy checkpoint audit — NO conversion performed")
    print("=" * 68)
    print(f"Path       : {path.resolve()}")
    print(f"Dataset    : {ckpt.get('dataset', 'missing')}")
    print(f"Setting    : {ckpt.get('setting', 'missing')}")
    print(f"Fusion     : {ckpt.get('fusion_mode', 'missing')}")
    print(f"Model seed : {ckpt.get('model_seed', ckpt.get('seed', 'missing'))}")
    print(f"Scaler     : {'present' if ckpt.get('scaler') else 'missing'}")
    print(f"History    : {'present' if ckpt.get('history') else 'missing'}")
    print(f"Metrics    : {'present' if ckpt.get('metrics') else 'missing'}")
    print(f"Tensors    : {len(ckpt['model_state_dict'])}")
    print()
    print(
        "This script intentionally refuses to generate a converted checkpoint. "
        "Architecture compatibility must be established by exact model code and "
        "strict loading, not by partial key mapping."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit_checkpoint(args.checkpoint)
