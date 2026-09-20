"""Baseline registry and strict checkpoint loading."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

import config as cfg

BASELINE_NAMES = tuple(cfg.BASELINE_EXPERIMENT_MATRIX["baselines"])


def get_baseline_config(name: str) -> dict:
    key = str(name).lower()
    if key not in cfg.BASELINES:
        raise ValueError(f"Unknown baseline {name!r}; allowed={sorted(cfg.BASELINES)}")
    return {
        "BASELINE": copy.deepcopy(cfg.BASELINES[key]),
        "PREDICTOR": {"task": "regression"},
        "TRAINING": copy.deepcopy(cfg.TRAINING),
    }


def build_baseline_model(name: str, config: dict | None = None):
    key = str(name).lower()
    config = copy.deepcopy(config) if config is not None else get_baseline_config(key)
    if key == "deepdta_style":
        from .deepdta import DeepDTAStyle
        return DeepDTAStyle(config)
    if key == "graphdta_gcn_style":
        from .graphdta import GraphDTAStyleGCN
        return GraphDTAStyleGCN(config)
    raise ValueError(f"Unknown baseline: {name!r}")


def _trusted_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_baseline_checkpoint(path: str | Path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = _trusted_load(path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Unrecognized baseline checkpoint")
    name = str(
        ckpt.get("baseline_name")
        or (ckpt.get("extra_metadata") or {}).get("baseline_name")
        or ""
    ).lower()
    if name not in BASELINE_NAMES:
        raise ValueError(f"Checkpoint lacks recognized baseline_name: {name!r}")
    config = ckpt.get("config")
    if not isinstance(config, dict) or "BASELINE" not in config:
        raise ValueError("Baseline checkpoint is missing exact BASELINE config")
    expected_arch = str(config["BASELINE"]["architecture_version"])
    if ckpt.get("architecture_version") != expected_arch:
        raise ValueError(
            f"Baseline architecture mismatch: checkpoint={ckpt.get('architecture_version')!r}, "
            f"config={expected_arch!r}"
        )
    model = build_baseline_model(name, config)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt
