#!/usr/bin/env python3
"""Single-sample DTA prediction with V3 checkpoint provenance checks.

Only load checkpoint files that you trust.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from data.build_graph import smiles_to_graph
from data.utils import denormalize_affinity, protein_seq_to_indices
from models.colab_model import load_model
from models.predictor import ARCHITECTURE_VERSION


def _trusted_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")



def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _scalers_match(a, b, atol=1e-10):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if str(a.get("method", "")).lower() != str(b.get("method", "")).lower():
        return False
    if str(a.get("method", "")).lower() == "minmax":
        for key in ("min", "max", "range"):
            if key not in a or key not in b:
                return False
            if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=atol):
                return False
        return b.get("fit_on") == "train_only"
    return True


def _validated_info(path: Path):
    ckpt = _trusted_torch_load(path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        return None
    if ckpt.get("architecture_version") != ARCHITECTURE_VERSION:
        return None

    dataset = ckpt.get("dataset")
    setting = ckpt.get("setting")
    scaler = ckpt.get("scaler")
    config = ckpt.get("config")
    if not dataset or not setting or not scaler or not isinstance(config, dict):
        return None

    dataset = str(dataset).upper()
    setting = str(setting).lower()
    scaler_path = Path(cfg.FINAL_DATA_DIR) / dataset / setting / "scaler.json"
    if not scaler_path.exists():
        return None
    final_scaler = _load_json(scaler_path)
    if not _scalers_match(scaler, final_scaler):
        return None

    return {
        "checkpoint": ckpt,
        "dataset": dataset,
        "setting": setting,
        "scaler": final_scaler,
        "config": config,
    }


def _discover_v3_checkpoint():
    ckpt_dir = Path(cfg.MODEL_DIR)
    if not ckpt_dir.exists():
        return None
    candidates = sorted(
        ckpt_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            if _validated_info(path) is not None:
                return path
        except Exception:
            pass
    return None


def run_prediction(
    smiles=None,
    sequence=None,
    ckpt_path=None,
    *,
    allow_legacy=False,
):
    if smiles is None:
        smiles = input("输入药物 SMILES: ").strip()
    if sequence is None:
        sequence = input("输入靶点蛋白序列: ").strip().upper()

    if ckpt_path is None:
        ckpt_path = _discover_v3_checkpoint()
        if ckpt_path is None:
            raise FileNotFoundError(
                "No validated V3 checkpoint was found; train a final model or pass --ckpt."
            )
    ckpt_path = Path(ckpt_path).expanduser().resolve()

    info = _validated_info(ckpt_path)
    if info is None and not allow_legacy:
        raise RuntimeError(
            "Checkpoint is not a validated V3 model aligned with data_final."
        )

    model, ckpt = load_model(ckpt_path, allow_legacy=allow_legacy)
    if info is None:
        print("[WARN] legacy checkpoint: debugging only, not paper/deployment evidence")
        scaler = ckpt.get("scaler", {"method": "none"})
        dataset = ckpt.get("dataset", "legacy/unknown")
        setting = ckpt.get("setting", "legacy/unknown")
        max_len = 1000
        truncation = "right"
    else:
        scaler = info["scaler"]
        dataset = info["dataset"]
        setting = info["setting"]
        pc = info["config"]["PROTEIN_ENCODER"]
        max_len = int(pc["max_len"])
        truncation = str(pc.get("truncation", "right"))

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("Invalid SMILES")
    canonical_smiles = Chem.MolToSmiles(
        mol, canonical=True, isomericSmiles=True
    )
    graph = smiles_to_graph(canonical_smiles)
    if graph is None:
        raise ValueError("Canonical SMILES graph construction failed")

    clean_seq = "".join(str(sequence).split()).upper()
    if not clean_seq:
        raise ValueError("Protein sequence is empty")
    if len(clean_seq) > max_len:
        print(
            f"[WARN] sequence length {len(clean_seq)} > max_len={max_len}; "
            f"using predefined truncation={truncation!r}."
        )

    seq_indices = protein_seq_to_indices(
        clean_seq,
        max_len=max_len,
        truncation=truncation,
    )
    seq_tensor = torch.tensor([seq_indices], dtype=torch.long)
    batch = Batch.from_data_list([graph])

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred_norm = float(
            model(batch.to(device), seq_tensor.to(device))
            .detach()
            .cpu()
            .reshape(-1)[0]
        )
    pred_raw = denormalize_affinity(pred_norm, scaler)

    print("\n" + "=" * 58)
    print("ZhiYao-Graph V3 — continuous DTA prediction")
    print("=" * 58)
    print(f"Checkpoint   : {ckpt_path.name}")
    print(f"Architecture : {ckpt.get('architecture_version', 'legacy')}")
    print(f"Dataset      : {dataset}")
    print(f"Setting      : {setting}")
    print(f"Protein len  : {len(clean_seq)} aa")
    print(f"Model max_len: {max_len} ({truncation})")
    print(f"Pred norm    : {pred_norm:.6f}")
    if scaler.get("method") != "none":
        print(f"Pred raw     : {float(pred_raw):.6f}")
    print(
        "This is a continuous affinity-regression output; no >0.5 binding "
        "classification or strong/moderate/weak threshold is applied."
    )
    return pred_norm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smiles")
    p.add_argument("--sequence")
    p.add_argument("--ckpt", type=Path)
    p.add_argument(
        "--allow-legacy",
        action="store_true",
        help="DEBUG ONLY: allow historical non-V3 checkpoint",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prediction(
        smiles=args.smiles,
        sequence=args.sequence,
        ckpt_path=args.ckpt,
        allow_legacy=args.allow_legacy,
    )
