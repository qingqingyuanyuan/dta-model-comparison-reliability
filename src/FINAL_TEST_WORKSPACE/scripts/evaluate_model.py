#!/usr/bin/env python3
"""Final held-out test evaluation for ZhiYao-Graph.

This script is the only canonical path that consumes ``test.csv`` for model
performance. It computes both:
  1. record-level metrics on every retained benchmark record (primary), and
  2. unique-input sensitivity metrics after median aggregation of repeated
     identical (canonical SMILES, protein sequence) labels.

The second view is important because the full-data policy intentionally retains
repeated assay records within a split while preventing identical inputs from
crossing split boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

from data.dataset import build_graph_cache, create_loader_from_frame  # noqa: E402
from data.utils import denormalize_affinity, load_fixed_split  # noqa: E402
from models.colab_model import load_model  # noqa: E402
from models.predictor import ARCHITECTURE_VERSION  # noqa: E402
from train.evaluate import Evaluator, METRIC_SCHEMA_VERSION  # noqa: E402
from scripts.experiment_integrity import require_full_training_complete  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _trusted_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    ckpt = _trusted_torch_load(path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Unrecognized checkpoint format")
    return ckpt


def _scalers_match(a: dict, b: dict, atol=1e-10) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if str(a.get("method", "")).lower() != str(b.get("method", "")).lower():
        return False
    method = str(a.get("method", "")).lower()
    if method == "minmax":
        for key in ("min", "max", "range"):
            if key not in a or key not in b:
                return False
            if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=atol):
                return False
        return b.get("fit_on") == "train_only"
    if method == "zscore":
        for key in ("mean", "std"):
            if key not in a or key not in b:
                return False
            if not np.isclose(float(a[key]), float(b[key]), rtol=0.0, atol=atol):
                return False
        return b.get("fit_on") == "train_only"
    return method == "none"


def _validate_provenance(ckpt, dataset, setting, fixed_scaler, *, allow_legacy):
    if ckpt.get("architecture_version") != ARCHITECTURE_VERSION:
        if not allow_legacy:
            raise RuntimeError(
                f"Final evaluation requires architecture={ARCHITECTURE_VERSION}; "
                f"got {ckpt.get('architecture_version')!r}"
            )
        print("[WARN] evaluating legacy architecture for historical debugging")

    ckpt_dataset = ckpt.get("dataset")
    ckpt_setting = ckpt.get("setting")
    ckpt_scaler = ckpt.get("scaler")
    if ckpt_dataset is None or ckpt_setting is None or ckpt_scaler is None:
        if allow_legacy:
            print("[WARN] legacy checkpoint provenance incomplete")
            return
        raise ValueError("Checkpoint lacks dataset/setting/scaler provenance")
    if str(ckpt_dataset).upper() != dataset:
        raise ValueError(f"Checkpoint dataset={ckpt_dataset!r} != {dataset!r}")
    if str(ckpt_setting).lower() != setting:
        raise ValueError(f"Checkpoint setting={ckpt_setting!r} != {setting!r}")
    if not _scalers_match(ckpt_scaler, fixed_scaler):
        if not allow_legacy:
            raise ValueError("Checkpoint scaler != fixed data_final train-only scaler")
        print("[WARN] legacy checkpoint scaler mismatch")

    if not allow_legacy and ckpt.get("test_evaluated_during_training") not in (False, None):
        raise ValueError("Checkpoint indicates test evaluation during training")


def _model_protocol(ckpt: dict, allow_legacy: bool) -> Tuple[int, str]:
    if ckpt.get("architecture_version") == ARCHITECTURE_VERSION:
        config = ckpt.get("config")
        if not isinstance(config, dict):
            raise ValueError("Canonical checkpoint is missing exact config")
        pc = config["PROTEIN_ENCODER"]
        return int(pc["max_len"]), str(pc.get("truncation", "right"))
    if not allow_legacy:
        raise RuntimeError("Legacy model protocol rejected")
    return 1000, "right"


def _prepare_test_loader(
    dataset,
    setting,
    *,
    data_root,
    max_len,
    truncation,
    batch_size,
    num_workers,
    show_progress,
):
    train_df, val_df, test_df, scaler = load_fixed_split(
        dataset, setting, data_root=data_root
    )
    graph_cache = build_graph_cache(
        test_df["Canonical SMILES"].astype(str).tolist(),
        show_progress=show_progress,
    )
    loader = create_loader_from_frame(
        test_df,
        split_name=f"{dataset}/{setting}/test",
        graph_cache=graph_cache,
        batch_size=batch_size,
        shuffle=False,
        model_seed=0,
        max_len=max_len,
        truncation=truncation,
        num_workers=num_workers,
        trim_protein_padding=True,
    )
    return train_df, val_df, test_df, scaler, loader


@torch.no_grad()
def _predict(model, loader, device):
    model = model.to(device)
    model.eval()
    preds, labels = [], []
    for drug_batch, protein_seqs, affinities in loader:
        y_pred = model(
            drug_batch.to(device, non_blocking=True),
            protein_seqs.to(device, non_blocking=True),
        )
        preds.append(y_pred.detach().cpu().numpy().reshape(-1))
        labels.append(affinities.detach().cpu().numpy().reshape(-1))
    if not preds:
        raise RuntimeError("Fixed test loader is empty")
    return np.concatenate(preds), np.concatenate(labels)


def _unique_input_sensitivity(record_df: pd.DataFrame, scaler: dict):
    """Median-aggregate repeated observed labels per identical model input."""
    key = ["Canonical SMILES", "Target Sequence"]
    grouped = (
        record_df.groupby(key, sort=False, as_index=False)
        .agg(
            Label_raw=("Label", "median"),
            Label_norm=("Label_norm", "median"),
            Prediction_norm=("Prediction_norm", "mean"),
            Prediction_raw=("Prediction_raw", "mean"),
            RecordCount=("Prediction_norm", "size"),
            PredictionMin_norm=("Prediction_norm", "min"),
            PredictionMax_norm=("Prediction_norm", "max"),
        )
    )
    grouped["PredictionSpread_norm"] = (
        grouped["PredictionMax_norm"] - grouped["PredictionMin_norm"]
    )
    max_spread = float(grouped["PredictionSpread_norm"].max()) if len(grouped) else 0.0
    if max_spread > 1e-5:
        print(
            f"[WARN] repeated identical inputs produced prediction spread up to {max_spread:.3e}"
        )

    metrics = Evaluator(scaler=scaler).evaluate(
        grouped["Label_norm"].to_numpy(float),
        grouped["Prediction_norm"].to_numpy(float),
    )
    return grouped, metrics


def _atomic_json(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def evaluate_checkpoint(
    *,
    dataset_name: str,
    setting: str,
    ckpt_path: Path,
    data_root: Path,
    batch_size: int = 64,
    num_workers: int = 0,
    device_name: str = "auto",
    output_dir: Path = Path("evaluation_outputs"),
    allow_legacy: bool = False,
    show_progress: bool = True,
    overwrite: bool = False,
) -> Dict[str, object]:
    dataset = str(dataset_name).upper()
    setting = str(setting).lower()
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    ckpt = _load_checkpoint(ckpt_path)
    if not allow_legacy:
        # Fifth-stage global seal: canonical held-out test evaluation is blocked
        # until primary + controlled baselines + sensitivity training are all frozen.
        require_full_training_complete()
    max_len, truncation = _model_protocol(ckpt, allow_legacy)

    train_df, val_df, test_df, scaler, loader = _prepare_test_loader(
        dataset,
        setting,
        data_root=data_root,
        max_len=max_len,
        truncation=truncation,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        show_progress=show_progress,
    )
    _validate_provenance(
        ckpt, dataset, setting, scaler, allow_legacy=allow_legacy
    )

    model, _ = load_model(ckpt_path, allow_legacy=allow_legacy)
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    stem = f"{ckpt_path.stem}__{dataset.lower()}__{setting}"
    record_path = output_dir / f"{stem}__test_predictions.csv"
    unique_path = output_dir / f"{stem}__unique_input_predictions.csv"
    meta_path = output_dir / f"{stem}__evaluation.json"
    existing = [p for p in (record_path, unique_path, meta_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Evaluation output already exists; refusing silent overwrite:\n  "
            + "\n  ".join(str(x) for x in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("FINAL HELD-OUT TEST EVALUATION")
    print(f"Architecture : {ckpt.get('architecture_version', 'legacy')}")
    print(f"Dataset      : {dataset}")
    print(f"Setting      : {setting}")
    print(f"Fusion       : {ckpt.get('fusion_mode')}")
    print(f"Model seed   : {ckpt.get('model_seed')}")
    print(f"Test rows    : {len(test_df):,}")
    print(f"Protein      : max_len={max_len}, truncation={truncation}")
    print(f"Metric schema: {METRIC_SCHEMA_VERSION}")
    print(f"Device       : {device}")
    print("=" * 76)

    y_pred_norm, y_true_norm = _predict(model, loader, device)
    if len(y_pred_norm) != len(test_df):
        raise RuntimeError("Prediction count does not match fixed test.csv")
    if not np.allclose(test_df["Label_norm"].to_numpy(float), y_true_norm):
        raise RuntimeError("Loader order does not match fixed test.csv")

    record_metrics = Evaluator(scaler=scaler).evaluate(y_true_norm, y_pred_norm)
    print("\n[Primary: record-level metrics]")
    Evaluator.print_metrics(record_metrics)

    record_df = test_df.copy()
    record_df.insert(0, "TestRowIndex", np.arange(len(record_df), dtype=int))
    record_df["Prediction_norm"] = y_pred_norm
    record_df["Prediction_raw"] = denormalize_affinity(y_pred_norm, scaler)
    record_df.to_csv(record_path, index=False)

    unique_df, unique_metrics = _unique_input_sensitivity(record_df, scaler)
    unique_df.to_csv(unique_path, index=False)
    print("\n[Sensitivity: unique-input median-label metrics]")
    Evaluator.print_metrics(unique_metrics)

    repeated_records = int(len(record_df) - len(unique_df))
    extra_metadata = ckpt.get("extra_metadata") or {}
    metadata = {
        "evaluation_artifact_version": 3,
        "experiment_family": extra_metadata.get("experiment_family", "primary"),
        "sensitivity_name": extra_metadata.get("sensitivity_name"),
        "sensitivity_overrides": extra_metadata.get("sensitivity_overrides"),
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "architecture_version": ckpt.get("architecture_version"),
        "dataset": dataset,
        "setting": setting,
        "fusion": ckpt.get("fusion_mode"),
        "model_seed": ckpt.get("model_seed"),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "test_csv": str((data_root / dataset / setting / "test.csv").resolve()),
        "test_csv_sha256": _sha256(data_root / dataset / setting / "test.csv"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(record_df)),
        "unique_model_inputs": int(len(unique_df)),
        "repeated_records_beyond_unique_inputs": repeated_records,
        "repeated_fraction": float(repeated_records / len(record_df)) if len(record_df) else 0.0,
        "scaler": scaler,
        "protein_protocol": {
            "max_len": max_len,
            "truncation": truncation,
            "dynamic_batch_padding_trim": True,
        },
        "parameter_summary": ckpt.get("parameter_summary"),
        "primary_metrics_level": "record",
        "record_level_metrics": record_metrics,
        "sensitivity_metrics_level": "unique_input_median_observed_label",
        "unique_input_level_metrics": unique_metrics,
        "record_prediction_file": str(record_path),
        "unique_input_prediction_file": str(unique_path),
        "note": (
            "Record-level metrics are primary because the benchmark-full policy retains all "
            "non-exact-duplicate records. Unique-input metrics are a prespecified sensitivity "
            "analysis preventing repeated assay records from receiving extra weight."
        ),
    }
    _atomic_json(meta_path, metadata)

    print(f"\nRecord predictions: {record_path}")
    print(f"Unique inputs     : {unique_path}")
    print(f"Evaluation meta   : {meta_path}")
    return metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=["kiba", "davis"])
    p.add_argument("--setting", required=True, choices=[str(x).lower() for x in cfg.DATA_CONFIG["settings"]])
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_outputs")
    p.add_argument(
        "--allow-legacy",
        action="store_true",
        help="DEBUG ONLY: evaluate a historical noncanonical checkpoint",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        dataset_name=args.data,
        setting=args.setting,
        ckpt_path=args.ckpt,
        data_root=args.data_root,
        batch_size=args.batch,
        num_workers=args.num_workers,
        device_name=args.device,
        output_dir=args.output_dir,
        allow_legacy=args.allow_legacy,
        show_progress=not args.no_progress,
        overwrite=args.overwrite,
    )
