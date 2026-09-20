#!/usr/bin/env python3
"""Held-out test evaluation for controlled reference baselines.

This mirrors the canonical evaluator's record-level primary metric and
unique-input sensitivity metric while loading the exact baseline architecture
stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from baselines.dataset import create_fixed_sequence_test_loader
from baselines.registry import load_baseline_checkpoint
from data.dataset import build_graph_cache, create_loader_from_frame
from data.utils import denormalize_affinity, load_fixed_split
from train.evaluate import Evaluator, METRIC_SCHEMA_VERSION
from scripts.experiment_integrity import require_full_training_complete


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _scalers_match(a: dict, b: dict, atol=1e-10):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if str(a.get("method", "")).lower() != str(b.get("method", "")).lower():
        return False
    method = str(a.get("method", "")).lower()
    if method == "minmax":
        return (
            all(
                k in a
                and k in b
                and np.isclose(float(a[k]), float(b[k]), rtol=0.0, atol=atol)
                for k in ("min", "max", "range")
            )
            and b.get("fit_on") == "train_only"
        )
    if method == "zscore":
        return (
            all(
                k in a
                and k in b
                and np.isclose(float(a[k]), float(b[k]), rtol=0.0, atol=atol)
                for k in ("mean", "std")
            )
            and b.get("fit_on") == "train_only"
        )
    return method == "none"


def _graph_test_loader(dataset, setting, data_root, bc, batch_size, num_workers, show_progress):
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
        max_len=int(bc["max_protein_len"]),
        truncation="right",
        num_workers=num_workers,
        trim_protein_padding=True,
    )
    return train_df, val_df, test_df, scaler, loader


def _prepare_loader(dataset, setting, data_root, ckpt, batch_size, num_workers, show_progress):
    bc = ckpt["config"]["BASELINE"]
    modality = str(bc["input_modality"])
    if modality == "smiles_sequence+protein_sequence":
        return create_fixed_sequence_test_loader(
            dataset,
            setting,
            data_root=data_root,
            batch_size=batch_size,
            max_smiles_len=int(bc["max_smiles_len"]),
            max_protein_len=int(bc["max_protein_len"]),
            protein_truncation="right",
            num_workers=num_workers,
        )
    if modality == "molecular_graph+protein_sequence":
        return _graph_test_loader(
            dataset,
            setting,
            data_root,
            bc,
            batch_size,
            num_workers,
            show_progress,
        )
    raise ValueError(f"Unsupported baseline input modality: {modality}")


@torch.no_grad()
def _predict(model, loader, device):
    model = model.to(device)
    model.eval()
    preds, labels = [], []
    for drug_input, proteins, affinities in loader:
        pred = model(
            drug_input.to(device, non_blocking=True),
            proteins.to(device, non_blocking=True),
        )
        preds.append(pred.detach().cpu().numpy().reshape(-1))
        labels.append(affinities.detach().cpu().numpy().reshape(-1))
    if not preds:
        raise RuntimeError("empty test loader")
    return np.concatenate(preds), np.concatenate(labels)


def _unique_input_sensitivity(record_df: pd.DataFrame, scaler: dict):
    grouped = (
        record_df.groupby(
            ["Canonical SMILES", "Target Sequence"],
            sort=False,
            as_index=False,
        )
        .agg(
            Label_raw=("Label", "median"),
            Label_norm=("Label_norm", "median"),
            Prediction_norm=("Prediction_norm", "mean"),
            Prediction_raw=("Prediction_raw", "mean"),
            RecordCount=("Prediction_norm", "size"),
        )
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


def evaluate_baseline(
    *,
    dataset_name: str,
    setting: str,
    ckpt_path: Path,
    data_root: Path,
    batch_size: int,
    num_workers: int,
    device_name: str,
    output_dir: Path,
    show_progress: bool = True,
    overwrite: bool = False,
):
    dataset = str(dataset_name).upper()
    setting = str(setting).lower()
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    # Same global held-out test seal as the canonical evaluator.
    require_full_training_complete()
    model, ckpt = load_baseline_checkpoint(ckpt_path)
    baseline_name = str(
        ckpt.get("baseline_name")
        or (ckpt.get("extra_metadata") or {}).get("baseline_name")
    ).lower()
    if str(ckpt.get("dataset", "")).upper() != dataset:
        raise ValueError("baseline checkpoint dataset mismatch")
    if str(ckpt.get("setting", "")).lower() != setting:
        raise ValueError("baseline checkpoint setting mismatch")
    if ckpt.get("test_evaluated_during_training") is not False:
        raise ValueError("baseline checkpoint does not prove sealed-test training")

    train_df, val_df, test_df, scaler, loader = _prepare_loader(
        dataset,
        setting,
        data_root,
        ckpt,
        int(batch_size),
        int(num_workers),
        show_progress,
    )
    if not _scalers_match(ckpt.get("scaler"), scaler):
        raise ValueError("baseline checkpoint scaler != fixed train-only scaler")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto"
        else device_name
    )

    stem = f"{ckpt_path.stem}__{dataset.lower()}__{setting}"
    record_path = output_dir / f"{stem}__test_predictions.csv"
    unique_path = output_dir / f"{stem}__unique_input_predictions.csv"
    meta_path = output_dir / f"{stem}__evaluation.json"
    existing = [p for p in (record_path, unique_path, meta_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Baseline evaluation output exists; use --overwrite:\n  "
            + "\n  ".join(str(x) for x in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("FINAL CONTROLLED-BASELINE HELD-OUT TEST")
    print(f"Baseline   : {baseline_name}")
    print(f"Architecture: {ckpt.get('architecture_version')}")
    print(f"Dataset    : {dataset}")
    print(f"Setting    : {setting}")
    print(f"Seed       : {ckpt.get('model_seed')}")
    print(f"Test rows  : {len(test_df):,}")
    print(f"Device     : {device}")
    print("=" * 76)

    y_pred_norm, y_true_norm = _predict(model, loader, device)
    if len(y_pred_norm) != len(test_df):
        raise RuntimeError("prediction count != fixed test rows")
    if not np.allclose(test_df["Label_norm"].to_numpy(float), y_true_norm):
        raise RuntimeError("baseline test loader order != test.csv")

    record_metrics = Evaluator(scaler=scaler).evaluate(y_true_norm, y_pred_norm)
    record_df = test_df.copy()
    record_df.insert(0, "TestRowIndex", np.arange(len(record_df), dtype=int))
    record_df["Prediction_norm"] = y_pred_norm
    record_df["Prediction_raw"] = denormalize_affinity(y_pred_norm, scaler)
    record_df.to_csv(record_path, index=False)

    unique_df, unique_metrics = _unique_input_sensitivity(record_df, scaler)
    unique_df.to_csv(unique_path, index=False)

    bc = ckpt["config"]["BASELINE"]
    metadata = {
        "evaluation_artifact_version": 1,
        "experiment_family": "controlled_baseline",
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "baseline_name": baseline_name,
        "architecture_version": ckpt.get("architecture_version"),
        "baseline_description": bc.get("description"),
        "dataset": dataset,
        "setting": setting,
        "model_seed": int(ckpt.get("model_seed")),
        "input_modality": bc.get("input_modality"),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": _sha256(ckpt_path),
        "test_csv": str((data_root / dataset / setting / "test.csv").resolve()),
        "test_csv_sha256": _sha256(data_root / dataset / setting / "test.csv"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_samples": int(len(record_df)),
        "unique_model_inputs": int(len(unique_df)),
        "scaler": scaler,
        "baseline_config": bc,
        "parameter_summary": ckpt.get("parameter_summary"),
        "record_level_metrics": record_metrics,
        "unique_input_level_metrics": unique_metrics,
        "primary_metrics_level": "record",
        "sensitivity_metrics_level": "unique_input_median_observed_label",
        "record_prediction_file": str(record_path),
        "unique_input_prediction_file": str(unique_path),
        "reporting_label": (
            "controlled style reimplementation; not a claim of exact published-repository reproduction"
        ),
    }
    _atomic_json(meta_path, metadata)

    print("\n[record-level]")
    Evaluator.print_metrics(record_metrics)
    print("\n[unique-input sensitivity]")
    Evaluator.print_metrics(unique_metrics)
    print(f"\nMetadata: {meta_path}")
    return metadata


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, choices=["kiba", "davis"])
    p.add_argument("--setting", required=True, choices=[str(x).lower() for x in cfg.DATA_CONFIG["settings"]])
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--batch", type=int, default=int(cfg.TRAINING["batch_size"]))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "evaluation_outputs" / "baselines",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_baseline(
        dataset_name=args.data,
        setting=args.setting,
        ckpt_path=args.ckpt,
        data_root=args.data_root,
        batch_size=args.batch,
        num_workers=args.num_workers,
        device_name=args.device,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
        overwrite=args.overwrite,
    )
