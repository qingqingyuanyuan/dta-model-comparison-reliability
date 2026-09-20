#!/usr/bin/env python3
"""Benchmark model efficiency on validation data only.

The held-out test set is never opened. Reported latency includes batch transfer
to the selected device plus model forward, but excludes CSV parsing, RDKit graph
construction and initial tokenization/cache construction.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from baselines.dataset import create_sequence_loader_from_frame
from baselines.registry import BASELINE_NAMES, load_baseline_checkpoint
from data.dataset import build_graph_cache, create_loader_from_frame
from data.utils import load_fixed_train_val
from models.colab_model import load_model as load_canonical_model


def _trusted_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _model_kind(ckpt):
    extra = ckpt.get("extra_metadata") or {}
    baseline_name = str(
        ckpt.get("baseline_name") or extra.get("baseline_name") or ""
    ).lower()
    if baseline_name in BASELINE_NAMES:
        return "controlled_baseline", baseline_name
    if ckpt.get("architecture_version") == cfg.MODEL_ARCHITECTURE_VERSION:
        return "canonical", str(ckpt.get("fusion_mode", "unknown")).lower()
    raise ValueError(
        f"Unsupported checkpoint architecture: {ckpt.get('architecture_version')!r}"
    )


def _prepare_validation_loader(path, ckpt, kind, name, data_root, batch_size, num_workers):
    dataset = str(ckpt.get("dataset", "")).upper()
    setting = str(ckpt.get("setting", "")).lower()
    if dataset not in {"KIBA", "DAVIS"} or setting not in {str(x).lower() for x in cfg.DATA_CONFIG["settings"]}:
        raise ValueError(f"Checkpoint lacks valid dataset/setting: {path}")

    train_df, val_df, scaler = load_fixed_train_val(
        dataset, setting, data_root=data_root
    )

    if kind == "controlled_baseline" and name == "deepdta_style":
        model, _ = load_baseline_checkpoint(path)
        bc = ckpt["config"]["BASELINE"]
        loader = create_sequence_loader_from_frame(
            val_df,
            batch_size=batch_size,
            shuffle=False,
            model_seed=0,
            max_smiles_len=int(bc["max_smiles_len"]),
            max_protein_len=int(bc["max_protein_len"]),
            protein_truncation="right",
            num_workers=num_workers,
            trim_padding=True,
        )
        return model, loader, dataset, setting

    if kind == "controlled_baseline":
        model, _ = load_baseline_checkpoint(path)
        max_len = int(ckpt["config"]["BASELINE"]["max_protein_len"])
    else:
        model, _ = load_canonical_model(path, allow_legacy=False)
        max_len = int(ckpt["config"]["PROTEIN_ENCODER"]["max_len"])

    graph_cache = build_graph_cache(
        val_df["Canonical SMILES"].astype(str).tolist(),
        show_progress=False,
    )
    loader = create_loader_from_frame(
        val_df,
        split_name=f"{dataset}/{setting}/val_efficiency",
        graph_cache=graph_cache,
        batch_size=batch_size,
        shuffle=False,
        model_seed=0,
        max_len=max_len,
        truncation="right",
        num_workers=num_workers,
        trim_protein_padding=True,
    )
    return model, loader, dataset, setting


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_one(path, *, data_root, device, batch_size, warmup_batches, measure_batches, num_workers):
    path = Path(path).expanduser().resolve()
    ckpt = _trusted_load(path)
    kind, name = _model_kind(ckpt)
    model, loader, dataset, setting = _prepare_validation_loader(
        path,
        ckpt,
        kind,
        name,
        data_root,
        batch_size,
        num_workers,
    )
    model = model.to(device).eval()

    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(int(warmup_batches)):
            try:
                drug_input, proteins, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                drug_input, proteins, _ = next(iterator)
            drug_input = drug_input.to(device, non_blocking=True)
            proteins = proteins.to(device, non_blocking=True)
            _ = model(drug_input, proteins)
        _synchronize(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        latencies = []
        total_samples = 0
        for _ in range(int(measure_batches)):
            try:
                drug_input, proteins, labels = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                drug_input, proteins, labels = next(iterator)

            _synchronize(device)
            t0 = time.perf_counter()
            drug_input = drug_input.to(device, non_blocking=True)
            proteins = proteins.to(device, non_blocking=True)
            pred = model(drug_input, proteins)
            _synchronize(device)
            elapsed = time.perf_counter() - t0
            if not torch.isfinite(pred).all():
                raise FloatingPointError("non-finite prediction during efficiency benchmark")
            latencies.append(elapsed)
            total_samples += int(labels.numel())

    total_seconds = float(sum(latencies))
    ms = np.asarray(latencies, dtype=float) * 1000.0
    params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    peak = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    extra = ckpt.get("extra_metadata") or {}
    return {
        "checkpoint": str(path),
        "checkpoint_bytes": int(path.stat().st_size),
        "architecture_version": ckpt.get("architecture_version"),
        "experiment_family": extra.get("experiment_family", kind),
        "model_name": name,
        "dataset": dataset,
        "setting": setting,
        "model_seed": ckpt.get("model_seed"),
        "trainable_parameters": params,
        "batch_size": int(batch_size),
        "warmup_batches": int(warmup_batches),
        "measure_batches": int(measure_batches),
        "measured_samples": int(total_samples),
        "mean_batch_latency_ms": float(ms.mean()),
        "median_batch_latency_ms": float(np.median(ms)),
        "p95_batch_latency_ms": float(np.quantile(ms, 0.95)),
        "throughput_samples_per_second": (
            float(total_samples / total_seconds) if total_seconds > 0 else None
        ),
        "cuda_peak_memory_bytes": peak,
        "device": str(device),
        "measurement_scope": (
            "validation split; device transfer + model forward; excludes CSV parsing, "
            "RDKit graph construction and token/cache construction"
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, type=Path)
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch", type=int, default=int(cfg.EFFICIENCY_BENCHMARK["batch_size"]))
    p.add_argument("--warmup-batches", type=int, default=int(cfg.EFFICIENCY_BENCHMARK["warmup_batches"]))
    p.add_argument("--measure-batches", type=int, default=int(cfg.EFFICIENCY_BENCHMARK["measure_batches"]))
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "efficiency" / "efficiency_benchmark.csv",
    )
    args = p.parse_args()

    if args.batch <= 0 or args.warmup_batches < 0 or args.measure_batches <= 0:
        raise ValueError("invalid benchmark batch counts")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    rows = []
    for path in args.ckpt:
        print(f"Benchmarking: {path}")
        rows.append(
            _benchmark_one(
                path,
                data_root=args.data_root,
                device=device,
                batch_size=args.batch,
                warmup_batches=args.warmup_batches,
                measure_batches=args.measure_batches,
                num_workers=args.num_workers,
            )
        )

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "validation_only",
        "device": str(device),
        "rows": rows,
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
