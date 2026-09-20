#!/usr/bin/env python3
"""Validation-only modality-utilization diagnostics for canonical V3 checkpoints.

The diagnostics perturb one modality at a time within a validation batch:
  - protein shuffle: drug fixed, protein reassigned;
  - drug shuffle: protein fixed, molecular graph reassigned.

Large changes indicate the prediction depends on that modality; near-zero changes
warn about possible modality collapse. These are *model-behavior diagnostics*,
not biological negative controls, causal explanations or binding-site evidence.

Held-out test diagnostics remain disabled by default and are not needed for the
paper's modality-utilization analysis.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

from data.dataset import build_graph_cache, create_loader_from_frame
from data.utils import load_fixed_train_val
from models.colab_model import load_model
from models.predictor import ARCHITECTURE_VERSION


def _summary(delta: np.ndarray) -> dict:
    delta = np.asarray(delta, dtype=float).reshape(-1)
    return {
        "mean_abs_prediction_change": float(delta.mean()),
        "median_abs_prediction_change": float(np.median(delta)),
        "p95_abs_prediction_change": float(np.quantile(delta, 0.95)),
        "fraction_nearly_unchanged_lt_1e-6": float((delta < 1e-6).mean()),
    }


def _atomic_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


@torch.no_grad()
def diagnose(
    ckpt_path: Path,
    data_root: Path,
    dataset: str,
    setting: str,
    *,
    max_batches: int = 20,
    batch_size: int = 64,
    output: Path | None = None,
    show_progress: bool = True,
):
    if max_batches <= 0 or batch_size <= 0:
        raise ValueError("max_batches and batch_size must be positive")

    model, ckpt = load_model(ckpt_path, allow_legacy=False)
    if ckpt.get("architecture_version") != ARCHITECTURE_VERSION:
        raise RuntimeError("Diagnostics require a canonical V3 checkpoint")
    if str(ckpt.get("dataset", "")).upper() != str(dataset).upper():
        raise ValueError("checkpoint dataset mismatch")
    if str(ckpt.get("setting", "")).lower() != str(setting).lower():
        raise ValueError("checkpoint setting mismatch")

    config = ckpt["config"]
    pc = config["PROTEIN_ENCODER"]
    _, frame, _ = load_fixed_train_val(dataset, setting, data_root=data_root)

    def _cycle_unique_entities(values, label):
        values = [str(v) for v in values]
        unique = list(dict.fromkeys(values))

        if len(unique) < 2:
            raise RuntimeError(
                f"Need at least two unique {label} entities for perturbation"
            )

        mapping = {
            v: unique[(i + 1) % len(unique)]
            for i, v in enumerate(unique)
        }

        perturbed = [mapping[v] for v in values]

        changed_fraction = float(
            np.mean([a != b for a, b in zip(values, perturbed)])
        )

        if changed_fraction != 1.0:
            raise RuntimeError(
                f"{label} perturbation failed to change every row: "
                f"changed_fraction={changed_fraction}"
            )

        return perturbed, changed_fraction, len(unique)

    protein_frame = frame.copy()
    (
        protein_frame["Target Sequence"],
        protein_changed_fraction,
        n_unique_proteins,
    ) = _cycle_unique_entities(
        frame["Target Sequence"].astype(str).tolist(),
        "protein",
    )

    drug_frame = frame.copy()
    (
        drug_frame["Canonical SMILES"],
        drug_changed_fraction,
        n_unique_drugs,
    ) = _cycle_unique_entities(
        frame["Canonical SMILES"].astype(str).tolist(),
        "drug",
    )

    # Perturbed drugs are selected from the same validation entity set,
    # therefore the original graph cache contains every required molecule.
    graph_cache = build_graph_cache(
        frame["Canonical SMILES"].astype(str).tolist(),
        show_progress=show_progress,
    )

    loader_kwargs = dict(
        graph_cache=graph_cache,
        batch_size=batch_size,
        shuffle=False,
        model_seed=0,
        max_len=pc["max_len"],
        truncation=pc.get("truncation", "right"),
        trim_protein_padding=True,
    )

    base_loader = create_loader_from_frame(
        frame,
        split_name=f"{dataset}/{setting}/val-diagnostic-base",
        **loader_kwargs,
    )

    protein_loader = create_loader_from_frame(
        protein_frame,
        split_name=f"{dataset}/{setting}/val-diagnostic-protein-perturbed",
        **loader_kwargs,
    )

    drug_loader = create_loader_from_frame(
        drug_frame,
        split_name=f"{dataset}/{setting}/val-diagnostic-drug-perturbed",
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    base_all = []
    protein_shuffle_all = []
    drug_shuffle_all = []
    checked = 0

    for i, (base_batch, protein_batch, drug_batch) in enumerate(
        zip(base_loader, protein_loader, drug_loader)
    ):
        if i >= max_batches:
            break

        drug_cpu, protein_cpu, _ = base_batch
        _, protein_perturbed_cpu, _ = protein_batch
        drug_perturbed_cpu, _, _ = drug_batch

        bs = int(protein_cpu.size(0))

        if bs < 1:
            continue

        if int(protein_perturbed_cpu.size(0)) != bs:
            raise RuntimeError(
                "Protein diagnostic loader lost row alignment"
            )

        if int(drug_perturbed_cpu.num_graphs) != bs:
            raise RuntimeError(
                "Drug diagnostic loader lost row alignment"
            )

        drug = drug_cpu.to(device)
        protein = protein_cpu.to(device)
        protein_perturbed = protein_perturbed_cpu.to(device)
        drug_perturbed = drug_perturbed_cpu.to(device)

        base = model(drug, protein)
        protein_shuffled = model(drug, protein_perturbed)
        drug_shuffled = model(drug_perturbed, protein)

        base_all.append(base.detach().cpu().numpy())
        protein_shuffle_all.append(
            protein_shuffled.detach().cpu().numpy()
        )
        drug_shuffle_all.append(
            drug_shuffled.detach().cpu().numpy()
        )

        checked += bs

    if not base_all:
        raise RuntimeError("No diagnostic batches were processed")

    base = np.concatenate(base_all)
    p_shuf = np.concatenate(protein_shuffle_all)
    d_shuf = np.concatenate(drug_shuffle_all)
    p_delta = np.abs(base - p_shuf)
    d_delta = np.abs(base - d_shuf)

    protein_weight_max = {
        name: float(param.detach().abs().max().cpu())
        for name, param in model.protein_encoder.named_parameters()
    }
    drug_weight_max = {
        name: float(param.detach().abs().max().cpu())
        for name, param in model.drug_encoder.named_parameters()
    }

    result = {
        "diagnostic_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(ckpt_path).resolve()),
        "architecture_version": ckpt.get("architecture_version"),
        "dataset": str(dataset).upper(),
        "setting": str(setting).lower(),
        "fusion": str(ckpt.get("fusion_mode", "")).lower(),
        "model_seed": int(ckpt.get("model_seed")),
        "split": "validation",
        "samples_checked": int(checked),
        "perturbation_scheme": "global_unique_entity_cycle",
        "protein_entity_changed_fraction": protein_changed_fraction,
        "drug_entity_changed_fraction": drug_changed_fraction,
        "n_unique_proteins": int(n_unique_proteins),
        "n_unique_drugs": int(n_unique_drugs),
        "protein_shuffle": _summary(p_delta),
        "drug_shuffle": _summary(d_delta),
        "protein_to_drug_sensitivity_ratio": float(
            p_delta.mean() / max(d_delta.mean(), 1e-12)
        ),
        "base_prediction_std": float(np.std(base)),
        "protein_parameter_max_abs": protein_weight_max,
        "drug_parameter_max_abs": drug_weight_max,
        "interpretation_guardrail": (
            "Within-batch modality shuffling is a model-utilization diagnostic only. "
            "It is not a biological negative set and does not establish causal or "
            "binding-site importance."
        ),
    }

    print("=" * 72)
    print("Validation modality-utilization diagnostic")
    print("=" * 72)
    print(f"Checkpoint : {ckpt_path}")
    print(f"Dataset    : {dataset}")
    print(f"Setting    : {setting}")
    print(f"Fusion     : {result['fusion']}")
    print(f"Samples    : {checked}")
    print(
        "Protein shuffle mean |Δprediction| : "
        f"{result['protein_shuffle']['mean_abs_prediction_change']:.6g}"
    )
    print(
        "Drug shuffle mean |Δprediction|    : "
        f"{result['drug_shuffle']['mean_abs_prediction_change']:.6g}"
    )
    print(
        "Protein/drug sensitivity ratio     : "
        f"{result['protein_to_drug_sensitivity_ratio']:.6g}"
    )
    print(
        "Interpretation: near-zero perturbation response warns about modality "
        "collapse. Do not interpret this as biological attribution."
    )

    if output is not None:
        _atomic_json(Path(output).expanduser().resolve(), result)
        print(f"Saved: {Path(output).expanduser().resolve()}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--data", required=True, choices=["kiba", "davis"])
    p.add_argument("--setting", required=True, choices=[str(x).lower() for x in cfg.DATA_CONFIG["settings"]])
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    diagnose(
        args.ckpt.expanduser().resolve(),
        args.data_root.expanduser().resolve(),
        args.data.upper(),
        args.setting,
        max_batches=args.max_batches,
        batch_size=args.batch,
        output=args.output,
        show_progress=not args.no_progress,
    )
