from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.dataset import build_graph_cache, create_loader_from_frame
from data.utils import protein_seq_to_indices


N_ENTITIES = 16

PARAM_EPS = 1e-6
REPR_EPS = 1e-7
PRED_EPS = 1e-7


def _tensor_feature_var(x: torch.Tensor) -> float:
    x = x.detach().float()

    if x.ndim < 2:
        return float(x.var(unbiased=False).item())

    return float(
        x.var(dim=0, unbiased=False).mean().item()
    )


def _weight_norm(module, attr: str) -> float:
    if module is None or not hasattr(module, attr):
        return float("nan")

    obj = getattr(module, attr)
    weight = getattr(obj, "weight", None)

    if weight is None:
        return float("nan")

    return float(
        weight.detach().float().norm().item()
    )


def _pick_unique_proteins(
    frame: pd.DataFrame,
    *,
    max_len: int,
    truncation: str,
    n: int,
):
    seen = set()
    selected = []

    for seq in dict.fromkeys(
        frame["Target Sequence"].astype(str).tolist()
    ):
        encoded = protein_seq_to_indices(
            seq,
            int(max_len),
            truncation,
        )

        key = tuple(encoded)

        if key in seen:
            continue

        seen.add(key)
        selected.append(seq)

        if len(selected) >= n:
            break

    if len(selected) < n:
        raise RuntimeError(
            f"Health gate requires {n} unique encoded proteins, "
            f"found only {len(selected)}"
        )

    return selected


def _pick_unique_drugs(frame: pd.DataFrame, n: int):
    drugs = list(
        dict.fromkeys(
            frame["Canonical SMILES"].astype(str).tolist()
        )
    )

    if len(drugs) < n:
        raise RuntimeError(
            f"Health gate requires {n} unique drugs, "
            f"found only {len(drugs)}"
        )

    return drugs[:n]


def _run_probe(
    model,
    frame,
    *,
    max_len,
    truncation,
    split_name,
):
    graph_cache = build_graph_cache(
        frame["Canonical SMILES"].astype(str).unique().tolist(),
        show_progress=False,
    )

    loader = create_loader_from_frame(
        frame,
        graph_cache=graph_cache,
        batch_size=len(frame),
        shuffle=False,
        model_seed=0,
        max_len=int(max_len),
        truncation=truncation,
        num_workers=0,
        trim_protein_padding=True,
        split_name=split_name,
    )

    drug, protein, _ = next(iter(loader))

    device = next(model.parameters()).device

    drug = drug.to(device)
    protein = protein.to(device)

    model.eval()

    with torch.no_grad():
        aux = model(
            drug,
            protein,
            return_aux=True,
        )

    pred = aux["prediction"].detach().float()

    if not torch.isfinite(pred).all():
        raise FloatingPointError(
            "Health gate saw non-finite predictions"
        )

    for name in (
        "drug_global",
        "protein_global",
        "fused",
    ):
        if not torch.isfinite(aux[name]).all():
            raise FloatingPointError(
                f"Health gate saw non-finite {name}"
            )

    return {
        "prediction_std": float(
            pred.std(unbiased=False).item()
        ),
        "drug_global_var": _tensor_feature_var(
            aux["drug_global"]
        ),
        "protein_global_var": _tensor_feature_var(
            aux["protein_global"]
        ),
        "fused_var": _tensor_feature_var(
            aux["fused"]
        ),
    }


def run_canonical_health_gate(
    model,
    val_frame: pd.DataFrame,
    config: dict,
    validation_predictions,
    *,
    dataset: str,
    setting: str,
    stem: str,
    output_path,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pc = config["PROTEIN_ENCODER"]

    max_len = int(pc["max_len"])
    truncation = pc.get(
        "truncation",
        "right",
    )

    proteins = _pick_unique_proteins(
        val_frame,
        max_len=max_len,
        truncation=truncation,
        n=N_ENTITIES,
    )

    drugs = _pick_unique_drugs(
        val_frame,
        N_ENTITIES,
    )

    fixed_drug = drugs[0]
    fixed_protein = proteins[0]

    protein_probe = pd.DataFrame(
        {
            "Canonical SMILES":
                [fixed_drug] * N_ENTITIES,
            "Target Sequence":
                proteins,
            "Label_norm":
                [0.0] * N_ENTITIES,
        }
    )

    drug_probe = pd.DataFrame(
        {
            "Canonical SMILES":
                drugs,
            "Target Sequence":
                [fixed_protein] * N_ENTITIES,
            "Label_norm":
                [0.0] * N_ENTITIES,
        }
    )

    protein_result = _run_probe(
        model,
        protein_probe,
        max_len=max_len,
        truncation=truncation,
        split_name=(
            f"health/{dataset}/{setting}/protein"
        ),
    )

    drug_result = _run_probe(
        model,
        drug_probe,
        max_len=max_len,
        truncation=truncation,
        split_name=(
            f"health/{dataset}/{setting}/drug"
        ),
    )

    val_pred = np.asarray(
        validation_predictions,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(val_pred).all():
        raise FloatingPointError(
            "Health gate received non-finite "
            "validation predictions"
        )

    val_prediction_std = float(
        np.std(val_pred)
    )

    protein_encoder = getattr(
        model,
        "protein_encoder",
        None,
    )

    drug_encoder = getattr(
        model,
        "drug_encoder",
        None,
    )

    metrics = {
        "stem": stem,
        "dataset": dataset,
        "setting": setting,
        "n_unique_proteins": N_ENTITIES,
        "n_unique_drugs": N_ENTITIES,

        "validation_prediction_std":
            val_prediction_std,

        "protein_prediction_std":
            protein_result["prediction_std"],

        "drug_prediction_std":
            drug_result["prediction_std"],

        "protein_global_var":
            protein_result[
                "protein_global_var"
            ],

        "drug_global_var":
            drug_result[
                "drug_global_var"
            ],

        "protein_probe_fused_var":
            protein_result["fused_var"],

        "drug_probe_fused_var":
            drug_result["fused_var"],

        "protein_embedding_norm":
            _weight_norm(
                protein_encoder,
                "embedding",
            ),

        "protein_token_proj_norm":
            _weight_norm(
                protein_encoder,
                "token_proj",
            ),

        "drug_input_proj_norm":
            _weight_norm(
                drug_encoder,
                "input_proj",
            ),
    }

    failures = []

    non_finite_metrics = [
        name
        for name, value in metrics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
        and not np.isfinite(float(value))
    ]
    if non_finite_metrics:
        failures.append(
            "NON_FINITE_HEALTH_METRIC:"
            + ",".join(sorted(non_finite_metrics))
        )

    if (
        metrics["validation_prediction_std"]
        <= PRED_EPS
    ):
        failures.append(
            "CONSTANT_OUTPUT_COLLAPSE"
        )

    protein_repr_bad = (
        metrics["protein_global_var"]
        <= REPR_EPS
    )

    protein_response_bad = (
        metrics["protein_prediction_std"]
        <= PRED_EPS
    )

    drug_repr_bad = (
        metrics["drug_global_var"]
        <= REPR_EPS
    )

    drug_response_bad = (
        metrics["drug_prediction_std"]
        <= PRED_EPS
    )

    if (
        metrics["protein_embedding_norm"]
        <= PARAM_EPS
        or
        metrics["protein_token_proj_norm"]
        <= PARAM_EPS
    ):
        failures.append(
            "PROTEIN_PARAMETER_COLLAPSE"
        )

    if (
        metrics["drug_input_proj_norm"]
        <= PARAM_EPS
    ):
        failures.append(
            "DRUG_PARAMETER_COLLAPSE"
        )

    if protein_repr_bad:
        failures.append(
            "PROTEIN_REPRESENTATION_COLLAPSE"
        )

    if protein_response_bad:
        failures.append(
            "PROTEIN_RESPONSE_COLLAPSE"
        )

    if drug_repr_bad:
        failures.append(
            "DRUG_REPRESENTATION_COLLAPSE"
        )

    if drug_response_bad:
        failures.append(
            "DRUG_RESPONSE_COLLAPSE"
        )

    if (
        protein_response_bad
        and drug_response_bad
    ):
        failures.append(
            "DUAL_MODALITY_RESPONSE_COLLAPSE"
        )

    result = {
        "status":
            "PASS"
            if not failures
            else "STOP",
        "failures": failures,
        "thresholds": {
            "parameter_eps": PARAM_EPS,
            "representation_eps": REPR_EPS,
            "prediction_eps": PRED_EPS,
        },
        "metrics": metrics,
        "test_policy":
            "validation_only_test_not_loaded",
    }

    tmp = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        output_path,
    )

    print()
    print("=" * 76)
    print("FINAL WHOLE-MODEL HEALTH GATE")
    print("=" * 76)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:32s} = {v:.9e}")
        else:
            print(f"{k:32s} = {v}")

    print(
        "FINAL_HEALTH_GATE                =",
        result["status"],
    )

    print(
        "HELD-OUT TEST                    = SEALED"
    )

    if failures:
        raise RuntimeError(
            "FINAL HEALTH GATE STOP: "
            + ", ".join(failures)
        )

    return result
