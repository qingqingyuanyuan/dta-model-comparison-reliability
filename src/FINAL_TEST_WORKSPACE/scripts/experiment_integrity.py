"""Shared V1C integrity checks for the full prespecified training program.

Held-out test access stays sealed unless the exact frozen Primary/Baseline/
Sensitivity plans are valid under the current protocol/source/config and every
checkpoint matches the provenance of its own plan.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from scripts.protocol_provenance import (
    checkpoint_provenance_matches,
    load_and_validate_plan,
    project_relative,
    resolve_project_path,
)

PRIMARY_PLAN = ROOT / "experiments" / "experiment_plan.json"
BASELINE_PLAN = ROOT / "experiments" / "baseline_plan.json"
SENSITIVITY_PLAN = ROOT / "experiments" / "sensitivity_plan.json"


def trusted_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def expected_primary_rows():
    rows = []
    for dataset, setting, fusion, seed in itertools.product(
        cfg.EXPERIMENT_MATRIX["datasets"],
        cfg.EXPERIMENT_MATRIX["settings"],
        cfg.EXPERIMENT_MATRIX["fusions"],
        cfg.EXPERIMENT_MATRIX["model_seeds"],
    ):
        dataset = str(dataset).upper()
        setting = str(setting).lower()
        fusion = str(fusion).lower()
        seed = int(seed)
        stem = f"{dataset.lower()}_{setting}_{fusion}_seed{seed}"
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "fusion": fusion,
                "seed": seed,
                "stem": stem,
                "checkpoint": project_relative(Path(cfg.MODEL_DIR) / f"{stem}.pt"),
            }
        )
    return rows


def expected_baseline_rows():
    rows = []
    for dataset, setting, baseline, seed in itertools.product(
        cfg.BASELINE_EXPERIMENT_MATRIX["datasets"],
        cfg.BASELINE_EXPERIMENT_MATRIX["settings"],
        cfg.BASELINE_EXPERIMENT_MATRIX["baselines"],
        cfg.BASELINE_EXPERIMENT_MATRIX["model_seeds"],
    ):
        dataset = str(dataset).upper()
        setting = str(setting).lower()
        baseline = str(baseline).lower()
        seed = int(seed)
        stem = f"{baseline}__{dataset.lower()}_{setting}_seed{seed}"
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "baseline": baseline,
                "seed": seed,
                "stem": stem,
                "checkpoint": project_relative(
                    Path(cfg.MODEL_DIR) / "baselines" / f"{stem}.pt"
                ),
            }
        )
    return rows


def expected_sensitivity_rows():
    rows = []
    for dataset, setting, variant, seed in itertools.product(
        cfg.SENSITIVITY_EXPERIMENT_MATRIX["datasets"],
        cfg.SENSITIVITY_EXPERIMENT_MATRIX["settings"],
        cfg.SENSITIVITY_EXPERIMENT_MATRIX["variants"],
        cfg.SENSITIVITY_EXPERIMENT_MATRIX["model_seeds"],
    ):
        dataset = str(dataset).upper()
        setting = str(setting).lower()
        variant = str(variant)
        seed = int(seed)
        spec = cfg.SENSITIVITY_EXPERIMENTS[variant]
        fusion = str(spec["fusion"]).lower()
        stem = f"sensitivity__{variant}__{dataset.lower()}_{setting}_{fusion}_seed{seed}"
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "variant": variant,
                "fusion": fusion,
                "seed": seed,
                "overrides": spec["overrides"],
                "reference": spec["reference"],
                "stem": stem,
                "checkpoint": project_relative(
                    Path(cfg.MODEL_DIR) / "sensitivity" / f"{stem}.pt"
                ),
            }
        )
    return rows


def _validated_frozen_groups():
    specs = {
        "primary": (PRIMARY_PLAN, expected_primary_rows()),
        "controlled_baseline": (BASELINE_PLAN, expected_baseline_rows()),
        "sensitivity": (SENSITIVITY_PLAN, expected_sensitivity_rows()),
    }
    groups = {}
    for family, (plan_path, expected) in specs.items():
        plan, protocol_metadata = load_and_validate_plan(
            plan_path,
            cfg,
            expected_family=family,
            expected_matrix=expected,
        )
        groups[family] = {
            "rows": [dict(row) for row in plan["matrix"]],
            "protocol_metadata": protocol_metadata,
            "plan_path": str(plan_path.resolve()),
        }
    return groups


def checkpoint_matches(row: dict, family: str, protocol_metadata: dict) -> bool:
    path = resolve_project_path(row["checkpoint"])
    if not path.exists():
        return False
    try:
        ck = trusted_load(path)
        if not isinstance(ck, dict):
            return False
        common = bool(
            str(ck.get("dataset", "")).upper() == row["dataset"]
            and str(ck.get("setting", "")).lower() == row["setting"]
            and int(ck.get("model_seed")) == int(row["seed"])
            and isinstance(ck.get("scaler"), dict)
            and ck["scaler"].get("fit_on") == "train_only"
            and ck.get("test_evaluated_during_training") is False
            and checkpoint_provenance_matches(ck, protocol_metadata)
        )
        if not common:
            return False
        extra = ck.get("extra_metadata") or {}

        if family == "primary":
            return bool(
                ck.get("architecture_version") == cfg.MODEL_ARCHITECTURE_VERSION
                and str(ck.get("fusion_mode", "")).lower() == row["fusion"]
                and extra.get("experiment_family", "primary") == "primary"
            )

        if family == "controlled_baseline":
            baseline = row["baseline"]
            return bool(
                ck.get("architecture_version")
                == cfg.BASELINES[baseline]["architecture_version"]
                and str(
                    ck.get("baseline_name") or extra.get("baseline_name") or ""
                ).lower() == baseline
                and extra.get("experiment_family") == "controlled_baseline"
            )

        if family == "sensitivity":
            config = ck.get("config") or {}
            return bool(
                ck.get("architecture_version") == cfg.MODEL_ARCHITECTURE_VERSION
                and str(ck.get("fusion_mode", "")).lower() == row["fusion"]
                and extra.get("experiment_family") == "sensitivity"
                and extra.get("sensitivity_name") == row["variant"]
                and int(config["PROTEIN_ENCODER"]["max_len"])
                == int(row["overrides"]["protein_max_len"])
                and str(config["DRUG_ENCODER"]["pooling"])
                == str(row["overrides"]["drug_pooling"])
            )
        return False
    except Exception:
        return False



PRED_EPS = 1e-7
PARAM_EPS = 1e-6


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_paths(row: dict, family: str):
    stem = row["stem"]

    if family == "controlled_baseline":
        out_dir = Path(cfg.TRAINING_OUTPUT_DIR) / "baselines"
    elif family == "sensitivity":
        out_dir = Path(cfg.TRAINING_OUTPUT_DIR) / "sensitivity"
    else:
        out_dir = Path(cfg.TRAINING_OUTPUT_DIR)

    return {
        "history": out_dir / f"{stem}__history.csv",
        "pred": out_dir / f"{stem}__val_predictions.csv",
        "meta": out_dir / f"{stem}__training.json",
        "health": out_dir / f"{stem}__health.json",
    }


def _read_csv(path: Path):
    if not path.is_file():
        return None, None

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
    except Exception:
        return None, None

    if not rows:
        return None, None

    return rows, fields


def _finite_column(rows, name):
    values = []

    for row in rows:
        try:
            value = float(row[name])
        except Exception:
            return None

        if not math.isfinite(value):
            return None

        values.append(value)

    return values


def _valid_complete_reason(
    row: dict,
    family: str,
    protocol_metadata: dict,
):
    if not checkpoint_matches(row, family, protocol_metadata):
        return "CHECKPOINT_OR_PROVENANCE_INVALID"

    ckpt_path = resolve_project_path(row["checkpoint"])
    paths = _artifact_paths(row, family)

    for name in ("history", "pred", "meta"):
        if not paths[name].is_file():
            return f"MISSING_{name.upper()}"

    try:
        meta = json.loads(
            paths["meta"].read_text(encoding="utf-8")
        )
    except Exception:
        return "INVALID_METADATA_JSON"

    if not isinstance(meta, dict):
        return "INVALID_METADATA"

    if str(meta.get("optimizer", "")).lower() != "adamw":
        return "METADATA_OPTIMIZER_NOT_ADAMW"

    required_prov = (
        "protocol_version",
        "config_sha256",
        "source_sha256",
        "data_manifest_sha256",
        "training_data_sha256",
        "plan_sha256",
    )

    for key in required_prov:
        if str(meta.get(key, "")) != str(
            protocol_metadata.get(key, "")
        ):
            return f"METADATA_{key.upper()}_MISMATCH"

    if meta.get("test_evaluated") is not False:
        return "TEST_EVALUATED_FLAG_INVALID"

    try:
        if str(meta["checkpoint_sha256"]) != _sha256_file(
            ckpt_path
        ):
            return "CHECKPOINT_SHA256_MISMATCH"
    except Exception:
        return "CHECKPOINT_SHA256_INVALID"

    if str(meta.get("dataset", "")).upper() != str(
        row["dataset"]
    ).upper():
        return "METADATA_DATASET_MISMATCH"

    if str(meta.get("setting", "")).lower() != str(
        row["setting"]
    ).lower():
        return "METADATA_SETTING_MISMATCH"

    try:
        if int(meta.get("model_seed")) != int(row["seed"]):
            return "METADATA_SEED_MISMATCH"
    except Exception:
        return "METADATA_SEED_INVALID"

    history_rows, history_fields = _read_csv(paths["history"])
    if history_rows is None:
        return "HISTORY_INVALID"

    required_history = (
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "val_prediction_std",
        "total_grad_norm",
        "protein_grad_norm",
        "drug_grad_norm",
        "protein_param_norm",
        "drug_param_norm",
    )

    if not set(required_history).issubset(
        set(history_fields or [])
    ):
        return "HISTORY_COLUMNS_MISSING"

    history_values = {}

    for name in required_history:
        values = _finite_column(history_rows, name)
        if values is None:
            return f"HISTORY_NON_FINITE_{name.upper()}"
        history_values[name] = values

    try:
        epochs = [int(x) for x in history_values["epoch"]]
        if any(
            abs(float(raw) - int(raw)) > 1e-9
            for raw in history_values["epoch"]
        ):
            return "HISTORY_EPOCH_NON_INTEGER"
    except Exception:
        return "HISTORY_EPOCH_INVALID"

    if len(set(epochs)) != len(epochs):
        return "HISTORY_EPOCH_DUPLICATE"

    if epochs != sorted(epochs):
        return "HISTORY_EPOCH_ORDER_INVALID"

    try:
        best_epoch = int(meta["best_epoch"])
        stop_epoch = int(meta["stop_epoch"])
    except Exception:
        return "BEST_STOP_EPOCH_INVALID"

    if best_epoch not in epochs:
        return "BEST_EPOCH_NOT_IN_HISTORY"

    if stop_epoch != epochs[-1]:
        return "STOP_EPOCH_HISTORY_MISMATCH"

    stop_reason = str(meta.get("stop_reason", "")).strip()
    if not stop_reason or stop_reason == "not_started":
        return "STOP_REASON_INVALID"

    pred_rows, pred_fields = _read_csv(paths["pred"])
    if pred_rows is None:
        return "VAL_PREDICTIONS_INVALID"

    required_pred = (
        "Label_norm",
        "Prediction_norm",
        "Prediction_raw",
    )

    if not set(required_pred).issubset(set(pred_fields or [])):
        return "VAL_PREDICTION_COLUMNS_MISSING"

    for name in required_pred:
        values = _finite_column(pred_rows, name)
        if values is None:
            return f"VAL_PREDICTION_NON_FINITE_{name.upper()}"

    preds = _finite_column(pred_rows, "Prediction_norm")
    if preds is None or len(preds) < 2:
        return "VAL_PREDICTION_COUNT_INVALID"

    mean_pred = sum(preds) / len(preds)
    pred_var = sum(
        (x - mean_pred) ** 2 for x in preds
    ) / len(preds)

    if math.sqrt(pred_var) <= PRED_EPS:
        return "CONSTANT_VALIDATION_PREDICTION"

    if family in ("primary", "sensitivity"):
        if not paths["health"].is_file():
            return "MISSING_HEALTH"

        try:
            health = json.loads(
                paths["health"].read_text(encoding="utf-8")
            )
        except Exception:
            return "INVALID_HEALTH_JSON"

        if not isinstance(health, dict):
            return "INVALID_HEALTH"

        if health.get("status") != "PASS":
            return "HEALTH_GATE_NOT_PASS"

        if health.get("test_policy") != (
            "validation_only_test_not_loaded"
        ):
            return "HEALTH_TEST_POLICY_INVALID"

        health_metrics = health.get("metrics")
        if not isinstance(health_metrics, dict):
            return "HEALTH_METRICS_INVALID"

        for name, value in health_metrics.items():
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    return f"HEALTH_NON_FINITE_{name.upper()}"

        meta_health = meta.get("whole_model_health_gate")
        if not isinstance(meta_health, dict):
            return "METADATA_HEALTH_GATE_MISSING"

        if meta_health.get("status") != "PASS":
            return "METADATA_HEALTH_GATE_NOT_PASS"

    if family == "controlled_baseline":
        if history_values["protein_param_norm"][-1] <= PARAM_EPS:
            return "BASELINE_PROTEIN_PARAMETER_COLLAPSE"

        if history_values["drug_param_norm"][-1] <= PARAM_EPS:
            return "BASELINE_DRUG_PARAMETER_COLLAPSE"

        if max(history_values["protein_grad_norm"]) <= 0.0:
            return "BASELINE_PROTEIN_GRADIENT_ABSENT"

        if max(history_values["drug_grad_norm"]) <= 0.0:
            return "BASELINE_DRUG_GRADIENT_ABSENT"

    return None


def valid_complete(
    row: dict,
    family: str,
    protocol_metadata: dict,
) -> bool:
    return (
        _valid_complete_reason(
            row,
            family,
            protocol_metadata,
        )
        is None
    )


def full_training_status():
    frozen = _validated_frozen_groups()
    status = {}
    missing_all = []

    for family, payload in frozen.items():
        rows = payload["rows"]
        protocol_metadata = payload["protocol_metadata"]

        invalid = {}

        for row in rows:
            reason = _valid_complete_reason(
                row,
                family,
                protocol_metadata,
            )
            if reason is not None:
                invalid[row["stem"]] = reason

        missing = list(invalid)

        status[family] = {
            "expected": len(rows),
            "complete": len(rows) - len(missing),
            "missing": missing,
            "invalid_reasons": invalid,
            "plan_sha256": protocol_metadata["plan_sha256"],
        }

        missing_all.extend(
            f"{family}:{x}"
            for x in missing
        )

    status["overall"] = {
        "expected": sum(
            status[f]["expected"]
            for f in frozen
        ),
        "complete": sum(
            status[f]["complete"]
            for f in frozen
        ),
        "all_complete": len(missing_all) == 0,
        "missing": missing_all,
    }

    return status


def require_full_training_complete():
    # Any missing/invalid plan raises before checkpoint counting, which keeps
    # held-out test sealed by default.
    status = full_training_status()
    if not status["overall"]["all_complete"]:
        missing = status["overall"]["missing"]
        expected = int(status["overall"]["expected"])
        raise RuntimeError(
            f"Held-out test remains sealed because the full prespecified {expected}-run "
            "training program is incomplete. Missing/invalid checkpoints:\n  "
            + "\n  ".join(missing[:30])
            + ("\n  ..." if len(missing) > 30 else "")
        )
    return status
