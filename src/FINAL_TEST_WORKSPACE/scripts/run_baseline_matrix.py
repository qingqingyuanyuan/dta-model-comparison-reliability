#!/usr/bin/env python3
"""Plan/train/evaluate the prespecified controlled baseline matrix.

Matrix: 2 datasets x 3 settings x 2 baseline families x 3 paired seeds = 36.
Training uses train+validation only. Held-out test is globally unlocked only after
all 126 prespecified primary + baseline + sensitivity checkpoints are complete and frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
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
from scripts.experiment_integrity import require_full_training_complete
from scripts.protocol_provenance import (
    build_plan_header,
    checkpoint_provenance_matches,
    load_and_validate_plan,
    project_relative,
    resolve_project_path,
    require_canonical_data_root,
)


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_seed(seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(cfg.TRAINING.get("deterministic", True))
        torch.backends.cudnn.benchmark = False


def _trusted_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_matrix():
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


def _checkpoint_matches(row, protocol_metadata):
    path = resolve_project_path(row["checkpoint"])
    if not path.exists():
        return False
    try:
        ck = _trusted_load(path)
        expected_arch = cfg.BASELINES[row["baseline"]]["architecture_version"]
        extra = ck.get("extra_metadata") or {}
        return bool(
            ck.get("architecture_version") == expected_arch
            and str(ck.get("dataset", "")).upper() == row["dataset"]
            and str(ck.get("setting", "")).lower() == row["setting"]
            and int(ck.get("model_seed")) == row["seed"]
            and str(extra.get("baseline_name", "")).lower() == row["baseline"]
            and ck.get("test_evaluated_during_training") is False
            and checkpoint_provenance_matches(ck, protocol_metadata)
        )
    except Exception:
        return False


def _write_plan(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **build_plan_header(cfg, family="controlled_baseline"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Controlled reference baselines on the same fixed splits/scalers. "
            "Names ending in -style are not exact external-repository reproductions."
        ),
        "matrix": rows,
        "n_runs": len(rows),
        "baseline_configs": cfg.BASELINES,
        "training_protocol": cfg.TRAINING,
        "test_policy": (
            "Held-out test remains sealed until the full prespecified 126-run "
            "primary + controlled-baseline + sensitivity training program is frozen."
        ),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _prepare_loaders(row, bc, args):
    # Lazy imports keep --phase plan usable on a CPU/planning host that does
    # not have PyG installed. Actual training still requires the real runtime.
    from baselines.dataset import create_fixed_sequence_train_val_loaders
    from data.dataset import create_fixed_train_val_loaders

    if bc["input_modality"] == "smiles_sequence+protein_sequence":
        return create_fixed_sequence_train_val_loaders(
            row["dataset"],
            row["setting"],
            data_root=Path(args.data_root),
            batch_size=int(cfg.TRAINING["batch_size"]),
            model_seed=row["seed"],
            max_smiles_len=int(bc["max_smiles_len"]),
            max_protein_len=int(bc["max_protein_len"]),
            protein_truncation="right",
            num_workers=args.num_workers,
        )
    if bc["input_modality"] == "molecular_graph+protein_sequence":
        return create_fixed_train_val_loaders(
            row["dataset"],
            row["setting"],
            data_root=Path(args.data_root),
            batch_size=int(cfg.TRAINING["batch_size"]),
            model_seed=row["seed"],
            max_len=int(bc["max_protein_len"]),
            truncation="right",
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            trim_protein_padding=True,
        )
    raise ValueError(f"Unsupported modality {bc['input_modality']}")


def _train_one(row, args, protocol_metadata):
    # Training-only imports are deliberately deferred so experiment plans can
    # be frozen before provisioning the GPU/PyG worker.
    from baselines.dataset import sequence_truncation_report
    from baselines.registry import build_baseline_model, get_baseline_config
    from data.utils import denormalize_affinity, load_fixed_train_val
    from train.evaluate import Evaluator, METRIC_SCHEMA_VERSION
    from train.trainer import Trainer

    ckpt_path = resolve_project_path(row["checkpoint"])
    out_dir = Path(cfg.TRAINING_OUTPUT_DIR) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / f"{row['stem']}__history.csv"
    val_pred_path = out_dir / f"{row['stem']}__val_predictions.csv"
    meta_path = out_dir / f"{row['stem']}__training.json"
    existing = [p for p in (ckpt_path, history_path, val_pred_path, meta_path) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Baseline training output exists:\n  " + "\n  ".join(str(x) for x in existing)
        )

    train_df, val_df, scaler = load_fixed_train_val(
        row["dataset"], row["setting"], data_root=Path(args.data_root)
    )
    config = get_baseline_config(row["baseline"])
    config["TRAINING"]["seed"] = row["seed"]
    config["TRAINING"]["device"] = args.device
    bc = config["BASELINE"]

    _set_seed(row["seed"])
    train_loader, val_loader, scaler2 = _prepare_loaders(row, bc, args)
    if scaler2 != scaler:
        raise RuntimeError("baseline scaler mismatch")

    model = build_baseline_model(row["baseline"], config)
    params = model.parameter_summary()
    trainer = Trainer(model, config, device=None if args.device == "auto" else args.device)

    print("=" * 76)
    print("CONTROLLED BASELINE TRAINING — TEST SEALED")
    print(f"Baseline : {row['baseline']}")
    print(f"Dataset  : {row['dataset']}")
    print(f"Setting  : {row['setting']}")
    print(f"Seed     : {row['seed']}")
    print(f"Rows     : train={len(train_df):,}, val={len(val_df):,}")
    print(f"Params   : {params['total']:,}")
    print("=" * 76)

    start = time.time()

    last_state_path = ckpt_path.with_suffix(
        ckpt_path.suffix + ".last_state"
    )

    if args.overwrite and last_state_path.exists():
        last_state_path.unlink()
        print(
            "OVERWRITE: removed existing LAST_STATE "
            f"{last_state_path}"
        )

    history = trainer.fit(
        train_loader,
        val_loader,
        last_state_path=last_state_path,
        protocol_metadata=protocol_metadata,
        resume_key=row["stem"],
    )
    val_loss, pred, true = trainer.evaluate(val_loader)
    metrics = Evaluator(scaler=scaler).evaluate(true, pred)
    if not np.isclose(val_loss, metrics["mse_norm"], atol=1e-9, rtol=1e-6):
        raise RuntimeError("baseline Trainer val MSE mismatch")

    val_out = val_df.copy()
    if len(val_out) != len(pred) or not np.allclose(
        val_out["Label_norm"].to_numpy(float), true
    ):
        raise RuntimeError("baseline validation loader order mismatch")
    val_out["Prediction_norm"] = pred
    val_out["Prediction_raw"] = denormalize_affinity(pred, scaler)
    val_out.to_csv(val_pred_path, index=False)
    history_columns = [
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
        "protein_embedding_norm",
        "protein_token_proj_norm",
        "drug_input_proj_norm",
    ]
    pd.DataFrame(
        {
            k: history[k]
            for k in history_columns
            if k in history
        }
    ).to_csv(history_path, index=False)

    truncation = sequence_truncation_report(
        pd.concat([train_df, val_df], ignore_index=True),
        max_smiles_len=int(bc.get("max_smiles_len", 10**9)),
        max_protein_len=int(bc["max_protein_len"]),
    )

    split_dir = Path(args.data_root) / row["dataset"] / row["setting"]
    provenance = {
        "train": {"path": str((split_dir / "train.csv").resolve()), "sha256": _sha256(split_dir / "train.csv")},
        "val": {"path": str((split_dir / "val.csv").resolve()), "sha256": _sha256(split_dir / "val.csv")},
        "scaler": {"path": str((split_dir / "scaler.json").resolve()), "sha256": _sha256(split_dir / "scaler.json")},
        "test_policy": {"path": str((split_dir / "test.csv").resolve()), "status": "sealed_not_read_or_hashed_during_training"},
    }
    split_meta_path = split_dir / "split_metadata.json"
    split_metadata = json.loads(split_meta_path.read_text(encoding="utf-8"))

    trainer.save(
        ckpt_path,
        scaler=scaler,
        history=history,
        dataset=row["dataset"],
        setting=row["setting"],
        model_seed=row["seed"],
        split_metadata=split_metadata,
        data_provenance=provenance,
        validation_metrics=metrics,
        protocol_metadata=protocol_metadata,
        extra_metadata={
            "run_kind": "final_controlled_baseline",
            "experiment_family": "controlled_baseline",
            "baseline_name": row["baseline"],
            "baseline_description": bc.get("description"),
            "input_modality": bc["input_modality"],
            "reporting_label": "style reimplementation; not exact published-repository reproduction",
            "truncation_audit_train_plus_val": truncation,
            "history_file": str(history_path.resolve()),
            "validation_prediction_file": str(val_pred_path.resolve()),
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "test_policy": "sealed_not_evaluated_during_training",
        },
    )

    # Registry expects a recognized name. Store a top-level copy too.
    ck = _trusted_load(ckpt_path)
    ck["baseline_name"] = row["baseline"]
    tmp_ckpt = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
    torch.save(ck, tmp_ckpt)
    os.replace(tmp_ckpt, ckpt_path)

    run_meta = {
        "experiment_family": "controlled_baseline",
        "baseline_name": row["baseline"],
        "architecture_version": bc["architecture_version"],
        "protocol_version": protocol_metadata["protocol_version"],
        "optimizer": protocol_metadata["optimizer"],
        "config_sha256": protocol_metadata["config_sha256"],
        "source_sha256": protocol_metadata["source_sha256"],
        "data_manifest_sha256": protocol_metadata["data_manifest_sha256"],
        "training_data_sha256": protocol_metadata["training_data_sha256"],
        "plan_sha256": protocol_metadata["plan_sha256"],
        "plan_path": protocol_metadata["plan_path"],
        "dataset": row["dataset"],
        "setting": row["setting"],
        "model_seed": row["seed"],
        "validation_metrics": metrics,
        "best_epoch": int(history["best_epoch"]),
        "stop_epoch": int(history["stop_epoch"]),
        "stop_reason": str(history["stop_reason"]),
        "history_file": str(history_path.resolve()),
        "validation_prediction_file": str(val_pred_path.resolve()),
        "parameter_summary": params,
        "truncation_audit_train_plus_val": truncation,
        "elapsed_seconds": float(time.time() - start),
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": _sha256(ckpt_path),
        "test_evaluated": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    tmp = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, meta_path)


def _run_command(cmd, dry_run):
    print("\n$", " ".join(str(x) for x in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["plan", "train", "evaluate"], default="plan")
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--unlock-test", default=None)
    p.add_argument(
        "--plan-file",
        type=Path,
        default=ROOT / "experiments" / "baseline_plan.json",
    )
    args = p.parse_args()

    if args.phase != "plan":
        require_canonical_data_root(args.data_root)

    if args.phase == "plan":
        rows = build_matrix()
        _write_plan(rows, args.plan_file)
        print(f"Baseline plan: {args.plan_file}")
        print(f"Runs: {len(rows)}")
        return

    expected_rows = build_matrix()
    plan, protocol_metadata = load_and_validate_plan(
        args.plan_file,
        cfg,
        expected_family="controlled_baseline",
        expected_matrix=expected_rows,
    )
    rows = [dict(row) for row in plan["matrix"]]
    print(f"Frozen baseline plan: {args.plan_file} ({len(rows)} runs)")
    print(f"Plan SHA256: {protocol_metadata['plan_sha256']}")

    if args.phase == "train":
        for i, row in enumerate(rows, 1):
            print(f"\n[{i}/{len(rows)}] {row['stem']}")
            if _checkpoint_matches(row, protocol_metadata) and not args.overwrite:
                print("SKIP: valid V1C-bound baseline checkpoint exists")
                continue
            try:
                if args.dry_run:
                    print("DRY RUN training")
                else:
                    _train_one(row, args, protocol_metadata)
            except Exception:
                if not args.continue_on_error:
                    raise
                import traceback
                traceback.print_exc()
        return

    if args.unlock_test != "FINAL_BASELINE_TEST_PHASE":
        raise RuntimeError(
            "Baseline held-out test is locked. Use exactly "
            "--unlock-test FINAL_BASELINE_TEST_PHASE after the full 126-run training program is frozen."
        )
    require_full_training_complete()

    for i, row in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] TEST {row['stem']}")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_baseline.py"),
            "--data", row["dataset"].lower(),
            "--setting", row["setting"],
            "--ckpt", str(resolve_project_path(row["checkpoint"])),
            "--data-root", str(args.data_root),
            "--batch", str(cfg.TRAINING["batch_size"]),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        if args.overwrite:
            cmd.append("--overwrite")
        code = _run_command(cmd, args.dry_run)
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
