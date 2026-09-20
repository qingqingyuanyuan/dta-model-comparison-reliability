#!/usr/bin/env python3
"""ZhiYao-Graph canonical entry point.

Training uses only the immutable train/validation splits. The held-out test set
is sealed and can be accessed only through an explicit evaluation phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np

import config as cfg


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_parser():
    p = argparse.ArgumentParser(description="ZhiYao-Graph continuous DTA system")
    p.add_argument(
        "--mode",
        default="app",
        choices=["app", "train", "predict", "download", "evaluate"],
    )
    p.add_argument("--data", default="kiba", choices=["kiba", "davis"])
    p.add_argument(
        "--setting", default="warm", choices=[str(x).lower() for x in cfg.DATA_CONFIG["settings"]]
    )
    p.add_argument(
        "--fusion",
        default="concat",
        choices=["concat", "product", "attention"],
    )
    p.add_argument("--data-root", default="data_final")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--run-tag", default=None)
    p.add_argument(
        "--sensitivity-name",
        default=None,
        help=(
            "Prespecified sensitivity variant from config.SENSITIVITY_EXPERIMENTS. "
            "Sensitivity runs are isolated from primary checkpoints."
        ),
    )
    p.add_argument(
        "--protein-max-len",
        type=int,
        default=None,
        help="Sensitivity-only protein max_len override",
    )
    p.add_argument(
        "--drug-pooling",
        choices=["add", "mean"],
        default=None,
        help="Sensitivity-only drug graph pooling override",
    )
    p.add_argument(
        "--final-run",
        action="store_true",
        help=(
            "Use canonical paper-run filename/protocol. Without this flag, "
            "training outputs are marked DEV so a smoke run cannot overwrite a final run."
        ),
    )
    p.add_argument(
        "--frozen-plan",
        type=Path,
        default=None,
        help="Required for --final-run; exact frozen plan artifact for provenance binding",
    )
    p.add_argument(
        "--allow-protocol-override",
        action="store_true",
        help="DEBUG ONLY: allow final-run epochs/batch/seed outside the frozen protocol",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--confirm-test-evaluation",
        action="store_true",
        help="Required for --mode evaluate; test evaluation is a deliberate final phase",
    )
    p.add_argument("--no-progress", action="store_true")
    return p


def _set_model_seed(seed: int, deterministic: bool = True):
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False


def _validate_training_files(data_root: Path, dataset: str, setting: str):
    """Validate training-phase artifacts without reading held-out test labels."""
    split_dir = data_root / dataset / setting
    required = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "scaler": split_dir / "scaler.json",
        "split_metadata": split_dir / "split_metadata.json",
    }
    # The test file must exist as part of the frozen dataset, but training does
    # not hash/read its contents here.
    test_path = split_dir / "test.csv"
    missing = [str(x) for x in [*required.values(), test_path] if not x.exists()]
    if missing:
        raise FileNotFoundError("Fixed data are incomplete:\n  " + "\n  ".join(missing))

    scaler = _read_json(required["scaler"])
    if scaler.get("fit_on") != "train_only":
        raise ValueError("Final scaler must be fit_on='train_only'")

    split_metadata = _read_json(required["split_metadata"])
    provenance = {
        key: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for key, path in required.items()
    }
    manifest = data_root / "manifest.json"
    if manifest.exists():
        provenance["manifest"] = {
            "path": str(manifest.resolve()),
            "sha256": _sha256(manifest),
        }
    provenance["test_policy"] = {
        "path": str(test_path.resolve()),
        "status": "sealed_not_read_or_hashed_during_training",
    }
    return scaler, split_metadata, provenance


def _run_stem(
    dataset,
    setting,
    fusion,
    seed,
    *,
    final_run,
    epochs,
    tag=None,
    sensitivity_name=None,
):
    base = f"{dataset.lower()}_{setting}_{fusion}_seed{int(seed)}"
    if sensitivity_name:
        safe_sens = "".join(
            ch if ch.isalnum() or ch in "-_" else "_"
            for ch in str(sensitivity_name)
        )
        base = f"sensitivity__{safe_sens}__{base}"
    if final_run:
        return base
    suffix = tag or f"DEV_E{int(epochs)}"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in suffix)
    return f"{base}__{safe}"


def _assert_final_protocol(args, cfg, epochs: int, batch: int):
    """Validate frozen primary or prespecified sensitivity protocol."""
    if not args.final_run:
        return
    if args.allow_protocol_override:
        raise ValueError(
            "--allow-protocol-override cannot be combined with --final-run after V1C hardening. "
            "Use a DEV run instead of writing a noncanonical formal checkpoint."
        )
    if args.frozen_plan is None:
        raise ValueError("--final-run requires --frozen-plan after V1C hardening")

    expected_epochs = int(cfg.TRAINING["num_epochs"])
    expected_batch = int(cfg.TRAINING["batch_size"])
    expected_seeds = {int(x) for x in cfg.EXPERIMENT_MATRIX["model_seeds"]}
    problems = []
    if epochs != expected_epochs:
        problems.append(f"epochs={epochs} != frozen {expected_epochs}")
    if batch != expected_batch:
        problems.append(f"batch={batch} != frozen {expected_batch}")
    if int(args.model_seed) not in expected_seeds:
        problems.append(
            f"model_seed={args.model_seed} not in frozen seeds {sorted(expected_seeds)}"
        )

    sensitivity = args.sensitivity_name
    if sensitivity is None:
        if args.protein_max_len is not None or args.drug_pooling is not None:
            problems.append(
                "primary final run cannot use --protein-max-len/--drug-pooling overrides"
            )
    else:
        if sensitivity not in cfg.SENSITIVITY_EXPERIMENTS:
            problems.append(
                f"unknown sensitivity_name={sensitivity!r}; "
                f"allowed={sorted(cfg.SENSITIVITY_EXPERIMENTS)}"
            )
        else:
            spec = cfg.SENSITIVITY_EXPERIMENTS[sensitivity]
            if str(args.fusion).lower() != str(spec["fusion"]).lower():
                problems.append(
                    f"sensitivity {sensitivity} requires fusion={spec['fusion']}"
                )
            ov = spec["overrides"]
            requested_max_len = (
                int(args.protein_max_len)
                if args.protein_max_len is not None
                else int(cfg.PROTEIN_ENCODER["max_len"])
            )
            requested_pooling = (
                str(args.drug_pooling)
                if args.drug_pooling is not None
                else str(cfg.DRUG_ENCODER["pooling"])
            )
            if requested_max_len != int(ov["protein_max_len"]):
                problems.append(
                    f"sensitivity {sensitivity} requires protein_max_len="
                    f"{ov['protein_max_len']}, got {requested_max_len}"
                )
            if requested_pooling != str(ov["drug_pooling"]):
                problems.append(
                    f"sensitivity {sensitivity} requires drug_pooling="
                    f"{ov['drug_pooling']}, got {requested_pooling}"
                )

    if problems:
        raise ValueError(
            "Final-run protocol mismatch:\n  " + "\n  ".join(problems)
            + "\nUse --allow-protocol-override only for an explicitly documented development run."
        )



def _load_final_plan_context(args, cfg, dataset: str, setting: str):
    """Bind one formal run to the exact frozen plan artifact."""
    from scripts.protocol_provenance import (
        load_and_validate_plan,
        project_relative,
    )

    family = "sensitivity" if args.sensitivity_name else "primary"
    expected_n = 36 if family == "sensitivity" else 54
    plan_path = args.frozen_plan.expanduser().resolve()
    plan, context = load_and_validate_plan(
        plan_path,
        cfg,
        expected_family=family,
        expected_matrix=None,
    )
    if int(plan.get("n_runs", -1)) != expected_n:
        raise RuntimeError(
            f"FROZEN_PLAN_MISMATCH — STOP: {family} plan must contain {expected_n} runs"
        )

    if family == "primary":
        expected_checkpoint = project_relative(
            Path(cfg.MODEL_DIR) / f"{dataset.lower()}_{setting}_{args.fusion}_seed{int(args.model_seed)}.pt"
        )
        candidates = [
            row for row in plan["matrix"]
            if str(row.get("dataset", "")).upper() == dataset
            and str(row.get("setting", "")).lower() == setting
            and str(row.get("fusion", "")).lower() == str(args.fusion).lower()
            and int(row.get("seed", -1)) == int(args.model_seed)
        ]
    else:
        spec = cfg.SENSITIVITY_EXPERIMENTS[args.sensitivity_name]
        expected_checkpoint = project_relative(
            Path(cfg.MODEL_DIR) / "sensitivity" /
            f"sensitivity__{args.sensitivity_name}__{dataset.lower()}_{setting}_{args.fusion}_seed{int(args.model_seed)}.pt"
        )
        candidates = [
            row for row in plan["matrix"]
            if str(row.get("dataset", "")).upper() == dataset
            and str(row.get("setting", "")).lower() == setting
            and str(row.get("fusion", "")).lower() == str(args.fusion).lower()
            and str(row.get("variant", "")) == str(args.sensitivity_name)
            and int(row.get("seed", -1)) == int(args.model_seed)
        ]
        if candidates:
            if candidates[0].get("overrides") != spec["overrides"]:
                raise RuntimeError("FROZEN_PLAN_MISMATCH — STOP: sensitivity overrides changed")

    if len(candidates) != 1:
        raise RuntimeError(
            "FROZEN_PLAN_MISMATCH — STOP: requested final run is not uniquely present in frozen matrix"
        )
    if str(candidates[0].get("checkpoint", "")) != expected_checkpoint:
        raise RuntimeError(
            "FROZEN_PLAN_MISMATCH — STOP: frozen checkpoint path is not canonical/project-relative"
        )
    return context

def main():
    args = _build_parser().parse_args()

    if args.mode == "app":
        os.system("streamlit run app/app.py")
        return
    if args.mode == "train":
        _run_train(args)
        return
    if args.mode == "predict":
        from scripts.predict import run_prediction

        run_prediction(ckpt_path=args.ckpt)
        return
    if args.mode == "download":
        from scripts.download_data import download_datasets

        download_datasets()
        return
    if args.mode == "evaluate":
        if args.ckpt is None:
            raise ValueError("--mode evaluate requires --ckpt")
        if not args.confirm_test_evaluation:
            raise RuntimeError(
                "Held-out test access is sealed. Re-run with "
                "--confirm-test-evaluation only after the training matrix is frozen."
            )
        import config as cfg
        from scripts.evaluate_model import evaluate_checkpoint

        evaluate_checkpoint(
            dataset_name=args.data,
            setting=args.setting,
            ckpt_path=args.ckpt.expanduser().resolve(),
            data_root=Path(args.data_root).expanduser().resolve(),
            batch_size=args.batch or int(cfg.TRAINING["batch_size"]),
            num_workers=args.num_workers,
            device_name=args.device,
            output_dir=Path(cfg.EVALUATION_OUTPUT_DIR),
            show_progress=not args.no_progress,
            overwrite=args.overwrite,
        )


def _run_train(args):
    import config as cfg

    epochs = int(args.epochs if args.epochs is not None else cfg.TRAINING["num_epochs"])
    batch = int(args.batch if args.batch is not None else cfg.TRAINING["batch_size"])
    if epochs <= 0 or batch <= 0:
        raise ValueError("epochs and batch must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be >= 0")

    _assert_final_protocol(args, cfg, epochs, batch)

    dataset = args.data.upper()
    setting = args.setting.lower()
    protocol_metadata = (
        _load_final_plan_context(args, cfg, dataset, setting)
        if args.final_run
        else None
    )

    import pandas as pd
    import torch
    from data.dataset import create_fixed_train_val_loaders
    from data.utils import denormalize_affinity, load_fixed_train_val
    from models.predictor import DTAPredictor
    from train.evaluate import Evaluator, METRIC_SCHEMA_VERSION
    from train.trainer import Trainer

    data_root = Path(args.data_root).expanduser().resolve()
    scaler, split_metadata, data_provenance = _validate_training_files(
        data_root, dataset, setting
    )

    _set_model_seed(
        args.model_seed,
        deterministic=bool(cfg.TRAINING.get("deterministic", True)),
    )

    train_df, val_df, scaler_again = load_fixed_train_val(
        dataset, setting, data_root=data_root
    )
    if scaler_again != scaler:
        raise RuntimeError("Scaler mismatch between training loaders")

    train_loader, val_loader, _ = create_fixed_train_val_loaders(
        dataset_name=dataset,
        setting=setting,
        data_root=data_root,
        batch_size=batch,
        model_seed=args.model_seed,
        max_len=(
            int(args.protein_max_len)
            if args.protein_max_len is not None
            else int(cfg.PROTEIN_ENCODER["max_len"])
        ),
        truncation=cfg.PROTEIN_ENCODER.get("truncation", "right"),
        num_workers=args.num_workers,
        show_progress=not args.no_progress,
        trim_protein_padding=True,
    )

    config_dict = {
        "DRUG_ENCODER": dict(cfg.DRUG_ENCODER),
        "PROTEIN_ENCODER": dict(cfg.PROTEIN_ENCODER),
        "FUSION": dict(cfg.FUSION),
        "PREDICTOR": dict(cfg.PREDICTOR),
        "TRAINING": dict(cfg.TRAINING),
    }
    config_dict["FUSION"]["fusion_mode"] = args.fusion
    if args.protein_max_len is not None:
        if args.sensitivity_name is None:
            raise ValueError("--protein-max-len requires --sensitivity-name")
        if int(args.protein_max_len) <= 0:
            raise ValueError("--protein-max-len must be positive")
        config_dict["PROTEIN_ENCODER"]["max_len"] = int(args.protein_max_len)
    if args.drug_pooling is not None:
        if args.sensitivity_name is None:
            raise ValueError("--drug-pooling requires --sensitivity-name")
        config_dict["DRUG_ENCODER"]["pooling"] = str(args.drug_pooling)
    config_dict["TRAINING"]["num_epochs"] = epochs
    config_dict["TRAINING"]["batch_size"] = batch
    config_dict["TRAINING"]["seed"] = int(args.model_seed)
    config_dict["TRAINING"]["device"] = args.device

    stem = _run_stem(
        dataset,
        setting,
        args.fusion,
        args.model_seed,
        final_run=args.final_run,
        epochs=epochs,
        tag=args.run_tag,
        sensitivity_name=args.sensitivity_name,
    )
    if args.sensitivity_name:
        checkpoint_dir = Path(cfg.MODEL_DIR) / "sensitivity"
        training_dir = Path(cfg.TRAINING_OUTPUT_DIR) / "sensitivity"
    else:
        checkpoint_dir = Path(cfg.MODEL_DIR)
        training_dir = Path(cfg.TRAINING_OUTPUT_DIR)
    ckpt_path = checkpoint_dir / f"{stem}.pt"
    history_path = training_dir / f"{stem}__history.csv"
    val_pred_path = training_dir / f"{stem}__val_predictions.csv"
    run_meta_path = training_dir / f"{stem}__training.json"

    existing = [p for p in (ckpt_path, history_path, val_pred_path, run_meta_path) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Training output already exists; refusing silent overwrite:\n  "
            + "\n  ".join(str(p) for p in existing)
        )
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 76)
    print("ZhiYao-Graph sealed-test training")
    print(
        f"Run kind : {'FINAL-SENSITIVITY' if args.final_run and args.sensitivity_name else 'FINAL' if args.final_run else 'DEV/SMOKE'}"
    )
    if args.sensitivity_name:
        print(f"Sensitivity: {args.sensitivity_name}")
    print(f"Dataset  : {dataset}")
    print(f"Setting  : {setting}")
    print(f"Fusion   : {args.fusion}")
    print(f"Seed     : {args.model_seed}")
    print(f"Rows     : train={len(train_df):,}, val={len(val_df):,}")
    print("Test     : SEALED (not loaded during training)")
    print(f"Metric   : {METRIC_SCHEMA_VERSION}")
    print("=" * 76)

    start_time = time.time()
    model = DTAPredictor(config_dict)
    parameter_summary = model.parameter_summary()
    print("Parameter summary:", parameter_summary)

    trainer = Trainer(model, config_dict, device=args.device if args.device != "auto" else None)

    # Formal runs automatically resume only from their own
    # provenance-bound LAST_STATE. DEV/smoke runs keep the
    # historical non-resume behavior.
    last_state_path = (
        ckpt_path.with_suffix(
            ckpt_path.suffix + ".last_state"
        )
        if args.final_run
        else None
    )

    if (
        args.overwrite
        and last_state_path is not None
        and last_state_path.exists()
    ):
        last_state_path.unlink()
        print(
            "OVERWRITE: removed existing LAST_STATE "
            f"{last_state_path}"
        )

    history = trainer.fit(
        train_loader,
        val_loader,
        last_state_path=last_state_path,
        protocol_metadata=(
            protocol_metadata
            if args.final_run
            else None
        ),
        resume_key=(
            stem if args.final_run else None
        ),
    )

    # Validation-only audit after restoring the best state.
    val_loss, y_pred_norm, y_true_norm = trainer.evaluate(val_loader)
    evaluator = Evaluator(scaler=scaler)
    validation_metrics = evaluator.evaluate(y_true_norm, y_pred_norm)

    # Validation-only whole-model health gate on the restored
    # best-validation state. Held-out test is never loaded here.
    from scripts.runtime_health_gate import run_canonical_health_gate

    health_path = training_dir / f"{stem}__health.json"

    health_result = run_canonical_health_gate(
        model,
        val_df,
        config_dict,
        y_pred_norm,
        dataset=dataset,
        setting=setting,
        stem=stem,
        output_path=health_path,
    )

    if not np.isclose(val_loss, validation_metrics["mse_norm"], atol=1e-9, rtol=1e-6):
        raise RuntimeError("Trainer validation loss != standardized validation MSE")
    evaluator.print_metrics(validation_metrics)

    val_pred_df = val_df.copy()
    if len(val_pred_df) != len(y_pred_norm):
        raise RuntimeError("Validation prediction count mismatch")
    if not np.allclose(val_pred_df["Label_norm"].to_numpy(float), y_true_norm):
        raise RuntimeError("Validation loader order does not match val.csv")
    val_pred_df["Prediction_norm"] = y_pred_norm
    val_pred_df["Prediction_raw"] = denormalize_affinity(y_pred_norm, scaler)
    val_pred_df.to_csv(val_pred_path, index=False)

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

    trainer.save(
        ckpt_path,
        scaler=scaler,
        history=history,
        fusion_mode=args.fusion,
        edge_dim=config_dict["DRUG_ENCODER"].get("edge_dim"),
        dataset=dataset,
        setting=setting,
        model_seed=args.model_seed,
        split_metadata=split_metadata,
        data_provenance=data_provenance,
        validation_metrics=validation_metrics,
        protocol_metadata=protocol_metadata,
        extra_metadata={
            "run_kind": (
                "final_sensitivity"
                if args.final_run and args.sensitivity_name
                else "final" if args.final_run else "dev"
            ),
            "experiment_family": "sensitivity" if args.sensitivity_name else "primary",
            "sensitivity_name": args.sensitivity_name,
            "sensitivity_overrides": (
                {
                    "protein_max_len": config_dict["PROTEIN_ENCODER"]["max_len"],
                    "drug_pooling": config_dict["DRUG_ENCODER"]["pooling"],
                }
                if args.sensitivity_name
                else None
            ),
            "history_file": str(history_path.resolve()),
            "validation_prediction_file": str(val_pred_path.resolve()),
            "data_pipeline": "data_final_fixed_keep_grouped_v2",
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "test_policy": "sealed_not_evaluated_during_training",
            "whole_model_health_gate": health_result,
            "whole_model_health_file": str(health_path.resolve()),
            "protein_encoding": {
                "vocab_size": config_dict["PROTEIN_ENCODER"]["vocab_size"],
                "padding_idx": config_dict["PROTEIN_ENCODER"].get("padding_idx", 0),
                "unk_idx": config_dict["PROTEIN_ENCODER"].get("unk_idx", 21),
                "max_len": config_dict["PROTEIN_ENCODER"]["max_len"],
                "truncation": config_dict["PROTEIN_ENCODER"].get("truncation", "right"),
                "dynamic_batch_padding_trim": True,
            },
        },
    )

    elapsed = float(time.time() - start_time)
    run_meta = {
        "run_kind": (
            "final_sensitivity"
            if args.final_run and args.sensitivity_name
            else "final" if args.final_run else "dev"
        ),
        "experiment_family": "sensitivity" if args.sensitivity_name else "primary",
        "sensitivity_name": args.sensitivity_name,
        "sensitivity_overrides": (
            {
                "protein_max_len": config_dict["PROTEIN_ENCODER"]["max_len"],
                "drug_pooling": config_dict["DRUG_ENCODER"]["pooling"],
            }
            if args.sensitivity_name
            else None
        ),
        "architecture_version": cfg.MODEL_ARCHITECTURE_VERSION,
        "protocol_version": protocol_metadata.get("protocol_version") if protocol_metadata else None,
        "optimizer": str(config_dict["TRAINING"].get("optimizer", "")).lower(),
        "config_sha256": protocol_metadata.get("config_sha256") if protocol_metadata else None,
        "source_sha256": protocol_metadata.get("source_sha256") if protocol_metadata else None,
        "data_manifest_sha256": protocol_metadata.get("data_manifest_sha256") if protocol_metadata else None,
        "training_data_sha256": protocol_metadata.get("training_data_sha256") if protocol_metadata else None,
        "plan_sha256": protocol_metadata.get("plan_sha256") if protocol_metadata else None,
        "plan_path": protocol_metadata.get("plan_path") if protocol_metadata else None,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "dataset": dataset,
        "setting": setting,
        "fusion": args.fusion,
        "model_seed": int(args.model_seed),
        "epochs_configured": epochs,
        "batch_size": batch,
        "best_epoch": history["best_epoch"],
        "stop_epoch": history["stop_epoch"],
        "stop_reason": history["stop_reason"],
        "validation_metrics": validation_metrics,
        "parameter_summary": parameter_summary,
        "elapsed_seconds": elapsed,
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": _sha256(ckpt_path),
        "history_file": str(history_path.resolve()),
        "validation_prediction_file": str(val_pred_path.resolve()),
        "whole_model_health_file": str(health_path.resolve()),
        "whole_model_health_gate": health_result,
        "test_evaluated": False,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    tmp = run_meta_path.with_suffix(run_meta_path.suffix + ".tmp")
    tmp.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run_meta_path)

    print(f"\nCheckpoint : {ckpt_path}")
    print(f"History    : {history_path}")
    print(f"Val preds  : {val_pred_path}")
    print(f"Run meta   : {run_meta_path}")
    print("Held-out test remains sealed.")


if __name__ == "__main__":
    main()
