#!/usr/bin/env python3
"""Plan/train/evaluate prespecified encoder sensitivity experiments.

Only one factor is changed at a time and ConcatFusion is fixed to isolate the
encoder/preprocessing factor:
  - protein_len_1000 vs primary 1500-aa protocol;
  - drug_pool_mean vs primary global-add pooling.

The 1500/add reference runs are NOT duplicated: they are the primary ConcatFusion
runs with the same paired seeds. This adds 36 training runs rather than 72.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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

def _trusted_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_matrix():
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


def _sensitivity_ckpt_matches(row, protocol_metadata):
    path = resolve_project_path(row["checkpoint"])
    if not path.exists():
        return False
    try:
        ck = _trusted_load(path)
        extra = ck.get("extra_metadata") or {}
        conf = ck.get("config") or {}
        return bool(
            ck.get("architecture_version") == cfg.MODEL_ARCHITECTURE_VERSION
            and str(ck.get("dataset", "")).upper() == row["dataset"]
            and str(ck.get("setting", "")).lower() == row["setting"]
            and str(ck.get("fusion_mode", "")).lower() == row["fusion"]
            and int(ck.get("model_seed")) == row["seed"]
            and extra.get("experiment_family") == "sensitivity"
            and extra.get("sensitivity_name") == row["variant"]
            and int(conf["PROTEIN_ENCODER"]["max_len"])
                == int(row["overrides"]["protein_max_len"])
            and str(conf["DRUG_ENCODER"]["pooling"])
                == str(row["overrides"]["drug_pooling"])
            and ck.get("test_evaluated_during_training") is False
            and checkpoint_provenance_matches(ck, protocol_metadata)
        )
    except Exception:
        return False


def _write_plan(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **build_plan_header(cfg, family="sensitivity"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "One-factor-at-a-time prespecified sensitivity analyses",
        "n_runs": len(rows),
        "n_extra_runs": len(rows),
        "matrix": rows,
        "sensitivity_specs": cfg.SENSITIVITY_EXPERIMENTS,
        "reference_rule": (
            "Each variant is paired by dataset/setting/seed to the primary "
            "ConcatFusion run. Main 1500-aa/add runs are not duplicated."
        ),
        "test_policy": (
            "Held-out test remains sealed until the full prespecified 126-run "
            "primary + controlled-baseline + sensitivity training program is frozen."
        ),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run(cmd, dry_run):
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
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--unlock-test", default=None)
    p.add_argument(
        "--plan-file",
        type=Path,
        default=ROOT / "experiments" / "sensitivity_plan.json",
    )
    args = p.parse_args()

    if args.phase != "plan":
        require_canonical_data_root(args.data_root)

    if args.phase == "plan":
        rows = build_matrix()
        _write_plan(rows, args.plan_file)
        print(f"Sensitivity plan: {args.plan_file}")
        print(f"Extra runs: {len(rows)}")
        return

    expected_rows = build_matrix()
    plan, protocol_metadata = load_and_validate_plan(
        args.plan_file,
        cfg,
        expected_family="sensitivity",
        expected_matrix=expected_rows,
    )
    rows = [dict(row) for row in plan["matrix"]]
    print(f"Frozen sensitivity plan: {args.plan_file} ({len(rows)} runs)")
    print(f"Plan SHA256: {protocol_metadata['plan_sha256']}")

    if args.phase == "train":
        for i, row in enumerate(rows, 1):
            print(f"\n[{i}/{len(rows)}] TRAIN {row['stem']}")
            if _sensitivity_ckpt_matches(row, protocol_metadata) and not args.overwrite:
                print("SKIP: valid V1C-bound sensitivity checkpoint exists")
                continue
            ov = row["overrides"]
            cmd = [
                sys.executable,
                str(ROOT / "run.py"),
                "--mode", "train",
                "--data", row["dataset"].lower(),
                "--setting", row["setting"],
                "--fusion", row["fusion"],
                "--model-seed", str(row["seed"]),
                "--final-run",
                "--frozen-plan", str(args.plan_file.resolve()),
                "--sensitivity-name", row["variant"],
                "--protein-max-len", str(ov["protein_max_len"]),
                "--drug-pooling", str(ov["drug_pooling"]),
                "--epochs", str(cfg.TRAINING["num_epochs"]),
                "--batch", str(cfg.TRAINING["batch_size"]),
                "--num-workers", str(args.num_workers),
                "--device", args.device,
                "--data-root", str(args.data_root),
            ]
            if args.no_progress:
                cmd.append("--no-progress")
            if args.overwrite:
                cmd.append("--overwrite")
            code = _run(cmd, args.dry_run)
            if code != 0 and not args.continue_on_error:
                raise SystemExit(code)
        return

    if args.unlock_test != "FINAL_SENSITIVITY_TEST_PHASE":
        raise RuntimeError(
            "Sensitivity held-out test is locked. Use exactly "
            "--unlock-test FINAL_SENSITIVITY_TEST_PHASE after training is frozen."
        )
    require_full_training_complete()

    out_dir = ROOT / "evaluation_outputs" / "sensitivity"
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] TEST {row['stem']}")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_model.py"),
            "--data", row["dataset"].lower(),
            "--setting", row["setting"],
            "--ckpt", str(resolve_project_path(row["checkpoint"])),
            "--data-root", str(args.data_root),
            "--batch", str(cfg.TRAINING["batch_size"]),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--output-dir", str(out_dir),
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        if args.overwrite:
            cmd.append("--overwrite")
        code = _run(cmd, args.dry_run)
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
