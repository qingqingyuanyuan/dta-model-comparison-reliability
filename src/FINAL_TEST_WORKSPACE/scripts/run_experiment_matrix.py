#!/usr/bin/env python3
"""Plan/run the prespecified ZhiYao-Graph primary fusion matrix.

Primary matrix = 2 datasets x 3 settings x 3 fusions x 3 paired model seeds
= 54 runs. Training and held-out test evaluation are separate phases.
"""

from __future__ import annotations

import argparse
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

ARCHITECTURE_VERSION = cfg.MODEL_ARCHITECTURE_VERSION


def _trusted_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _selected(values, requested):
    if not requested:
        return list(values)
    allowed = {str(x).lower(): x for x in values}
    out = []
    for item in requested:
        key = str(item).lower()
        if key not in allowed:
            raise ValueError(f"Unknown selection {item!r}; allowed={list(values)}")
        out.append(allowed[key])
    return out


def build_matrix(args):
    datasets = _selected(cfg.EXPERIMENT_MATRIX["datasets"], args.data)
    settings = _selected(cfg.EXPERIMENT_MATRIX["settings"], args.setting)
    fusions = _selected(cfg.EXPERIMENT_MATRIX["fusions"], args.fusion)
    allowed_seeds = [int(x) for x in cfg.EXPERIMENT_MATRIX["model_seeds"]]
    seeds = [int(x) for x in (args.seed or allowed_seeds)]
    unknown = sorted(set(seeds) - set(allowed_seeds))
    if unknown:
        raise ValueError(f"Unknown final seeds {unknown}; allowed={allowed_seeds}")

    rows = []
    for dataset, setting, fusion, seed in itertools.product(
        datasets, settings, fusions, seeds
    ):
        dataset = str(dataset).upper()
        setting = str(setting).lower()
        fusion = str(fusion).lower()
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


def build_full_matrix():
    class _Args:
        data = None
        setting = None
        fusion = None
        seed = None
    return build_matrix(_Args())


def _row_key(row):
    return (
        str(row["dataset"]).upper(),
        str(row["setting"]).lower(),
        str(row["fusion"]).lower(),
        int(row["seed"]),
    )


def _read_stem_file(path: Path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Primary shard stem file not found: {path}"
        )

    stems = [
        x.strip()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]

    if not stems:
        raise RuntimeError("PRIMARY_SHARD_EMPTY — STOP")

    if len(stems) != len(set(stems)):
        raise RuntimeError(
            "PRIMARY_SHARD_DUPLICATE_STEM — STOP"
        )

    return stems


def _select_frozen_rows(frozen_rows, args):
    stem_file = getattr(args, "stem_file", None)

    if stem_file is not None:
        if any(
            getattr(args, x, None)
            for x in ("data", "setting", "fusion", "seed")
        ):
            raise RuntimeError(
                "PRIMARY_SHARD_SELECTION_CONFLICT — STOP"
            )

        requested = _read_stem_file(stem_file)
        frozen = {
            str(row["stem"]): dict(row)
            for row in frozen_rows
        }

        unknown = [x for x in requested if x not in frozen]
        if unknown:
            raise RuntimeError(
                "PRIMARY_SHARD_UNKNOWN_STEM — STOP: "
                + ", ".join(unknown[:10])
            )

        return [frozen[x] for x in requested]

    requested = build_matrix(args)
    keys = {_row_key(row) for row in requested}

    selected = [
        dict(row)
        for row in frozen_rows
        if _row_key(row) in keys
    ]

    if len(selected) != len(requested):
        raise RuntimeError(
            "FROZEN_PLAN_MISMATCH — STOP: "
            "requested primary subset not found"
        )

    return selected


def _checkpoint_matches(path: Path, row: dict, protocol_metadata: dict) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    try:
        ck = _trusted_load(path)
        return bool(
            isinstance(ck, dict)
            and ck.get("architecture_version") == ARCHITECTURE_VERSION
            and str(ck.get("dataset", "")).upper() == row["dataset"]
            and str(ck.get("setting", "")).lower() == row["setting"]
            and str(ck.get("fusion_mode", "")).lower() == row["fusion"]
            and int(ck.get("model_seed")) == int(row["seed"])
            and ck.get("test_evaluated_during_training") is False
            and checkpoint_provenance_matches(ck, protocol_metadata)
        )
    except Exception:
        return False


def _evaluation_meta_path(row: dict) -> Path:
    ckpt = resolve_project_path(row["checkpoint"])
    return Path(cfg.EVALUATION_OUTPUT_DIR) / (
        f"{ckpt.stem}__{row['dataset'].lower()}__{row['setting']}__evaluation.json"
    )


def _write_plan(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **build_plan_header(cfg, family="primary"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture_version": ARCHITECTURE_VERSION,
        "data_policy": cfg.DATA_CONFIG,
        "training_protocol": cfg.TRAINING,
        "evaluation_protocol": cfg.EVALUATION,
        "matrix": rows,
        "n_runs": len(rows),
        "test_policy": (
            "Train all primary, controlled-baseline and sensitivity runs using "
            "train/validation only. Held-out test evaluation remains sealed until "
            "the full prespecified 126-run training program is frozen."
        ),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _append_status(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run(cmd, *, dry_run):
    print("\n$", " ".join(str(x) for x in cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=ROOT)


def train_phase(rows, args, status_path, protocol_metadata):
    for i, row in enumerate(rows, 1):
        ckpt = resolve_project_path(row["checkpoint"])
        print(f"\n[{i}/{len(rows)}] TRAIN {row['stem']}")
        if _checkpoint_matches(ckpt, row, protocol_metadata) and not args.overwrite:
            print("  SKIP: valid V1C-bound final checkpoint already exists")
            _append_status(status_path, {**row, "phase": "train", "status": "skipped_valid"})
            continue

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

        code = _run(cmd, dry_run=args.dry_run)
        status = "dry_run" if args.dry_run else "success" if code == 0 else "failed"
        _append_status(status_path, {**row, "phase": "train", "status": status, "returncode": code})
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)


def evaluate_phase(rows, args, status_path):
    if args.unlock_test != "FINAL_TEST_PHASE":
        raise RuntimeError(
            "Held-out test is locked. For the deliberate final evaluation phase, "
            "pass exactly: --unlock-test FINAL_TEST_PHASE"
        )

    require_full_training_complete()

    for i, row in enumerate(rows, 1):
        meta = _evaluation_meta_path(row)
        print(f"\n[{i}/{len(rows)}] TEST {row['stem']}")
        if meta.exists() and not args.overwrite:
            print("  SKIP: evaluation metadata already exists")
            _append_status(status_path, {**row, "phase": "evaluate", "status": "skipped_existing"})
            continue

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_model.py"),
            "--data", row["dataset"].lower(),
            "--setting", row["setting"],
            "--ckpt", str(resolve_project_path(row["checkpoint"])),
            "--batch", str(cfg.TRAINING["batch_size"]),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
            "--data-root", str(args.data_root),
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        if args.overwrite:
            cmd.append("--overwrite")
        code = _run(cmd, dry_run=args.dry_run)
        status = "dry_run" if args.dry_run else "success" if code == 0 else "failed"
        _append_status(status_path, {**row, "phase": "evaluate", "status": status, "returncode": code})
        if code != 0 and not args.continue_on_error:
            raise SystemExit(code)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["plan", "train", "evaluate"], default="plan")
    p.add_argument("--data", nargs="*", choices=["KIBA", "DAVIS", "kiba", "davis"])
    p.add_argument("--setting", nargs="*", choices=[str(x).lower() for x in cfg.EXPERIMENT_MATRIX["settings"]])
    p.add_argument("--fusion", nargs="*", choices=["concat", "product", "attention"])
    p.add_argument("--seed", nargs="*", type=int)
    p.add_argument("--stem-file", type=Path, default=None)
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
        default=ROOT / "experiments" / "experiment_plan.json",
    )
    p.add_argument(
        "--status-file",
        type=Path,
        default=ROOT / "experiments" / "experiment_status.jsonl",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.phase != "plan":
        require_canonical_data_root(args.data_root)

    if args.phase == "plan":
        full_rows = build_full_matrix()
        _write_plan(full_rows, args.plan_file)
        print(f"Experiment plan: {args.plan_file}")
        print(f"Runs: {len(full_rows)}")
        for row in full_rows:
            print(f"  {row['stem']}")
        return

    expected_full = build_full_matrix()
    plan, protocol_metadata = load_and_validate_plan(
        args.plan_file,
        cfg,
        expected_family="primary",
        expected_matrix=expected_full,
    )
    rows = _select_frozen_rows(plan["matrix"], args)
    print(f"Frozen plan: {args.plan_file} ({plan.get('n_runs')} runs)")
    print(f"Plan SHA256: {protocol_metadata['plan_sha256']}")
    print(f"Selected from frozen matrix: {len(rows)}")

    if args.phase == "train":
        train_phase(rows, args, args.status_file, protocol_metadata)
    else:
        evaluate_phase(rows, args, args.status_file)


if __name__ == "__main__":
    main()
