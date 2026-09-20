#!/usr/bin/env python3
"""Run validation-only modality diagnostics across all prespecified primary checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from scripts.experiment_integrity import checkpoint_matches, expected_primary_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--max-batches", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "diagnostics",
    )
    args = p.parse_args()

    rows = expected_primary_rows()
    invalid = [r["stem"] for r in rows if not checkpoint_matches(r)]
    if invalid:
        raise RuntimeError(
            "Primary training checkpoints incomplete; diagnostics are intended "
            "for frozen trained models. Missing/invalid:\n  "
            + "\n  ".join(invalid[:20])
            + ("\n  ..." if len(invalid) > 20 else "")
        )

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    for i, row in enumerate(rows, 1):
        out = out_dir / f"{row['stem']}__modality_diagnostic.json"
        print(f"\n[{i}/{len(rows)}] {row['stem']}")
        if out.exists() and not args.overwrite:
            print("SKIP: existing diagnostic")
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "model_diagnostics.py"),
            "--ckpt", str(row["checkpoint"]),
            "--data", row["dataset"].lower(),
            "--setting", row["setting"],
            "--data-root", str(args.data_root),
            "--max-batches", str(args.max_batches),
            "--batch", str(args.batch),
            "--output", str(out),
        ]
        if args.no_progress:
            cmd.append("--no-progress")
        code = subprocess.call(cmd, cwd=ROOT)
        if code != 0:
            failures.append(row["stem"])
            if not args.continue_on_error:
                raise SystemExit(code)

    records = []
    for row in rows:
        path = out_dir / f"{row['stem']}__modality_diagnostic.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "dataset": meta["dataset"],
                "setting": meta["setting"],
                "fusion": meta["fusion"],
                "seed": meta["model_seed"],
                "samples_checked": meta["samples_checked"],
                "protein_shuffle_mean_abs_delta": meta["protein_shuffle"]["mean_abs_prediction_change"],
                "drug_shuffle_mean_abs_delta": meta["drug_shuffle"]["mean_abs_prediction_change"],
                "protein_to_drug_sensitivity_ratio": meta["protein_to_drug_sensitivity_ratio"],
                "base_prediction_std": meta["base_prediction_std"],
                "diagnostic_file": str(path),
            }
        )
    if records:
        df = pd.DataFrame(records).sort_values(
            ["dataset", "setting", "fusion", "seed"]
        )
        df.to_csv(out_dir / "individual_modality_diagnostics.csv", index=False)
        summary = (
            df.groupby(["dataset", "setting", "fusion"], as_index=False)
            .agg(
                n_seeds=("seed", "size"),
                protein_shuffle_mean=("protein_shuffle_mean_abs_delta", "mean"),
                protein_shuffle_sd=("protein_shuffle_mean_abs_delta", "std"),
                drug_shuffle_mean=("drug_shuffle_mean_abs_delta", "mean"),
                drug_shuffle_sd=("drug_shuffle_mean_abs_delta", "std"),
                protein_drug_ratio_mean=("protein_to_drug_sensitivity_ratio", "mean"),
                protein_drug_ratio_sd=("protein_to_drug_sensitivity_ratio", "std"),
            )
        )
        summary.to_csv(out_dir / "summary_modality_diagnostics.csv", index=False)
        print(f"\nSaved summary: {out_dir / 'summary_modality_diagnostics.csv'}")

    if failures:
        print(f"Failures: {len(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
