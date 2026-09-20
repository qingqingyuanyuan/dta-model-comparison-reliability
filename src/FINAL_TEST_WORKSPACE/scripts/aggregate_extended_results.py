#!/usr/bin/env python3
"""Aggregate primary, controlled-baseline and sensitivity evaluation artifacts.

Outputs are machine-readable long-form tables. No p-values are generated for
the 3-seed design; reporting is mean ± SD plus seed-wise paired differences.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from train.evaluate import METRIC_SCHEMA_VERSION

METRICS = ("ci", "mse_norm", "rmse_norm", "mae_norm")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_rows():
    rows, missing = [], []
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
        path = Path(cfg.EVALUATION_OUTPUT_DIR) / (
            f"{stem}__{dataset.lower()}__{setting}__evaluation.json"
        )
        if not path.exists():
            missing.append(f"primary:{stem}")
            continue
        meta = _load_json(path)
        if meta.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            raise ValueError(f"metric schema mismatch: {path}")
        rec = meta["record_level_metrics"]
        uniq = meta["unique_input_level_metrics"]
        row = {
            "experiment_family": "primary",
            "model": fusion,
            "dataset": dataset,
            "setting": setting,
            "seed": seed,
            "evaluation_file": str(path.resolve()),
        }
        for m in METRICS:
            row[f"record_{m}"] = float(rec[m])
            row[f"unique_{m}"] = float(uniq[m])
        rows.append(row)
    return rows, missing


def _baseline_rows():
    rows, missing = [], []
    out_dir = Path(cfg.EVALUATION_OUTPUT_DIR) / "baselines"
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
        path = out_dir / f"{stem}__{dataset.lower()}__{setting}__evaluation.json"
        if not path.exists():
            missing.append(f"baseline:{stem}")
            continue
        meta = _load_json(path)
        if meta.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            raise ValueError(f"metric schema mismatch: {path}")
        if str(meta.get("baseline_name", "")).lower() != baseline:
            raise ValueError(f"baseline name mismatch: {path}")
        rec = meta["record_level_metrics"]
        uniq = meta["unique_input_level_metrics"]
        row = {
            "experiment_family": "controlled_baseline",
            "model": baseline,
            "dataset": dataset,
            "setting": setting,
            "seed": seed,
            "evaluation_file": str(path.resolve()),
        }
        for m in METRICS:
            row[f"record_{m}"] = float(rec[m])
            row[f"unique_{m}"] = float(uniq[m])
        rows.append(row)
    return rows, missing


def _sensitivity_rows():
    rows, missing = [], []
    out_dir = Path(cfg.EVALUATION_OUTPUT_DIR) / "sensitivity"
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
        fusion = str(cfg.SENSITIVITY_EXPERIMENTS[variant]["fusion"]).lower()
        stem = f"sensitivity__{variant}__{dataset.lower()}_{setting}_{fusion}_seed{seed}"
        path = out_dir / f"{stem}__{dataset.lower()}__{setting}__evaluation.json"
        if not path.exists():
            missing.append(f"sensitivity:{stem}")
            continue
        meta = _load_json(path)
        if meta.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
            raise ValueError(f"metric schema mismatch: {path}")
        if meta.get("sensitivity_name") != variant:
            raise ValueError(f"sensitivity name mismatch: {path}")
        rec = meta["record_level_metrics"]
        uniq = meta["unique_input_level_metrics"]
        row = {
            "experiment_family": "sensitivity",
            "model": variant,
            "dataset": dataset,
            "setting": setting,
            "seed": seed,
            "reference_model": "concat",
            "evaluation_file": str(path.resolve()),
        }
        for m in METRICS:
            row[f"record_{m}"] = float(rec[m])
            row[f"unique_{m}"] = float(uniq[m])
        rows.append(row)
    return rows, missing


def _mean_sd(df: pd.DataFrame):
    metric_cols = [c for c in df.columns if c.startswith("record_") or c.startswith("unique_")]
    rows = []
    for keys, grp in df.groupby(
        ["experiment_family", "model", "dataset", "setting"], sort=True
    ):
        row = {
            "experiment_family": keys[0],
            "model": keys[1],
            "dataset": keys[2],
            "setting": keys[3],
            "n_seeds": int(len(grp)),
            "seeds": ",".join(str(int(x)) for x in sorted(grp["seed"])),
        }
        for col in metric_cols:
            vals = grp[col].to_numpy(float)
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _sensitivity_deltas(all_df: pd.DataFrame):
    primary = all_df[
        (all_df["experiment_family"] == "primary")
        & (all_df["model"] == "concat")
    ]
    sens = all_df[all_df["experiment_family"] == "sensitivity"]
    rows = []
    for (dataset, setting, variant), grp in sens.groupby(
        ["dataset", "setting", "model"], sort=True
    ):
        ref = primary[
            (primary["dataset"] == dataset)
            & (primary["setting"] == setting)
        ].set_index("seed")
        var = grp.set_index("seed")
        seeds = sorted(set(ref.index) & set(var.index))
        for metric in METRICS:
            col = f"record_{metric}"
            diffs = np.array(
                [float(var.loc[s, col]) - float(ref.loc[s, col]) for s in seeds],
                dtype=float,
            )
            if not len(diffs):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "setting": setting,
                    "variant": variant,
                    "reference": "primary_concat",
                    "metric": metric,
                    "difference_definition": "variant_minus_reference",
                    "n_paired_seeds": len(seeds),
                    "paired_seeds": ",".join(map(str, seeds)),
                    "mean_difference": float(diffs.mean()),
                    "sd_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                    "individual_differences": ",".join(f"{x:.10g}" for x in diffs),
                }
            )
    return pd.DataFrame(rows)


def _baseline_deltas(all_df: pd.DataFrame):
    primary = all_df[all_df["experiment_family"] == "primary"]
    base = all_df[all_df["experiment_family"] == "controlled_baseline"]
    rows = []
    for (dataset, setting, baseline), grp in base.groupby(
        ["dataset", "setting", "model"], sort=True
    ):
        b = grp.set_index("seed")
        for fusion in cfg.EXPERIMENT_MATRIX["fusions"]:
            ref = primary[
                (primary["dataset"] == dataset)
                & (primary["setting"] == setting)
                & (primary["model"] == fusion)
            ].set_index("seed")
            seeds = sorted(set(ref.index) & set(b.index))
            for metric in METRICS:
                col = f"record_{metric}"
                diffs = np.array(
                    [float(ref.loc[s, col]) - float(b.loc[s, col]) for s in seeds],
                    dtype=float,
                )
                if not len(diffs):
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "zhiy_graph_model": fusion,
                        "baseline": baseline,
                        "metric": metric,
                        "difference_definition": "zhiy_graph_minus_baseline",
                        "n_paired_seeds": len(seeds),
                        "paired_seeds": ",".join(map(str, seeds)),
                        "mean_difference": float(diffs.mean()),
                        "sd_difference": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                        "individual_differences": ",".join(f"{x:.10g}" for x in diffs),
                    }
                )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "extended"
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow exploratory aggregation before all 126 final evaluations exist",
    )
    args = p.parse_args()

    primary, m1 = _primary_rows()
    baselines, m2 = _baseline_rows()
    sens, m3 = _sensitivity_rows()
    missing = m1 + m2 + m3
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Extended experiment matrix incomplete: {len(missing)} missing evaluations.\n  "
            + "\n  ".join(missing[:30])
            + ("\n  ..." if len(missing) > 30 else "")
        )
    loaded = primary + baselines + sens
    if not loaded:
        raise RuntimeError("No evaluation artifacts found")

    all_df = pd.DataFrame(loaded).sort_values(
        ["experiment_family", "dataset", "setting", "model", "seed"]
    )
    summary = _mean_sd(all_df)
    sens_delta = _sensitivity_deltas(all_df)
    base_delta = _baseline_deltas(all_df)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "all_individual_runs.csv", index=False)
    summary.to_csv(out / "all_models_summary_mean_sd.csv", index=False)
    sens_delta.to_csv(out / "sensitivity_paired_differences.csv", index=False)
    base_delta.to_csv(out / "baseline_vs_zhiy_graph_paired_differences.csv", index=False)

    completeness = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "expected_primary": int(cfg.EXTENDED_EXPERIMENT_COUNTS["primary_fusion_runs"]),
        "expected_baselines": int(cfg.EXTENDED_EXPERIMENT_COUNTS["controlled_baseline_runs"]),
        "expected_sensitivity": int(cfg.EXTENDED_EXPERIMENT_COUNTS["sensitivity_runs"]),
        "expected_total": int(cfg.EXTENDED_EXPERIMENT_COUNTS["total_prespecified_training_runs"]),
        "found_primary": len(primary),
        "found_baselines": len(baselines),
        "found_sensitivity": len(sens),
        "missing": missing,
        "complete": len(missing) == 0,
        "reporting_rule": (
            "Mean ± SD across 3 prespecified seeds. Seed-wise differences are descriptive; "
            "no inferential p-values are generated from n=3 paired runs."
        ),
    }
    (out / "completeness.json").write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved extended results to: {out}")
    print(f"Complete: {completeness['complete']}")


if __name__ == "__main__":
    main()
