#!/usr/bin/env python3
"""Aggregate final ZhiYao-Graph evaluation artifacts across paired model seeds."""

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


PRIMARY_METRICS = ("ci", "mse_norm", "rmse_norm", "mae_norm")


def _expected_rows():
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
        meta = Path(cfg.EVALUATION_OUTPUT_DIR) / (
            f"{stem}__{dataset.lower()}__{setting}__evaluation.json"
        )
        rows.append(
            {
                "dataset": dataset,
                "setting": setting,
                "fusion": fusion,
                "seed": seed,
                "stem": stem,
                "meta": meta,
            }
        )
    return rows


def _load_one(row):
    path = row["meta"]
    if not path.exists():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("metric_schema_version") != METRIC_SCHEMA_VERSION:
        raise ValueError(
            f"Metric schema mismatch in {path}: {meta.get('metric_schema_version')!r}"
        )
    checks = {
        "dataset": str(meta.get("dataset", "")).upper(),
        "setting": str(meta.get("setting", "")).lower(),
        "fusion": str(meta.get("fusion", "")).lower(),
        "seed": int(meta.get("model_seed")),
    }
    for key in ("dataset", "setting", "fusion", "seed"):
        if checks[key] != row[key]:
            raise ValueError(f"Evaluation metadata mismatch {key} in {path}")

    rec = meta.get("record_level_metrics") or {}
    uniq = meta.get("unique_input_level_metrics") or {}
    out = {
        "dataset": row["dataset"],
        "setting": row["setting"],
        "fusion": row["fusion"],
        "seed": row["seed"],
        "test_samples": int(meta.get("test_samples", 0)),
        "unique_model_inputs": int(meta.get("unique_model_inputs", 0)),
        "checkpoint_sha256": meta.get("checkpoint_sha256"),
        "evaluation_file": str(path.resolve()),
    }
    for metric in PRIMARY_METRICS:
        if metric not in rec or metric not in uniq:
            raise ValueError(f"Missing metric {metric} in {path}")
        out[f"record_{metric}"] = float(rec[metric])
        out[f"unique_{metric}"] = float(uniq[metric])
    for metric in ("mse_raw", "rmse_raw", "mae_raw"):
        if metric in rec:
            out[f"record_{metric}"] = float(rec[metric])
        if metric in uniq:
            out[f"unique_{metric}"] = float(uniq[metric])
    return out


def _summary(individual: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in individual.columns
        if c.startswith("record_") or c.startswith("unique_")
    ]
    rows = []
    for keys, grp in individual.groupby(["dataset", "setting", "fusion"], sort=True):
        row = {
            "dataset": keys[0],
            "setting": keys[1],
            "fusion": keys[2],
            "n_seeds": int(len(grp)),
            "seeds": ",".join(str(int(x)) for x in sorted(grp["seed"])),
        }
        for col in metric_cols:
            vals = grp[col].to_numpy(float)
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_differences(individual: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired fusion differences; no underpowered p-values are invented."""
    rows = []
    fusion_pairs = list(itertools.combinations(cfg.EXPERIMENT_MATRIX["fusions"], 2))
    for (dataset, setting), block in individual.groupby(["dataset", "setting"], sort=True):
        for fa, fb in fusion_pairs:
            a = block[block["fusion"] == fa].set_index("seed")
            b = block[block["fusion"] == fb].set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            for metric in PRIMARY_METRICS:
                col = f"record_{metric}"
                if not seeds:
                    continue
                diffs = np.array([float(a.loc[s, col]) - float(b.loc[s, col]) for s in seeds])
                rows.append(
                    {
                        "dataset": dataset,
                        "setting": setting,
                        "fusion_a": fa,
                        "fusion_b": fb,
                        "metric": metric,
                        "difference_definition": "a_minus_b",
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
        "--output-dir", type=Path, default=ROOT / "results" / "final"
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow aggregation before all prespecified primary evaluation artifacts exist",
    )
    args = p.parse_args()

    expected = _expected_rows()
    loaded = []
    missing = []
    for row in expected:
        item = _load_one(row)
        if item is None:
            missing.append(row["stem"])
        else:
            loaded.append(item)

    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"Final result matrix incomplete: {len(missing)} missing runs.\n  "
            + "\n  ".join(missing[:20])
            + ("\n  ..." if len(missing) > 20 else "")
        )
    if not loaded:
        raise RuntimeError("No final evaluation artifacts found")

    individual = pd.DataFrame(loaded).sort_values(
        ["dataset", "setting", "fusion", "seed"]
    )
    summary = _summary(individual)
    paired = _paired_differences(individual)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    individual.to_csv(out / "individual_runs.csv", index=False)
    summary.to_csv(out / "summary_mean_sd.csv", index=False)
    paired.to_csv(out / "paired_seed_differences.csv", index=False)

    completeness = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "expected_runs": len(expected),
        "found_runs": len(loaded),
        "complete": len(missing) == 0,
        "missing_runs": missing,
        "reporting_rule": (
            "Report individual seeds and mean ± SD. Paired differences use the "
            "same seeds across fusion strategies. No significance p-values are "
            "reported for the 3-seed primary matrix."
        ),
    }
    (out / "completeness.json").write_text(
        json.dumps(completeness, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Individual runs : {out / 'individual_runs.csv'}")
    print(f"Mean ± SD       : {out / 'summary_mean_sd.csv'}")
    print(f"Paired deltas   : {out / 'paired_seed_differences.csv'}")
    print(f"Complete matrix : {completeness['complete']} ({len(loaded)}/{len(expected)})")


if __name__ == "__main__":
    main()
