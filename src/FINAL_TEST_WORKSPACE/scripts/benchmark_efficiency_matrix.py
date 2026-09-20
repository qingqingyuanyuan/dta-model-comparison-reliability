#!/usr/bin/env python3
"""Benchmark a prespecified representative efficiency panel on validation data.

Panel: KIBA-warm and Davis-warm, seed 42, three ZhiYao-Graph fusion models plus
two controlled baselines = 10 checkpoints. Test data are never used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from scripts.experiment_integrity import checkpoint_matches


def build_panel():
    seed = int(cfg.EFFICIENCY_BENCHMARK["representative_seed"])
    rows = []
    for dataset in ("KIBA", "DAVIS"):
        setting = "warm"
        for fusion in cfg.EXPERIMENT_MATRIX["fusions"]:
            stem = f"{dataset.lower()}_{setting}_{fusion}_seed{seed}"
            rows.append(
                {
                    "family": "primary",
                    "dataset": dataset,
                    "setting": setting,
                    "model": fusion,
                    "seed": seed,
                    "stem": stem,
                    "checkpoint": Path(cfg.MODEL_DIR) / f"{stem}.pt",
                }
            )
        for baseline in cfg.BASELINE_EXPERIMENT_MATRIX["baselines"]:
            stem = f"{baseline}__{dataset.lower()}_{setting}_seed{seed}"
            rows.append(
                {
                    "family": "controlled_baseline",
                    "dataset": dataset,
                    "setting": setting,
                    "model": baseline,
                    "seed": seed,
                    "stem": stem,
                    "checkpoint": Path(cfg.MODEL_DIR) / "baselines" / f"{stem}.pt",
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", type=Path, default=ROOT / "data_final")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "efficiency" / "efficiency_benchmark.csv",
    )
    args = p.parse_args()

    rows = build_panel()
    missing = [r["stem"] for r in rows if not checkpoint_matches(r)]
    if missing:
        raise RuntimeError(
            "Efficiency panel checkpoints are incomplete:\n  "
            + "\n  ".join(missing)
        )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_efficiency.py"),
        "--ckpt",
        *[str(r["checkpoint"]) for r in rows],
        "--data-root", str(args.data_root),
        "--device", args.device,
        "--batch", str(cfg.EFFICIENCY_BENCHMARK["batch_size"]),
        "--warmup-batches", str(cfg.EFFICIENCY_BENCHMARK["warmup_batches"]),
        "--measure-batches", str(cfg.EFFICIENCY_BENCHMARK["measure_batches"]),
        "--num-workers", str(args.num_workers),
        "--output", str(args.output),
    ]
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
