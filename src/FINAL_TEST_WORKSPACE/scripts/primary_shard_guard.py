#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config as cfg
from scripts import run_experiment_matrix as pm
from scripts.protocol_provenance import (
    load_and_validate_plan,
    require_canonical_data_root,
)

def read_stems(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    rows = [
        x.strip()
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]

    if not rows:
        raise RuntimeError("PRIMARY_SHARD_EMPTY — STOP")

    if len(rows) != len(set(rows)):
        raise RuntimeError(
            "PRIMARY_SHARD_INTERNAL_DUPLICATE — STOP"
        )

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan-file", type=Path, required=True)
    p.add_argument("--shard-a", type=Path, required=True)
    p.add_argument("--shard-b", type=Path, required=True)
    p.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data_final",
    )
    args = p.parse_args()

    require_canonical_data_root(args.data_root)

    plan, ctx = load_and_validate_plan(
        args.plan_file,
        cfg,
        expected_family="primary",
        expected_matrix=pm.build_full_matrix(),
    )

    expected = {x["stem"] for x in plan["matrix"]}
    a = set(read_stems(args.shard_a))
    b = set(read_stems(args.shard_b))

    overlap = a & b
    missing = expected - (a | b)
    extra = (a | b) - expected

    if overlap or missing or extra:
        raise RuntimeError(
            "PRIMARY_AB_SHARD_MISMATCH — STOP: "
            f"overlap={len(overlap)}, "
            f"missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    print("PRIMARY A/B SHARD GUARD = PASS")
    print("Shard A =", len(a))
    print("Shard B =", len(b))
    print("Intersection = 0")
    print("Union =", len(a | b))
    print("Plan SHA256 =", ctx["plan_sha256"])
    print("Data SHA256 =", ctx["training_data_sha256"])
    print("HELD-OUT TEST = SEALED")


if __name__ == "__main__":
    main()
