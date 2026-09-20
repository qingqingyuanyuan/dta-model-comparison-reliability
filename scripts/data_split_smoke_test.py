#!/usr/bin/env python3
"""Validate finalized split invariants without requiring PyTorch Geometric.

Checks every dataset/setting defined in config.py. This is a hard gate before
formal training because it catches accidental split leakage, scaler leakage,
and stale/missing cold-drug artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from data.utils import load_fixed_split


def _pair_set(df):
    return set(zip(df["Canonical SMILES"].astype(str), df["Target Sequence"].astype(str)))


def _assert_disjoint(a, b, label):
    overlap = a & b
    if overlap:
        raise AssertionError(f"{label}: overlap={len(overlap)}")


def _check_one(dataset: str, setting: str):
    train, val, test, scaler = load_fixed_split(dataset, setting, data_root=ROOT / "data_final")

    if scaler.get("fit_on") != "train_only":
        raise AssertionError(f"{dataset}/{setting}: scaler is not train_only")
    if str(scaler.get("method")) != "minmax":
        raise AssertionError(f"{dataset}/{setting}: expected minmax scaler")

    # Recompute normalization exactly from the saved train-only scaler.
    lo = float(scaler["min"])
    hi = float(scaler["max"])
    rng = hi - lo
    if rng <= 0:
        raise AssertionError(f"{dataset}/{setting}: non-positive scaler range")
    for split_name, frame in (("train", train), ("val", val), ("test", test)):
        expected = (frame["Label"].to_numpy(float) - lo) / rng
        actual = frame["Label_norm"].to_numpy(float)
        err = float(np.max(np.abs(expected - actual))) if len(frame) else 0.0
        if err > 1e-10:
            raise AssertionError(
                f"{dataset}/{setting}/{split_name}: Label_norm mismatch maxerr={err}"
            )

    trp, vap, tep = map(_pair_set, (train, val, test))
    _assert_disjoint(trp, vap, f"{dataset}/{setting} train-val input")
    _assert_disjoint(trp, tep, f"{dataset}/{setting} train-test input")
    _assert_disjoint(vap, tep, f"{dataset}/{setting} val-test input")

    if setting == "cold_drug":
        tr = set(train["Canonical SMILES"].astype(str))
        va = set(val["Canonical SMILES"].astype(str))
        te = set(test["Canonical SMILES"].astype(str))
        _assert_disjoint(tr, va, f"{dataset}/cold_drug train-val drug")
        _assert_disjoint(tr, te, f"{dataset}/cold_drug train-test drug")
        _assert_disjoint(va, te, f"{dataset}/cold_drug val-test drug")

    if setting == "cold_target":
        tr = set(train["Target Sequence"].astype(str))
        va = set(val["Target Sequence"].astype(str))
        te = set(test["Target Sequence"].astype(str))
        _assert_disjoint(tr, va, f"{dataset}/cold_target train-val protein")
        _assert_disjoint(tr, te, f"{dataset}/cold_target train-test protein")
        _assert_disjoint(va, te, f"{dataset}/cold_target val-test protein")

    split_meta = ROOT / "data_final" / dataset / setting / "split_metadata.json"
    if not split_meta.exists():
        raise AssertionError(f"missing split metadata: {split_meta}")
    meta = json.loads(split_meta.read_text(encoding="utf-8"))
    if str(meta.get("setting")) != setting:
        raise AssertionError(f"{dataset}/{setting}: split metadata setting mismatch")
    if meta.get("label_scaler_fit") != "train_only":
        raise AssertionError(f"{dataset}/{setting}: split metadata scaler policy mismatch")

    return len(train), len(val), len(test)


def main():
    expected_settings = [str(x).lower() for x in cfg.DATA_CONFIG["settings"]]
    if expected_settings != ["warm", "cold_drug", "cold_target"]:
        raise AssertionError(f"Unexpected formal settings: {expected_settings}")

    for dataset in cfg.DATA_CONFIG["datasets"]:
        dataset = str(dataset).upper()
        for setting in expected_settings:
            counts = _check_one(dataset, setting)
            print(
                f"[PASS] {dataset:5s}/{setting:11s} "
                f"train={counts[0]:,} val={counts[1]:,} test={counts[2]:,}"
            )

    print("\nALL DATA SPLIT SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
