#!/usr/bin/env python3
"""Fast tests for the final ZhiYao-Graph regression metric implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.evaluate import (
    Evaluator,
    _concordance_index_reference,
    concordance_index,
)


def assert_close(a, b, tol=1e-12):
    if not np.isclose(float(a), float(b), rtol=0.0, atol=tol):
        raise AssertionError(f"{a} != {b}")


def test_known_cases():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert_close(concordance_index(y, y), 1.0)
    assert_close(concordance_index(y, -y), 0.0)
    assert_close(concordance_index(y, np.zeros_like(y)), 0.5)

    # One comparable pair has tied predictions -> half credit.
    assert_close(concordance_index([1, 2], [3, 3]), 0.5)

    # Observed-label ties are excluded entirely.
    assert_close(concordance_index([1, 1, 2], [100, -100, 0]), 0.5)
    print("[PASS] known CI cases and tie semantics")


def test_fast_vs_reference():
    rng = np.random.default_rng(20260906)
    for n in (2, 3, 7, 19, 64):
        for _ in range(100):
            # Integer values deliberately create ties in both true and predicted labels.
            y = rng.integers(0, 5, size=n).astype(float)
            p = rng.integers(0, 7, size=n).astype(float)
            fast = concordance_index(y, p)
            ref = _concordance_index_reference(y, p)
            assert_close(fast, ref)
    print("[PASS] O(n log n) CI exactly matches O(n^2) reference")


def test_affine_invariance():
    rng = np.random.default_rng(7)
    y = rng.normal(size=100)
    p = rng.normal(size=100)
    base = concordance_index(y, p)
    assert_close(base, concordance_index(5.0 * y + 11.0, 2.0 * p - 3.0))
    print("[PASS] CI positive-affine invariance")


def test_metric_scaling():
    scaler = {
        "method": "minmax",
        "min": 5.0,
        "max": 15.0,
        "range": 10.0,
        "fit_on": "train_only",
    }
    y = np.array([0.0, 0.5, 1.0])
    p = np.array([0.1, 0.4, 0.8])
    m = Evaluator(scaler=scaler).evaluate(y, p)
    assert_close(m["mse_raw"], m["mse_norm"] * 100.0)
    assert_close(m["rmse_raw"], m["rmse_norm"] * 10.0)
    assert_close(m["mae_raw"], m["mae_norm"] * 10.0)
    assert 0.0 <= m["ci"] <= 1.0
    print("[PASS] normalized/raw metric scaling")


def main():
    test_known_cases()
    test_fast_vs_reference()
    test_affine_invariance()
    test_metric_scaling()
    print("\nALL METRIC SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
