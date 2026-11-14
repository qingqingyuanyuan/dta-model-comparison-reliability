"""Standardized regression metrics for ZhiYao-Graph final experiments.

The final paper task is continuous drug-target affinity (DTA) regression.
This module intentionally standardizes one metric schema for all training,
evaluation, aggregation, figures and Web reporting.

Key fixes relative to the historical implementation
----------------------------------------------------
- Concordance Index (CI) now gives 0.5 credit to tied predictions.
- CI excludes pairs tied in the observed label, matching the usual DTA pairwise
  concordance definition.
- CI is computed in O(n log n), not O(n^2), so full KIBA/Davis test sets are
  practical.
- Normalized and raw-scale error metrics use explicit names and one schema.
- Classification metrics are deliberately rejected in this final DTA module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


METRIC_SCHEMA_VERSION = "zhiy_graph_regression_metrics_v2"


def _as_finite_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise ValueError(f"{name} contains {bad} non-finite values")
    return arr


class _FenwickTree:
    """Fenwick tree storing integer counts over 1-indexed ranks."""

    def __init__(self, size: int):
        self.tree = np.zeros(int(size) + 1, dtype=np.int64)

    def add(self, index: int, value: int = 1) -> None:
        i = int(index)
        n = self.tree.size
        while i < n:
            self.tree[i] += int(value)
            i += i & -i

    def prefix_sum(self, index: int) -> int:
        s = 0
        i = int(index)
        while i > 0:
            s += int(self.tree[i])
            i -= i & -i
        return s


def concordance_index(y_true, y_pred) -> float:
    """Compute pairwise concordance with exact tie handling in O(n log n).

    For every pair with different observed labels, the pair receives:
      - 1.0 if prediction ordering agrees with observed ordering;
      - 0.5 if the two predictions are exactly tied;
      - 0.0 otherwise.

    Pairs tied in ``y_true`` are not comparable and are excluded.

    This is invariant to any strictly increasing affine transformation of the
    labels/predictions and is suitable for full KIBA/Davis evaluation.
    """
    y = _as_finite_1d(y_true, "y_true")
    p = _as_finite_1d(y_pred, "y_pred")
    if y.size != p.size:
        raise ValueError(f"Length mismatch: y_true={y.size}, y_pred={p.size}")
    if y.size < 2:
        return 0.5

    # Compress prediction values to exact ordered ranks. Equal predictions get
    # the same rank, which lets us award 0.5 to prediction ties.
    _, pred_rank0 = np.unique(p, return_inverse=True)
    pred_rank = pred_rank0.astype(np.int64) + 1

    # Stable sorting is useful for deterministic debugging; same-y groups are
    # scored before any member of that group is inserted into the tree, so
    # observed-label ties are never compared with each other.
    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    r_sorted = pred_rank[order]

    tree = _FenwickTree(int(pred_rank.max()))
    comparable = 0
    concordant_score = 0.0
    previous_count = 0

    start = 0
    n = y_sorted.size
    while start < n:
        end = start + 1
        while end < n and y_sorted[end] == y_sorted[start]:
            end += 1

        # Compare this true-label group only against strictly lower true labels.
        if previous_count:
            for rank in r_sorted[start:end]:
                less = tree.prefix_sum(int(rank) - 1)
                leq = tree.prefix_sum(int(rank))
                equal = leq - less
                concordant_score += float(less) + 0.5 * float(equal)
                comparable += previous_count

        # Insert the whole observed-tie group only after scoring it.
        for rank in r_sorted[start:end]:
            tree.add(int(rank), 1)
        previous_count += end - start
        start = end

    return concordant_score / comparable if comparable else 0.5


def _concordance_index_reference(y_true, y_pred) -> float:
    """Slow O(n^2) reference used by the metric smoke test only."""
    y = _as_finite_1d(y_true, "y_true")
    p = _as_finite_1d(y_pred, "y_pred")
    if y.size != p.size:
        raise ValueError("Length mismatch")

    score = 0.0
    comparable = 0
    for i in range(y.size):
        for j in range(i + 1, y.size):
            if y[i] == y[j]:
                continue
            comparable += 1
            true_sign = np.sign(y[i] - y[j])
            pred_sign = np.sign(p[i] - p[j])
            if pred_sign == true_sign:
                score += 1.0
            elif pred_sign == 0:
                score += 0.5
    return score / comparable if comparable else 0.5


@dataclass(frozen=True)
class RegressionMetricSet:
    n: int
    mse_norm: float
    rmse_norm: float
    mae_norm: float
    ci: float
    mse_raw: float | None = None
    rmse_raw: float | None = None
    mae_raw: float | None = None
    scaler_range: float | None = None

    def as_dict(self) -> Dict[str, float | int | str]:
        out: Dict[str, float | int | str] = {
            "metric_schema_version": METRIC_SCHEMA_VERSION,
            "n": int(self.n),
            "mse_norm": float(self.mse_norm),
            "rmse_norm": float(self.rmse_norm),
            "mae_norm": float(self.mae_norm),
            "ci": float(self.ci),
        }
        if self.mse_raw is not None:
            out.update(
                {
                    "mse_raw": float(self.mse_raw),
                    "rmse_raw": float(self.rmse_raw),
                    "mae_raw": float(self.mae_raw),
                    "scaler_range": float(self.scaler_range),
                }
            )
        return out


class Evaluator:
    """Final continuous-DTA evaluator.

    ``task`` is retained as a constructor argument for compatibility with old
    call sites, but only ``regression`` is accepted for final experiments.
    """

    def __init__(self, task: str = "regression", scaler: Mapping | None = None, **_):
        if str(task).lower() != "regression":
            raise RuntimeError(
                "The final ZhiYao-Graph evaluator is regression-only. "
                "Historical thresholded classification/AUC is not a paper metric."
            )
        self.task = "regression"
        self.scaler = scaler

    def evaluate(self, y_true, y_pred) -> Dict[str, float | int | str]:
        y = _as_finite_1d(y_true, "y_true")
        p = _as_finite_1d(y_pred, "y_pred")
        if y.size != p.size:
            raise ValueError(f"Length mismatch: y_true={y.size}, y_pred={p.size}")

        mse = float(mean_squared_error(y, p))
        metric_set = RegressionMetricSet(
            n=int(y.size),
            mse_norm=mse,
            rmse_norm=float(np.sqrt(mse)),
            mae_norm=float(mean_absolute_error(y, p)),
            ci=float(concordance_index(y, p)),
        )

        if self.scaler is not None and self.scaler.get("method", "none") != "none":
            method = str(self.scaler.get("method", "none")).lower()
            if method == "minmax":
                y_raw_values = y * float(self.scaler["range"]) + float(self.scaler["min"])
                p_raw_values = p * float(self.scaler["range"]) + float(self.scaler["min"])
            elif method == "zscore":
                y_raw_values = y * float(self.scaler["std"]) + float(self.scaler["mean"])
                p_raw_values = p * float(self.scaler["std"]) + float(self.scaler["mean"])
            else:
                raise ValueError(f"Unsupported scaler method: {method!r}")

            y_raw = _as_finite_1d(y_raw_values, "y_true_raw")
            p_raw = _as_finite_1d(p_raw_values, "y_pred_raw")
            mse_raw = float(mean_squared_error(y_raw, p_raw))
            metric_set = RegressionMetricSet(
                n=metric_set.n,
                mse_norm=metric_set.mse_norm,
                rmse_norm=metric_set.rmse_norm,
                mae_norm=metric_set.mae_norm,
                ci=metric_set.ci,
                mse_raw=mse_raw,
                rmse_raw=float(np.sqrt(mse_raw)),
                mae_raw=float(mean_absolute_error(y_raw, p_raw)),
                scaler_range=float(self.scaler.get("range", np.nan)),
            )

        return metric_set.as_dict()

    @staticmethod
    def print_metrics(metrics: Mapping) -> None:
        print("  ── normalized label scale ──")
        for key, label in (
            ("mse_norm", "MSE"),
            ("rmse_norm", "RMSE"),
            ("mae_norm", "MAE"),
            ("ci", "CI"),
        ):
            if key in metrics:
                print(f"    {label:>8}: {float(metrics[key]):.6f}")

        if "mse_raw" in metrics:
            range_value = metrics.get("scaler_range")
            range_text = (
                f"range={float(range_value):.6f}"
                if range_value is not None and np.isfinite(float(range_value))
                else "raw label scale"
            )
            print(f"  ── original label scale ({range_text}) ──")
            for key, label in (
                ("mse_raw", "MSE"),
                ("rmse_raw", "RMSE"),
                ("mae_raw", "MAE"),
            ):
                if key in metrics:
                    print(f"    {label:>8}: {float(metrics[key]):.6f}")
            print(f"    {'CI':>8}: {float(metrics['ci']):.6f} (scale invariant)")
