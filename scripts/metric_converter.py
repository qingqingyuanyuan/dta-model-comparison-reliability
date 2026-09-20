#!/usr/bin/env python3
"""Convert normalized regression metrics back to the original affinity scale.

This is the finalized-data replacement for the old
``scripts/metric_converter.py``.

Data-integrity rules
--------------------
1. The conversion scale MUST come from a saved ``scaler.json`` generated for a
   fixed split under ``data_final``.
2. The script does NOT inspect or concatenate raw/source train/test files.
3. The script does NOT invent a default KIBA/DAVIS range.
4. The script does NOT contain hard-coded "current model" metrics.
5. The script does NOT contain hard-coded literature comparison numbers.
6. For min-max scaling, the scaler must be marked ``fit_on='train_only'``.
7. CI is dimensionless and is therefore reported unchanged; this script does
   not recompute CI.

Examples
--------
# KIBA warm
python scripts/metric_converter.py \
    --data kiba --setting warm \
    --mse 0.0023 --rmse 0.0476 --mae 0.0338 --ci 0.892

# Davis cold-target
python scripts/metric_converter.py \
    --data davis --setting cold_target \
    --mse 0.0031 --ci 0.81

# Explicit scaler path
python scripts/metric_converter.py \
    --scaler data_final/KIBA/warm/scaler.json \
    --rmse 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
VALID_DATASETS = ("KIBA", "DAVIS")
VALID_SETTINGS = tuple(str(x).lower() for x in cfg.DATA_CONFIG["settings"])


def load_scaler(path: Path) -> Dict[str, object]:
    """Load and strictly validate a finalized scaler."""
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        scaler = json.load(f)

    if not isinstance(scaler, dict):
        raise ValueError(f"Invalid scaler JSON in {path}: expected an object")

    method = str(scaler.get("method", "")).lower()
    if method == "minmax":
        required = {"min", "max", "range"}
        missing = required - set(scaler)
        if missing:
            raise ValueError(
                f"Min-max scaler {path} is missing fields: {sorted(missing)}"
            )

        y_min = float(scaler["min"])
        y_max = float(scaler["max"])
        y_range = float(scaler["range"])

        if not np.isfinite([y_min, y_max, y_range]).all():
            raise ValueError(f"Scaler {path} contains non-finite values")
        if y_range <= 0:
            raise ValueError(f"Scaler {path} has non-positive range")
        if not np.isclose(y_max - y_min, y_range, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"Scaler {path} is internally inconsistent: "
                f"max-min={y_max-y_min}, range={y_range}"
            )
        if scaler.get("fit_on") != "train_only":
            raise ValueError(
                f"Final min-max scaler must be fit_on='train_only'; "
                f"got {scaler.get('fit_on')!r} in {path}"
            )

    elif method == "zscore":
        required = {"mean", "std"}
        missing = required - set(scaler)
        if missing:
            raise ValueError(
                f"Z-score scaler {path} is missing fields: {sorted(missing)}"
            )
        mean = float(scaler["mean"])
        std = float(scaler["std"])
        if not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError(f"Invalid z-score scaler in {path}")
        if scaler.get("fit_on") not in (None, "train_only"):
            raise ValueError(
                f"Final scaler should be fit on train only; "
                f"got {scaler.get('fit_on')!r}"
            )

    elif method == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported scaler method {scaler.get('method')!r} in {path}"
        )

    return scaler


def resolve_scaler_path(
    *,
    scaler_path: Optional[Path],
    data: Optional[str],
    setting: Optional[str],
    data_root: Path,
) -> Path:
    """Resolve exactly one finalized scaler source."""
    if scaler_path is not None:
        if data is not None or setting is not None:
            raise ValueError(
                "Use either --scaler OR --data/--setting, not both."
            )
        return scaler_path.expanduser().resolve()

    if data is None or setting is None:
        raise ValueError(
            "A finalized scaler is required. Provide either:\n"
            "  --data kiba|davis --setting warm|cold_drug|cold_target\n"
            "or:\n"
            "  --scaler path/to/scaler.json\n"
            "No default range is allowed."
        )

    dataset = data.upper()
    setting_name = setting.lower()

    if dataset not in VALID_DATASETS:
        raise ValueError(f"Unsupported dataset: {data!r}")
    if setting_name not in VALID_SETTINGS:
        raise ValueError(f"Unsupported setting: {setting!r}")

    return (
        data_root.expanduser().resolve()
        / dataset
        / setting_name
        / "scaler.json"
    )


def metric_scale_factor(scaler: Dict[str, object]) -> float:
    """Return the positive affine scale factor from normalized to raw units."""
    method = str(scaler["method"]).lower()
    if method == "minmax":
        return float(scaler["range"])
    if method == "zscore":
        return float(scaler["std"])
    if method == "none":
        return 1.0
    raise ValueError(f"Unsupported scaler method: {method}")


def convert_metrics(
    *,
    scaler: Dict[str, object],
    mse: Optional[float] = None,
    rmse: Optional[float] = None,
    mae: Optional[float] = None,
    loss: Optional[float] = None,
    ci: Optional[float] = None,
) -> Dict[str, float]:
    """Convert normalized-scale regression metrics to raw affinity units."""
    factor = metric_scale_factor(scaler)
    result: Dict[str, float] = {}

    for name, value in (
        ("mse_norm", mse),
        ("rmse_norm", rmse),
        ("mae_norm", mae),
        ("val_loss_norm", loss),
        ("ci", ci),
    ):
        if value is not None and not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    if mse is not None:
        result["mse_norm"] = float(mse)
        result["mse_raw"] = float(mse) * factor**2

    if rmse is not None:
        result["rmse_norm"] = float(rmse)
        result["rmse_raw"] = float(rmse) * factor

    if mae is not None:
        result["mae_norm"] = float(mae)
        result["mae_raw"] = float(mae) * factor

    if loss is not None:
        if float(loss) < 0:
            raise ValueError("--loss (MSE loss) must be >= 0")
        result["val_loss_norm"] = float(loss)
        result["val_mse_raw"] = float(loss) * factor**2
        result["val_rmse_norm"] = math.sqrt(float(loss))
        result["val_rmse_raw"] = math.sqrt(float(loss)) * factor

    if ci is not None:
        result["ci"] = float(ci)

    return result


def print_scaler_summary(
    scaler: Dict[str, object],
    scaler_path: Path,
) -> None:
    method = str(scaler["method"]).lower()

    print("=" * 68)
    print("  指标量纲换算 — 固定训练集 scaler")
    print("=" * 68)
    print(f"  scaler 文件: {scaler_path}")
    print(f"  method:      {method}")
    if "fit_on" in scaler:
        print(f"  fit_on:      {scaler.get('fit_on')}")

    if method == "minmax":
        print(f"  train min:   {float(scaler['min']):.8f}")
        print(f"  train max:   {float(scaler['max']):.8f}")
        print(f"  train range: {float(scaler['range']):.8f}")
    elif method == "zscore":
        print(f"  train mean:  {float(scaler['mean']):.8f}")
        print(f"  train std:   {float(scaler['std']):.8f}")
    else:
        print("  原始标签未缩放")


def print_metric_table(
    result: Dict[str, float],
    scaler: Dict[str, object],
) -> None:
    if not result:
        print("\n没有提供待换算指标；仅完成 scaler 检查。")
        return

    print("\n" + "-" * 68)
    print(f"  {'指标':<18} {'归一化尺度':>20} {'原始亲和力量纲':>22}")
    print("-" * 68)

    if "mse_norm" in result:
        print(
            f"  {'MSE':<18} "
            f"{result['mse_norm']:>20.8f} "
            f"{result['mse_raw']:>22.8f}"
        )

    if "rmse_norm" in result:
        print(
            f"  {'RMSE':<18} "
            f"{result['rmse_norm']:>20.8f} "
            f"{result['rmse_raw']:>22.8f}"
        )

    if "mae_norm" in result:
        print(
            f"  {'MAE':<18} "
            f"{result['mae_norm']:>20.8f} "
            f"{result['mae_raw']:>22.8f}"
        )

    if "val_loss_norm" in result:
        print(
            f"  {'Val loss (MSE)':<18} "
            f"{result['val_loss_norm']:>20.8f} "
            f"{result['val_mse_raw']:>22.8f}"
        )
        print(
            f"  {'Val RMSE':<18} "
            f"{result['val_rmse_norm']:>20.8f} "
            f"{result['val_rmse_raw']:>22.8f}"
        )

    if "ci" in result:
        print(
            f"  {'CI':<18} "
            f"{result['ci']:>20.8f} "
            f"{result['ci']:>22.8f}"
        )

    factor = metric_scale_factor(scaler)
    print("-" * 68)
    print(f"  线性尺度因子 a = {factor:.8f}")
    print("  RMSE_raw = RMSE_norm × a")
    print("  MAE_raw  = MAE_norm × a")
    print("  MSE_raw  = MSE_norm × a²")
    print("  CI 为无量纲排序指标，不因正的仿射尺度换算而改变。")
    print(
        "  注意：本工具只做量纲换算，不重新计算 CI，"
        "也不提供文献基线比较。"
    )


def maybe_save_json(
    *,
    result: Dict[str, float],
    scaler: Dict[str, object],
    scaler_path: Path,
    output: Optional[Path],
) -> None:
    if output is None:
        return

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scaler_path": str(scaler_path),
        "scaler": scaler,
        "converted_metrics": result,
    }
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n已保存换算结果: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a finalized train-only scaler to convert normalized regression "
            "metrics back to raw affinity units."
        )
    )

    scaler_group = parser.add_argument_group("finalized scaler source")
    scaler_group.add_argument(
        "--data",
        choices=["kiba", "davis"],
        help="Dataset name; requires --setting",
    )
    scaler_group.add_argument(
        "--setting",
        choices=[str(x).lower() for x in cfg.DATA_CONFIG["settings"]],
        help="Fixed evaluation setting; requires --data",
    )
    scaler_group.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data_final",
        help="Root generated by scripts/prepare_final_datasets.py",
    )
    scaler_group.add_argument(
        "--scaler",
        type=Path,
        help="Explicit finalized scaler.json path",
    )

    metric_group = parser.add_argument_group("normalized metrics")
    metric_group.add_argument(
        "--mse",
        type=float,
        default=None,
        help="MSE on normalized label scale",
    )
    metric_group.add_argument(
        "--rmse",
        type=float,
        default=None,
        help="RMSE on normalized label scale",
    )
    metric_group.add_argument(
        "--mae",
        type=float,
        default=None,
        help="MAE on normalized label scale",
    )
    metric_group.add_argument(
        "--loss",
        type=float,
        default=None,
        help="Validation loss when the loss is normalized-scale MSE",
    )
    metric_group.add_argument(
        "--ci",
        type=float,
        default=None,
        help="Already-computed CI; displayed unchanged",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the conversion result",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scaler_path = resolve_scaler_path(
        scaler_path=args.scaler,
        data=args.data,
        setting=args.setting,
        data_root=args.data_root,
    )
    scaler = load_scaler(scaler_path)

    print_scaler_summary(scaler, scaler_path)

    result = convert_metrics(
        scaler=scaler,
        mse=args.mse,
        rmse=args.rmse,
        mae=args.mae,
        loss=args.loss,
        ci=args.ci,
    )
    print_metric_table(result, scaler)

    maybe_save_json(
        result=result,
        scaler=scaler,
        scaler_path=scaler_path,
        output=args.output,
    )


if __name__ == "__main__":
    main()
