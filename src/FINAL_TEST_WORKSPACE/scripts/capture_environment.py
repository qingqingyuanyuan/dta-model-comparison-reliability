#!/usr/bin/env python3
"""Capture the software/hardware/protocol environment for reproducibility.

The output intentionally includes the SHA256 of the frozen project ``config.py``
and the prespecified experiment-program summary. This makes an environment
snapshot useful for checking not only package versions but also *which protocol*
was intended when a run was launched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _version(module_name):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as e:
        return f"unavailable: {type(e).__name__}: {e}"


def _sha256(path: Path):
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty():
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except Exception:
        return None


def _protocol_snapshot():
    try:
        import config as cfg

        return {
            "model_architecture_version": getattr(
                cfg, "MODEL_ARCHITECTURE_VERSION", None
            ),
            "metric_schema_version": getattr(cfg, "EVALUATION", {}).get(
                "metric_schema_version"
            ),
            "primary_matrix": getattr(cfg, "EXPERIMENT_MATRIX", None),
            "baseline_matrix": getattr(cfg, "BASELINE_EXPERIMENT_MATRIX", None),
            "sensitivity_matrix": getattr(
                cfg, "SENSITIVITY_EXPERIMENT_MATRIX", None
            ),
            "extended_experiment_counts": getattr(
                cfg, "EXTENDED_EXPERIMENT_COUNTS", None
            ),
            "training": getattr(cfg, "TRAINING", None),
            "evaluation": getattr(cfg, "EVALUATION", None),
            "efficiency_benchmark": getattr(
                cfg, "EFFICIENCY_BENCHMARK", None
            ),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def capture():
    config_path = ROOT / "config.py"
    final_manifest = ROOT / "data_final" / "manifest.json"

    payload = {
        "capture_schema_version": 2,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "project_protocol": {
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "data_final_manifest_path": (
                str(final_manifest.resolve()) if final_manifest.exists() else None
            ),
            "data_final_manifest_sha256": _sha256(final_manifest),
            "snapshot": _protocol_snapshot(),
        },
        "packages": {
            "torch": _version("torch"),
            "torch_geometric": _version("torch_geometric"),
            "rdkit": _version("rdkit"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "sklearn": _version("sklearn"),
            "streamlit": _version("streamlit"),
            "matplotlib": _version("matplotlib"),
        },
    }

    try:
        import torch

        payload["torch_runtime"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if hasattr(torch.backends, "cudnn")
                else None
            ),
            "gpu_count": int(torch.cuda.device_count()),
            "gpus": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as e:
        payload["torch_runtime"] = {"error": str(e)}

    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "environment.json",
    )
    args = p.parse_args()
    payload = capture()
    path = args.output.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    print(path)
    print(f"config_sha256={payload['project_protocol']['config_sha256']}")
    counts = payload["project_protocol"]["snapshot"].get(
        "extended_experiment_counts"
    )
    if counts:
        print(
            "prespecified_training_runs="
            f"{counts.get('total_prespecified_training_runs')}"
        )


if __name__ == "__main__":
    main()
