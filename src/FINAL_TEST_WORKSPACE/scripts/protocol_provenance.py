"""V1C protocol/provenance helpers for frozen formal training plans.

This module deliberately does not read held-out test data.  It only fingerprints
source/configuration files and validates frozen plan/checkpoint metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
PLAN_VERSION = 2

# Hash the scientific/runtime code that can change training behavior.  We hash
# model/data/train/baseline packages recursively, plus the exact orchestration
# scripts used by the 126-run program.  ShortComm/other unrelated scripts are
# intentionally not included.
_SOURCE_GLOBS = (
    "run.py",
    "train/**/*.py",
    "models/**/*.py",
    "data/**/*.py",
    "baselines/**/*.py",
)
_SOURCE_EXACT = (
    "scripts/runtime_health_gate.py",
    "scripts/run_experiment_matrix.py",
    "scripts/run_baseline_matrix.py",
    "scripts/run_sensitivity_matrix.py",
    "scripts/experiment_integrity.py",
    "scripts/protocol_provenance.py",
    "scripts/primary_shard_guard.py",
)


def sha256_file(path: Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_files(root: Path = ROOT) -> List[Path]:
    root = Path(root).resolve()
    found = set()
    for pattern in _SOURCE_GLOBS:
        for path in root.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                found.add(path.resolve())
    for rel in _SOURCE_EXACT:
        path = (root / rel).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Critical source file missing: {path}")
        found.add(path)
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def source_manifest(root: Path = ROOT) -> Dict[str, str]:
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _source_files(root)
    }


def source_sha256(root: Path = ROOT) -> str:
    manifest = source_manifest(root)
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def config_sha256(root: Path = ROOT) -> str:
    return sha256_file(Path(root) / "config.py")


def data_manifest_sha256(root: Path = ROOT) -> str:
    manifest = Path(root) / "data_final" / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Frozen data manifest missing: {manifest}"
        )
    return sha256_file(manifest)


def training_data_sha256(cfg_module, root: Path = ROOT) -> str:
    data_root = (Path(root) / "data_final").resolve()

    files = {}
    for dataset in cfg_module.EXPERIMENT_MATRIX["datasets"]:
        for setting in cfg_module.EXPERIMENT_MATRIX["settings"]:
            split_dir = data_root / str(dataset).upper() / str(setting).lower()

            for name in (
                "train.csv",
                "val.csv",
                "scaler.json",
                "split_metadata.json",
            ):
                path = split_dir / name
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Frozen train/val artifact missing: {path}"
                    )
                relpath = path.relative_to(data_root).as_posix()
                files[relpath] = sha256_file(path)

    canonical = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()


def require_canonical_data_root(data_root, root: Path = ROOT):
    expected = (Path(root) / "data_final").resolve()
    actual = Path(data_root).expanduser().resolve()

    if actual != expected:
        raise RuntimeError(
            f"NONCANONICAL_DATA_ROOT — STOP: "
            f"expected={expected}, got={actual}"
        )

    return expected


def optimizer_name(cfg_module) -> str:
    return str(cfg_module.TRAINING.get("optimizer", "")).lower()


def current_provenance(cfg_module, root: Path = ROOT) -> Dict[str, str]:
    return {
        "protocol_version": str(cfg_module.PROTOCOL_VERSION),
        "optimizer": optimizer_name(cfg_module),
        "config_sha256": config_sha256(root),
        "source_sha256": source_sha256(root),
        "data_manifest_sha256": data_manifest_sha256(root),
        "training_data_sha256": training_data_sha256(cfg_module, root),
    }


def resolve_project_path(value: Any, root: Path = ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(root) / path).resolve()


def project_relative(path: Any, root: Path = ROOT) -> str:
    root = Path(root).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Path must stay inside project root: {p}") from exc


def _validate_matrix_shape(matrix: Any) -> List[dict]:
    if not isinstance(matrix, list) or not matrix:
        raise RuntimeError("Frozen plan matrix must be a non-empty list")
    if not all(isinstance(row, dict) for row in matrix):
        raise RuntimeError("Every frozen plan matrix row must be an object")
    stems = [str(row.get("stem", "")) for row in matrix]
    if any(not stem for stem in stems):
        raise RuntimeError("Every frozen plan row must contain a non-empty stem")
    if len(set(stems)) != len(stems):
        raise RuntimeError("Frozen plan contains duplicate stems")
    return matrix


def build_plan_header(cfg_module, *, family: str, root: Path = ROOT) -> Dict[str, Any]:
    manifest = source_manifest(root)
    prov = current_provenance(cfg_module, root)
    return {
        "plan_version": PLAN_VERSION,
        "family": str(family),
        **prov,
        "source_manifest": manifest,
    }


def load_and_validate_plan(
    path: Path,
    cfg_module,
    *,
    expected_family: str,
    expected_matrix: Sequence[Mapping[str, Any]] | None,
    root: Path = ROOT,
) -> Tuple[dict, Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen plan not found: {path}")
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Frozen plan is not valid JSON: {path}") from exc
    if not isinstance(plan, dict):
        raise RuntimeError("Frozen plan root must be a JSON object")

    current = current_provenance(cfg_module, root)
    problems = []
    if int(plan.get("plan_version", -1)) != PLAN_VERSION:
        problems.append(
            f"plan_version={plan.get('plan_version')!r} != required {PLAN_VERSION}"
        )
    if str(plan.get("family", "")) != str(expected_family):
        problems.append(
            f"family={plan.get('family')!r} != expected {expected_family!r}"
        )
    for key in (
        "protocol_version",
        "optimizer",
        "config_sha256",
        "source_sha256",
        "data_manifest_sha256",
        "training_data_sha256",
    ):
        if str(plan.get(key, "")) != str(current[key]):
            problems.append(f"{key} mismatch")

    matrix = _validate_matrix_shape(plan.get("matrix"))
    if expected_matrix is not None:
        expected = [dict(row) for row in expected_matrix]
        if matrix != expected:
            problems.append("frozen matrix != current canonical matrix")
    if int(plan.get("n_runs", -1)) != len(matrix):
        problems.append(
            f"n_runs={plan.get('n_runs')!r} != matrix length {len(matrix)}"
        )

    if problems:
        raise RuntimeError(
            "FROZEN_PLAN_MISMATCH — STOP:\n  " + "\n  ".join(problems)
        )

    try:
        logical_plan_path = project_relative(path, root)
    except ValueError:
        logical_plan_path = str(path.resolve())
    context = {
        **current,
        "plan_sha256": sha256_file(path),
        "plan_path": logical_plan_path,
        "plan_family": str(expected_family),
    }
    return plan, context


def checkpoint_provenance_matches(
    checkpoint: Mapping[str, Any], expected: Mapping[str, str]
) -> bool:
    if not isinstance(checkpoint, Mapping):
        return False
    return all(
        str(checkpoint.get(key, "")) == str(expected.get(key, ""))
        for key in (
            "protocol_version",
            "optimizer",
            "config_sha256",
            "source_sha256",
            "data_manifest_sha256",
            "training_data_sha256",
            "plan_sha256",
        )
    )


def require_checkpoint_provenance(
    checkpoint: Mapping[str, Any], expected: Mapping[str, str]
) -> None:
    if checkpoint_provenance_matches(checkpoint, expected):
        return
    mismatches = []
    for key in (
        "protocol_version",
        "optimizer",
        "config_sha256",
        "source_sha256",
        "data_manifest_sha256",
        "training_data_sha256",
        "plan_sha256",
    ):
        if str(checkpoint.get(key, "")) != str(expected.get(key, "")):
            mismatches.append(key)
    raise RuntimeError(
        "CHECKPOINT_PROTOCOL_MISMATCH — STOP: " + ", ".join(mismatches)
    )
