#!/usr/bin/env python3
"""Package the finalized fixed DTA splits for Colab.

This script is the replacement for the old ``scripts/prepare_data_for_colab.py``.

Important rules
---------------
1. It DOES NOT download raw data.
2. It DOES NOT concatenate source train/test files and re-split them.
3. It DOES NOT sample 30,000 / 50,000 / 25,000 interactions.
4. It DOES NOT fit a scaler.
5. It DOES NOT silently replace invalid SMILES with a dummy molecule.
6. It preserves bond ``edge_attr`` in addition to node features and edge_index.
7. It packages the immutable files already created by
   ``scripts/prepare_final_datasets.py``.

Recommended workflow
--------------------
python scripts/prepare_final_datasets.py
python scripts/prepare_data_for_colab.py --dataset KIBA DAVIS --setting warm cold_drug cold_target
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg

from data.build_graph import smiles_to_graph  # noqa: E402


REQUIRED_COLUMNS = {
    "Canonical SMILES",
    "Target Sequence",
    "Label",
    "Label_norm",
}

VALID_DATASETS = tuple(str(x).upper() for x in cfg.DATA_CONFIG["datasets"])
VALID_SETTINGS = tuple(str(x).lower() for x in cfg.DATA_CONFIG["settings"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _validate_split_frame(df: pd.DataFrame, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    required = sorted(REQUIRED_COLUMNS)
    if df[required].isna().any().any():
        raise ValueError(f"{name} contains missing required values")

    # Repeated identical model inputs are permitted within a finalized split
    # when same_input_policy=keep_grouped. Cross-split overlap is checked
    # separately and remains forbidden.


def _validate_cross_split_independence(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    setting: str,
) -> None:
    def pair_set(df: pd.DataFrame):
        return set(
            zip(
                df["Canonical SMILES"].astype(str),
                df["Target Sequence"].astype(str),
            )
        )

    tr_pairs = pair_set(train)
    va_pairs = pair_set(val)
    te_pairs = pair_set(test)

    if tr_pairs & va_pairs:
        raise ValueError("Train/validation model-input overlap detected")
    if tr_pairs & te_pairs:
        raise ValueError("Train/test model-input overlap detected")
    if va_pairs & te_pairs:
        raise ValueError("Validation/test model-input overlap detected")

    if setting == "cold_drug":
        tr_drug = set(train["Canonical SMILES"].astype(str))
        va_drug = set(val["Canonical SMILES"].astype(str))
        te_drug = set(test["Canonical SMILES"].astype(str))
        if tr_drug & va_drug:
            raise ValueError("Cold-drug train/validation drug overlap detected")
        if tr_drug & te_drug:
            raise ValueError("Cold-drug train/test drug overlap detected")
        if va_drug & te_drug:
            raise ValueError("Cold-drug validation/test drug overlap detected")

    if setting == "cold_target":
        tr_seq = set(train["Target Sequence"].astype(str))
        va_seq = set(val["Target Sequence"].astype(str))
        te_seq = set(test["Target Sequence"].astype(str))

        if tr_seq & va_seq:
            raise ValueError("Cold-target train/validation protein overlap detected")
        if tr_seq & te_seq:
            raise ValueError("Cold-target train/test protein overlap detected")
        if va_seq & te_seq:
            raise ValueError("Cold-target validation/test protein overlap detected")


def _graph_to_numpy(graph, smiles: str) -> dict:
    if graph is None:
        raise ValueError(
            f"Could not build graph for canonical SMILES {smiles!r}. "
            "Invalid molecules must be removed during final data preparation; "
            "do not use a dummy fallback graph."
        )

    if graph.x.ndim != 2:
        raise ValueError(f"Invalid node-feature shape for {smiles!r}: {graph.x.shape}")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"Invalid edge_index shape for {smiles!r}: {graph.edge_index.shape}"
        )
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        raise ValueError(f"Missing edge_attr for {smiles!r}")
    if graph.edge_attr.ndim != 2:
        raise ValueError(
            f"Invalid edge_attr shape for {smiles!r}: {graph.edge_attr.shape}"
        )
    if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
        raise ValueError(
            f"Edge count mismatch for {smiles!r}: "
            f"{graph.edge_index.shape[1]} indices vs "
            f"{graph.edge_attr.shape[0]} edge features"
        )

    return {
        "x": graph.x.detach().cpu().numpy().astype(np.float32, copy=False),
        "edge_index": graph.edge_index.detach().cpu().numpy().astype(
            np.int64, copy=False
        ),
        "edge_attr": graph.edge_attr.detach().cpu().numpy().astype(
            np.float32, copy=False
        ),
    }


def build_unique_graph_cache(
    smiles_values: Iterable[str],
    *,
    show_progress: bool = True,
) -> Dict[str, dict]:
    """Construct each unique canonical molecule exactly once."""
    unique_smiles = list(OrderedDict.fromkeys(str(x) for x in smiles_values))

    iterator = tqdm(
        unique_smiles,
        desc="Building unique molecular graphs",
        disable=not show_progress,
    )

    cache: Dict[str, dict] = {}
    for smi in iterator:
        cache[smi] = _graph_to_numpy(smiles_to_graph(smi), smi)

    return cache


def _split_payload(df: pd.DataFrame) -> dict:
    """Store raw sequences/labels and references to the shared graph cache."""
    payload = {
        "canonical_smiles": df["Canonical SMILES"].astype(str).tolist(),
        "target_sequences": df["Target Sequence"].astype(str).tolist(),
        "labels_raw": df["Label"].astype(float).to_numpy(dtype=np.float32),
        "labels_norm": df["Label_norm"].astype(float).to_numpy(dtype=np.float32),
    }

    # Keep useful provenance columns when present, without making them required.
    optional_columns = [
        "Target ID",
        "SMILES raw",
        "Source Split",
        "Source Row",
        "InputGroupSize",
        "InputLabelNunique",
        "InputLabelStd",
        "InputLabelMin",
        "InputLabelMax",
        "SameInputPolicy",
    ]
    provenance = {}
    for col in optional_columns:
        if col in df.columns:
            provenance[col] = df[col].tolist()
    payload["provenance"] = provenance

    return payload


def package_one(
    dataset: str,
    setting: str,
    *,
    data_root: Path,
    output_dir: Path,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    dataset = dataset.upper()
    setting = setting.lower()

    if dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose {VALID_DATASETS}")
    if setting not in VALID_SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; choose {VALID_SETTINGS}")

    split_dir = data_root / dataset / setting

    paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
        "scaler": split_dir / "scaler.json",
        "split_metadata": split_dir / "split_metadata.json",
    }

    for key in ("train", "val", "test", "scaler"):
        if not paths[key].exists():
            raise FileNotFoundError(
                f"Missing finalized {dataset}/{setting} file: {paths[key]}"
            )

    train = pd.read_csv(paths["train"])
    val = pd.read_csv(paths["val"])
    test = pd.read_csv(paths["test"])

    _validate_split_frame(train, f"{dataset}/{setting}/train")
    _validate_split_frame(val, f"{dataset}/{setting}/val")
    _validate_split_frame(test, f"{dataset}/{setting}/test")
    _validate_cross_split_independence(train, val, test, setting)

    scaler = _load_json(paths["scaler"])
    if scaler.get("fit_on") != "train_only":
        raise ValueError(
            f"{dataset}/{setting} scaler is not marked fit_on='train_only'"
        )

    # Re-check normalized labels against the saved training-only scaler.
    y_min = float(scaler["min"])
    y_max = float(scaler["max"])
    y_range = float(scaler["range"])
    if y_range <= 0 or not np.isclose(y_max - y_min, y_range):
        raise ValueError(f"Invalid scaler for {dataset}/{setting}: {scaler}")

    for split_name, frame in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        expected = (frame["Label"].to_numpy(float) - y_min) / y_range
        stored = frame["Label_norm"].to_numpy(float)
        if not np.allclose(expected, stored, rtol=1e-7, atol=1e-8):
            raise ValueError(
                f"{dataset}/{setting}/{split_name}: Label_norm does not "
                "match the saved training-only scaler"
            )

    all_smiles = pd.concat(
        [
            train["Canonical SMILES"],
            val["Canonical SMILES"],
            test["Canonical SMILES"],
        ],
        ignore_index=True,
    ).astype(str)

    graph_cache = build_unique_graph_cache(
        all_smiles.tolist(),
        show_progress=show_progress,
    )

    split_metadata = {}
    if paths["split_metadata"].exists():
        split_metadata = _load_json(paths["split_metadata"])

    bundle = {
        "format_version": 2,
        "task": "DTA_regression",
        "dataset": dataset,
        "setting": setting,
        "graph_key": "Canonical SMILES",
        "graph_feature_spec": {
            "node_feature_dim": (
                int(next(iter(graph_cache.values()))["x"].shape[1])
                if graph_cache else None
            ),
            "edge_feature_dim": (
                int(next(iter(graph_cache.values()))["edge_attr"].shape[1])
                if graph_cache else None
            ),
        },
        "graph_cache": graph_cache,
        "splits": {
            "train": _split_payload(train),
            "val": _split_payload(val),
            "test": _split_payload(test),
        },
        "scaler": scaler,
        "split_metadata": split_metadata,
        "counts": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
            "total": int(len(train) + len(val) + len(test)),
            "unique_molecules": int(len(graph_cache)),
            "unique_proteins": int(
                pd.concat(
                    [
                        train["Target Sequence"],
                        val["Target Sequence"],
                        test["Target Sequence"],
                    ],
                    ignore_index=True,
                ).nunique()
            ),
        },
        "source_files": {
            key: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for key, path in paths.items()
            if path.exists()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{setting}_final.pkl"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Use --overwrite to replace it."
        )

    with out_path.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Small sidecar summary that can be inspected without unpickling the bundle.
    summary_path = out_path.with_suffix(".json")
    summary = {
        "format_version": bundle["format_version"],
        "task": bundle["task"],
        "dataset": dataset,
        "setting": setting,
        "counts": bundle["counts"],
        "graph_feature_spec": bundle["graph_feature_spec"],
        "scaler": scaler,
        "split_metadata": split_metadata,
        "bundle_sha256": _sha256(out_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[OK] {dataset}/{setting}: "
        f"train={len(train):,}, val={len(val):,}, test={len(test):,}, "
        f"unique molecules={len(graph_cache):,}"
    )
    print(f"     Saved: {out_path}")
    print(f"     Summary: {summary_path}")

    return out_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Package finalized ZhiYao-Graph DTA splits for Colab without "
            "sampling, re-splitting, or re-fitting normalization."
        )
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=VALID_DATASETS,
        default=list(VALID_DATASETS),
        help="Datasets to package. Default: KIBA DAVIS",
    )
    parser.add_argument(
        "--setting",
        nargs="+",
        choices=VALID_SETTINGS,
        default=list(VALID_SETTINGS),
        help="Evaluation settings. Default: warm cold_drug cold_target",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data_final",
        help="Root produced by scripts/prepare_final_datasets.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "colab" / "final_packages",
        help="Where Colab pickle bundles are written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing package",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm graph-building progress bars",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(
            f"Final data root does not exist: {args.data_root}\n"
            "Run scripts/prepare_final_datasets.py first."
        )

    print("Final data root:", args.data_root.resolve())
    print("Colab package output:", args.output_dir.resolve())
    print("Datasets:", ", ".join(args.dataset))
    print("Settings:", ", ".join(args.setting))
    print()

    outputs = []
    for dataset in args.dataset:
        for setting in args.setting:
            outputs.append(
                package_one(
                    dataset,
                    setting,
                    data_root=args.data_root,
                    output_dir=args.output_dir,
                    overwrite=args.overwrite,
                    show_progress=not args.no_progress,
                )
            )

    print("\nCompleted packages:")
    for path in outputs:
        print(" -", path)


if __name__ == "__main__":
    main()
