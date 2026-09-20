"""ZhiYao-Graph data utility functions.

This module is used by downstream training/evaluation code AFTER
``scripts/prepare_final_datasets.py`` has created the fixed final datasets.

Design rules
------------
1. Final experiments read fixed files from ``data_final``.
2. Training/evaluation code must not create a new random split internally.
3. Min-max scaling parameters must come from the saved TRAIN-ONLY scaler.
4. Source/raw loaders are retained only for provenance/legacy conversion and
   must not be used as the main training path.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

import config as cfg


FINAL_DATASETS = {str(x).upper() for x in cfg.DATA_CONFIG["datasets"]}
FINAL_SETTINGS = {str(x).lower() for x in cfg.DATA_CONFIG["settings"]}
REQUIRED_FINAL_COLUMNS = {
    "SMILES",
    "Canonical SMILES",
    "Target Sequence",
    "Label",
    "Label_norm",
}


def _project_root() -> Path:
    """Return the project root assuming this file lives in ``data/utils.py``."""
    return Path(__file__).resolve().parents[1]


def _resolve_data_root(data_root: str | os.PathLike | None) -> Path:
    if data_root is None:
        return _project_root() / "data_final"
    return Path(data_root).expanduser().resolve()


def _validate_dataset_and_setting(dataset_name: str, setting: str) -> Tuple[str, str]:
    dataset = str(dataset_name).upper()
    setting = str(setting).lower()
    if dataset not in FINAL_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name!r}. "
            f"Expected one of {sorted(FINAL_DATASETS)}."
        )
    if setting not in FINAL_SETTINGS:
        raise ValueError(
            f"Unsupported setting: {setting!r}. "
            f"Expected one of {sorted(FINAL_SETTINGS)}."
        )
    return dataset, setting


def load_scaler(scaler_path: str | os.PathLike) -> Dict[str, float | str]:
    """Load and validate a saved train-only affinity scaler."""
    path = Path(scaler_path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file does not exist: {path}")

    scaler = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(scaler, dict):
        raise ValueError(f"Invalid scaler format in {path}: expected JSON object")

    method = scaler.get("method")
    if method == "minmax":
        required = {"min", "max", "range"}
        missing = required - set(scaler)
        if missing:
            raise ValueError(f"Scaler {path} is missing fields: {sorted(missing)}")
        if float(scaler["range"]) <= 0:
            raise ValueError(f"Scaler {path} has non-positive range")
        fit_on = scaler.get("fit_on")
        if fit_on != "train_only":
            raise ValueError(
                f"Final scaler {path} must be fitted on train only: fit_on={fit_on!r}"
            )
    elif method == "zscore":
        required = {"mean", "std"}
        missing = required - set(scaler)
        if missing:
            raise ValueError(f"Scaler {path} is missing fields: {sorted(missing)}")
        if float(scaler["std"]) <= 0:
            raise ValueError(f"Scaler {path} has non-positive std")
        if scaler.get("fit_on") != "train_only":
            raise ValueError(
                f"Final scaler {path} must be fitted on train only: "
                f"fit_on={scaler.get('fit_on')!r}"
            )
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unsupported scaler method in {path}: {method!r}")

    return scaler


def _validate_final_frame(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_FINAL_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Final split file {path} is missing required columns: {sorted(missing)}"
        )

    if df[list(REQUIRED_FINAL_COLUMNS)].isna().any().any():
        raise ValueError(f"Final split file {path} contains missing required values")

    # Repeated identical model inputs are allowed WITHIN a split when
    # prepare_final_datasets.py uses same_input_policy=keep_grouped.
    # Leakage prevention is enforced below by checking that no input group
    # crosses train/val/test.



def _model_input_pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["Canonical SMILES"].astype(str),
            frame["Target Sequence"].astype(str),
        )
    )


def load_fixed_train_val(
    dataset_name: str,
    setting: str = "warm",
    data_root: str | os.PathLike | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | str]]:
    """Load only the fixed training and validation splits plus train-only scaler.

    This is the preferred training-phase loader. It deliberately does not read
    ``test.csv`` so the held-out test labels remain sealed until the explicit
    evaluation phase.
    """
    dataset, setting = _validate_dataset_and_setting(dataset_name, setting)
    root = _resolve_data_root(data_root)
    split_dir = root / dataset / setting
    paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "scaler": split_dir / "scaler.json",
    }
    missing_paths = [str(x) for x in paths.values() if not x.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Final prepared train/val data are incomplete. Missing:\n  "
            + "\n  ".join(missing_paths)
        )

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    _validate_final_frame(train_df, paths["train"])
    _validate_final_frame(val_df, paths["val"])

    pair_overlap = len(_model_input_pair_set(train_df) & _model_input_pair_set(val_df))
    if pair_overlap:
        raise ValueError(
            f"Model-input leakage detected in {dataset}/{setting} train-val: "
            f"{pair_overlap}"
        )

    if setting == "cold_drug":
        drug_overlap = len(
            set(train_df["Canonical SMILES"].astype(str))
            & set(val_df["Canonical SMILES"].astype(str))
        )
        if drug_overlap:
            raise ValueError(
                f"Canonical-drug leakage detected in {dataset}/cold_drug "
                f"train-val: {drug_overlap}"
            )

    if setting == "cold_target":
        protein_overlap = len(
            set(train_df["Target Sequence"].astype(str))
            & set(val_df["Target Sequence"].astype(str))
        )
        if protein_overlap:
            raise ValueError(
                f"Protein-sequence leakage detected in {dataset}/cold_target "
                f"train-val: {protein_overlap}"
            )

    scaler = load_scaler(paths["scaler"])
    return train_df, val_df, scaler

def load_fixed_split(
    dataset_name: str,
    setting: str = "warm",
    data_root: str | os.PathLike | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float | str]]:
    """Load one immutable final train/val/test split and its train-only scaler.

    Parameters
    ----------
    dataset_name:
        ``KIBA`` or ``DAVIS`` (case-insensitive).
    setting:
        ``warm``, ``cold_drug`` or ``cold_target``.
    data_root:
        Root generated by ``scripts/prepare_final_datasets.py``.  Defaults to
        ``<project_root>/data_final``.

    Returns
    -------
    train_df, val_df, test_df, scaler
        The three fixed split DataFrames and the saved scaler dictionary.

    Notes
    -----
    This function NEVER re-splits, resamples, concatenates source train/test
    files, or refits a scaler.
    """
    dataset, setting = _validate_dataset_and_setting(dataset_name, setting)
    root = _resolve_data_root(data_root)
    split_dir = root / dataset / setting

    paths = {
        "train": split_dir / "train.csv",
        "val": split_dir / "val.csv",
        "test": split_dir / "test.csv",
        "scaler": split_dir / "scaler.json",
    }
    missing_paths = [str(p) for p in paths.values() if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Final prepared data are incomplete. Missing:\n  " + "\n  ".join(missing_paths)
        )

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])
    for frame, name in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        _validate_final_frame(frame, paths[name])

    # Cross-split model-input leakage must never occur.
    train_pairs, val_pairs, test_pairs = map(
        _model_input_pair_set, (train_df, val_df, test_df)
    )
    overlaps = {
        "train-val": len(train_pairs & val_pairs),
        "train-test": len(train_pairs & test_pairs),
        "val-test": len(val_pairs & test_pairs),
    }
    if any(overlaps.values()):
        raise ValueError(f"Model-input leakage detected in {dataset}/{setting}: {overlaps}")

    if setting == "cold_drug":
        train_d = set(train_df["Canonical SMILES"].astype(str))
        val_d = set(val_df["Canonical SMILES"].astype(str))
        test_d = set(test_df["Canonical SMILES"].astype(str))
        drug_overlaps = {
            "train-val": len(train_d & val_d),
            "train-test": len(train_d & test_d),
            "val-test": len(val_d & test_d),
        }
        if any(drug_overlaps.values()):
            raise ValueError(
                f"Canonical-drug leakage detected in {dataset}/cold_drug: "
                f"{drug_overlaps}"
            )

    if setting == "cold_target":
        train_p = set(train_df["Target Sequence"].astype(str))
        val_p = set(val_df["Target Sequence"].astype(str))
        test_p = set(test_df["Target Sequence"].astype(str))
        protein_overlaps = {
            "train-val": len(train_p & val_p),
            "train-test": len(train_p & test_p),
            "val-test": len(val_p & test_p),
        }
        if any(protein_overlaps.values()):
            raise ValueError(
                f"Protein-sequence leakage detected in {dataset}/cold_target: "
                f"{protein_overlaps}"
            )

    scaler = load_scaler(paths["scaler"])
    return train_df, val_df, test_df, scaler


def apply_affinity_scaler(Y, scaler: Mapping[str, object]):
    """Transform labels with an already-fitted scaler.

    The scaler should normally be loaded from ``data_final/.../scaler.json``.
    No fitting is performed here.
    """
    arr = np.asarray(Y, dtype=np.float64)
    method = scaler.get("method", "none")

    if method == "minmax":
        y_range = float(scaler["range"])
        if y_range <= 0:
            raise ValueError("Min-max scaler range must be positive")
        return (arr - float(scaler["min"])) / y_range
    if method == "zscore":
        std = float(scaler["std"])
        if std <= 0:
            raise ValueError("Z-score scaler std must be positive")
        return (arr - float(scaler["mean"])) / std
    if method == "none":
        return arr
    raise ValueError(f"Unsupported scaler method: {method!r}")


def normalize_affinity(Y: np.ndarray, method="minmax", scaler=None):
    """Apply a PRE-FITTED scaler; fitting on arbitrary input is intentionally blocked.

    This function previously fitted min/max on whichever array was passed in.
    That made it easy for training/evaluation code to fit on the full dataset,
    including validation/test labels.  The final pipeline no longer permits
    that behavior.

    Use::

        scaler = load_scaler("data_final/KIBA/warm/scaler.json")
        y_norm = normalize_affinity(y, scaler=scaler)

    For final experiments, the normalized column is already present as
    ``Label_norm`` in each prepared CSV, so most training code should not need
    to call this function at all.
    """
    if scaler is None:
        raise RuntimeError(
            "normalize_affinity no longer fits a scaler from its input. "
            "Load the train-only scaler from data_final/<DATASET>/<SETTING>/scaler.json "
            "or use the prepared Label_norm column."
        )
    if method is not None and str(method).lower() != str(scaler.get("method", method)).lower():
        raise ValueError(
            f"Requested method={method!r} does not match scaler method={scaler.get('method')!r}"
        )
    return apply_affinity_scaler(Y, scaler)


def denormalize_affinity(Y_norm, scaler: Mapping[str, object]):
    """Convert normalized affinity values back to the original saved label scale."""
    arr = np.asarray(Y_norm, dtype=np.float64)
    method = scaler.get("method", "none")

    if method == "minmax":
        return arr * float(scaler["range"]) + float(scaler["min"])
    if method == "zscore":
        return arr * float(scaler["std"]) + float(scaler["mean"])
    if method == "none":
        return arr
    raise ValueError(f"Unsupported scaler method: {method!r}")


def create_dataset_from_csv(
    csv_path: str | os.PathLike,
    smiles_col: str = "SMILES",
    target_col: str = "Target Sequence",
    affinity_col: str = "Label_norm",
):
    """Load model inputs and labels from one prepared final split CSV."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    required = [smiles_col, target_col, affinity_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} is missing columns: {missing}")

    return (
        df[smiles_col].astype(str).to_numpy(),
        df[target_col].astype(str).to_numpy(),
        df[affinity_col].to_numpy(dtype=np.float32),
    )


def load_raw_data(dataset_name="kiba", data_dir="data/raw"):
    """Legacy raw-matrix loader retained for provenance/conversion only.

    Final training/evaluation code should use :func:`load_fixed_split` instead.
    """
    warnings.warn(
        "load_raw_data() is a legacy/provenance loader. Final experiments must "
        "use data_final fixed splits via load_fixed_split().",
        DeprecationWarning,
        stacklevel=2,
    )

    data_dir = os.path.join(data_dir, str(dataset_name).lower())
    with open(os.path.join(data_dir, "drugs.txt"), "r", encoding="utf-8") as f:
        drug_dict = json.load(f)
    with open(os.path.join(data_dir, "targets.txt"), "r", encoding="utf-8") as f:
        target_dict = json.load(f)

    drug_keys = list(drug_dict.keys())
    target_keys = list(target_dict.keys())
    drugs = np.array([drug_dict[k] for k in drug_keys])
    targets = np.array([target_dict[k] for k in target_keys])

    try:
        Y = np.load(
            os.path.join(data_dir, "Y.txt"),
            allow_pickle=True,
            encoding="latin1",
        )
    except Exception:
        Y = np.loadtxt(os.path.join(data_dir, "Y.txt"))

    Y = np.array(Y, dtype=np.float64)
    if Y.shape[0] == len(target_keys) and Y.shape[1] == len(drug_keys):
        Y = Y.T
    return drugs, targets, Y


def split_data(*args, **kwargs):
    """Disabled for final experiments to prevent accidental re-splitting.

    Fixed train/val/test CSV files are generated once by
    ``scripts/prepare_final_datasets.py`` and must be reused by every model.
    """
    raise RuntimeError(
        "split_data() is disabled in the final pipeline. "
        "Use load_fixed_split(dataset_name, setting) to read the fixed split files."
    )


# Canonical protein vocabulary for V3.
# 0 is reserved exclusively for padding.  Unknown/ambiguous residues map to 21.
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_INDEX = 0
UNK_INDEX = 21
PROTEIN_VOCAB_SIZE = 22
AA_TO_INDEX = {aa: i + 1 for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
AA_TO_INDEX["X"] = UNK_INDEX
DEFAULT_PROTEIN_MAX_LEN = 1500


def protein_seq_to_indices(
    seq: str,
    max_len: int = DEFAULT_PROTEIN_MAX_LEN,
    truncation: str = "right",
):
    """Map a protein sequence to the canonical V3 integer encoding.

    Vocabulary
    ----------
    PAD=0; 20 standard amino acids=1..20; X/unknown=21.

    ``truncation='right'`` keeps the first ``max_len`` residues and is the
    predefined primary protocol.  The choice is explicit and saved in config;
    a later length-sensitivity experiment can compare alternatives.
    """
    max_len = int(max_len)
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    if truncation not in {"right", "head_tail"}:
        raise ValueError("truncation must be 'right' or 'head_tail'")

    clean = "".join(str(seq).split()).upper()
    if not clean:
        raise ValueError("Protein sequence is empty")

    if len(clean) > max_len:
        if truncation == "right":
            clean = clean[:max_len]
        else:
            head = max_len // 2
            tail = max_len - head
            clean = clean[:head] + clean[-tail:]

    indices = [AA_TO_INDEX.get(aa, UNK_INDEX) for aa in clean]
    indices.extend([PAD_INDEX] * (max_len - len(indices)))
    return indices
