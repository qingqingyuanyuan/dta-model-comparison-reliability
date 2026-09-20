#!/usr/bin/env python3
"""Prepare reproducible full-data KIBA and Davis datasets for ZhiYao-Graph.

This script is the single entry point for the data-preparation stage. It uses
all available source records, applies explicit quality control, creates fixed
warm/cold-drug/cold-target splits, and fits label scalers on training labels only.

Default full-data policy
------------------------
The default ``--same-input-policy keep_grouped`` is chosen for the paper's
"use the complete datasets" plan:

* exact duplicate records are removed;
* repeated model inputs (Canonical SMILES + protein sequence) are RETAINED;
* all rows sharing the same model input are forced into the SAME warm split;
* cold-drug splits are grouped by canonical drug identity;
* cold-target splits are grouped by protein sequence;
* conflicting repeated labels are recorded, not silently averaged.

This is especially important for DAVIS: several distinct Target IDs share the
same protein sequence (e.g. variant aliases), so median aggregation would
collapse benchmark target identities and reduce 30,056 source rows to 25,772
unique input pairs. Keeping them grouped preserves the full benchmark records
while preventing identical model inputs from leaking across train/val/test.

For a sensitivity analysis, ``--same-input-policy median`` or ``mean`` can be
used with a DIFFERENT output directory. Those modes consolidate identical
model inputs and should not be confused with the benchmark-full main data.

The script does NOT strip salts, neutralize charges, remove stereochemistry,
or convert affinity label definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase


REQUIRED_COLUMNS = ["Target ID", "Target Sequence", "SMILES", "Label"]
DATASETS = ("KIBA", "DAVIS")
PAIR_COLUMNS = ["Canonical SMILES", "Target Sequence"]


@dataclass
class CleaningStats:
    dataset: str
    same_input_policy: str
    source_train_rows: int
    source_test_rows: int
    source_total_rows: int
    missing_required_rows: int
    non_numeric_label_rows: int
    invalid_smiles_rows: int
    exact_duplicate_rows_removed: int
    cleaned_rows: int
    repeated_input_groups: int
    conflicting_label_groups: int
    rows_in_repeated_input_groups: int
    rows_in_conflicting_label_groups: int
    rows_removed_by_same_input_aggregation: int
    unique_canonical_smiles: int
    unique_target_ids: int
    unique_protein_sequences: int
    protein_sequences_gt_1000: int
    rows_with_protein_gt_1000: int
    source_sequence_overlap_count: int
    multi_id_sequence_count: int


@dataclass
class SplitStats:
    dataset: str
    setting: str
    train_rows: int
    val_rows: int
    test_rows: int
    total_rows: int
    train_unique_drugs: int
    val_unique_drugs: int
    test_unique_drugs: int
    train_unique_proteins: int
    val_unique_proteins: int
    test_unique_proteins: int
    repeated_input_rows_train: int
    repeated_input_rows_val: int
    repeated_input_rows_test: int
    pair_overlap_train_val: int
    pair_overlap_train_test: int
    pair_overlap_val_test: int
    drug_overlap_train_val: int
    drug_overlap_train_test: int
    drug_overlap_val_test: int
    protein_overlap_train_val: int
    protein_overlap_train_test: int
    protein_overlap_val_test: int
    scaler_min: float
    scaler_max: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare full cleaned KIBA/DAVIS warm, cold-drug and cold-target splits."
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "data" / "colab" / "BatchDTA_processed_data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data_final",
    )
    parser.add_argument("--warm-seed", type=int, default=42)
    parser.add_argument("--cold-seed", type=int, default=42)
    parser.add_argument("--cold-drug-seed", type=int, default=42)
    parser.add_argument(
        "--same-input-policy",
        choices=("keep_grouped", "median", "mean"),
        default="keep_grouped",
        help=(
            "keep_grouped retains all non-exact-duplicate source rows and keeps "
            "identical model inputs in the same split. median/mean aggregate them."
        ),
    )
    parser.add_argument(
        "--warm-ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--cold-drug-ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Drug-disjoint train/val/test ratios, assigned by canonical drug groups.",
    )
    parser.add_argument(
        "--cold-val-protein-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate a non-empty output directory.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_paths(source_root: Path, dataset: str) -> Dict[str, Path]:
    ds_dir = source_root / dataset
    return {
        "train": ds_dir / f"train_{dataset}_unseenP_seenD.csv",
        "test": ds_dir / f"test_{dataset}_unseenP_seenD.csv",
    }


def ensure_required_columns(df: pd.DataFrame, path: Path) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")


def _normalize_sequence_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def read_sources(source_root: Path, dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    paths = source_paths(source_root, dataset)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])
    ensure_required_columns(train_df, paths["train"])
    ensure_required_columns(test_df, paths["test"])

    train_df = train_df[REQUIRED_COLUMNS].copy()
    test_df = test_df[REQUIRED_COLUMNS].copy()
    train_df["Source Split"] = "source_train"
    test_df["Source Split"] = "source_test"
    train_df["Source Row"] = np.arange(len(train_df), dtype=np.int64)
    test_df["Source Row"] = np.arange(len(test_df), dtype=np.int64)
    return train_df, test_df


def canonicalize_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _join_unique(values: Iterable[object]) -> str:
    uniq = sorted({str(v) for v in values if pd.notna(v) and str(v) != ""})
    return ";".join(uniq)


def _input_group_report(full: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Canonical SMILES", "Target Sequence", "InputGroupSize", "TargetIDCount",
        "Target IDs", "LabelNunique", "LabelMin", "LabelMax", "LabelRange",
        "LabelMean", "LabelMedian", "Source Splits",
    ]
    rows = []
    for (smi, seq), g in full.groupby(PAIR_COLUMNS, sort=False):
        if len(g) <= 1:
            continue
        labels = g["Label"].astype(float)
        ids = sorted(set(g["Target ID"].astype(str)))
        rows.append({
            "Canonical SMILES": smi,
            "Target Sequence": seq,
            "InputGroupSize": int(len(g)),
            "TargetIDCount": int(len(ids)),
            "Target IDs": ";".join(ids),
            "LabelNunique": int(labels.nunique()),
            "LabelMin": float(labels.min()),
            "LabelMax": float(labels.max()),
            "LabelRange": float(labels.max() - labels.min()),
            "LabelMean": float(labels.mean()),
            "LabelMedian": float(labels.median()),
            "Source Splits": _join_unique(g["Source Split"]),
        })
    return pd.DataFrame(rows, columns=columns)


def _sequence_alias_report(full: pd.DataFrame) -> pd.DataFrame:
    columns = ["Target Sequence", "SequenceLength", "TargetIDCount", "Target IDs", "Rows"]
    rows = []
    for seq, g in full.groupby("Target Sequence", sort=False):
        ids = sorted(set(g["Target ID"].astype(str)))
        if len(ids) <= 1:
            continue
        rows.append({
            "Target Sequence": seq,
            "SequenceLength": len(seq),
            "TargetIDCount": len(ids),
            "Target IDs": ";".join(ids),
            "Rows": len(g),
        })
    return pd.DataFrame(rows, columns=columns)


def clean_full_dataset(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    same_input_policy: str,
):
    full = pd.concat([train_df, test_df], ignore_index=True)
    n_train, n_test = len(train_df), len(test_df)

    # Normalize source sequence identity before checking unseen-protein overlap.
    source_train_sequences = set(
        _normalize_sequence_series(train_df["Target Sequence"].dropna())
    )
    source_test_sequences = set(
        _normalize_sequence_series(test_df["Target Sequence"].dropna())
    )
    source_overlap_sequences = source_train_sequences & source_test_sequences

    missing_mask = full[REQUIRED_COLUMNS].isna().any(axis=1)
    missing_required_rows = int(missing_mask.sum())
    full = full.loc[~missing_mask].copy()

    full["Target ID"] = full["Target ID"].astype(str).str.strip()
    full["Target Sequence"] = _normalize_sequence_series(full["Target Sequence"])
    full["SMILES raw"] = full["SMILES"].astype(str).str.strip()

    numeric_label = pd.to_numeric(full["Label"], errors="coerce")
    non_numeric_label_rows = int(numeric_label.isna().sum())
    full["Label"] = numeric_label
    full = full.dropna(subset=["Label"]).copy()

    smiles_map = {
        s: canonicalize_smiles(s)
        for s in pd.unique(full["SMILES raw"])
    }
    full["Canonical SMILES"] = full["SMILES raw"].map(smiles_map)
    invalid_mask = full["Canonical SMILES"].isna()
    invalid_smiles_rows = int(invalid_mask.sum())
    full = full.loc[~invalid_mask].copy()
    full["SMILES"] = full["Canonical SMILES"]

    # Remove true duplicate content independent of which source file contained it.
    exact_subset = ["Target ID", "Target Sequence", "Canonical SMILES", "Label"]
    before_exact = len(full)
    full = full.drop_duplicates(subset=exact_subset, keep="first").copy()
    exact_removed = before_exact - len(full)

    # Diagnose repeated model inputs AFTER exact duplicates are removed.
    group_sizes = full.groupby(PAIR_COLUMNS, sort=False).size()
    repeated = group_sizes[group_sizes > 1]
    repeated_input_groups = int(len(repeated))
    rows_in_repeated_input_groups = int(repeated.sum()) if len(repeated) else 0

    label_nunique = full.groupby(PAIR_COLUMNS, sort=False)["Label"].nunique(dropna=True)
    conflict_index = label_nunique[label_nunique > 1].index
    conflicting_label_groups = int(len(conflict_index))
    conflict_pairs = set(conflict_index.tolist())
    rows_in_conflicting_label_groups = int(
        sum(
            (smi, seq) in conflict_pairs
            for smi, seq in zip(full["Canonical SMILES"], full["Target Sequence"])
        )
    )

    repeated_report = _input_group_report(full)
    alias_report = _sequence_alias_report(full)

    before_policy = len(full)
    if same_input_policy == "keep_grouped":
        cleaned = full.copy()
        group_meta = full.groupby(PAIR_COLUMNS, sort=False)["Label"].agg(
            InputGroupSize="size",
            InputLabelNunique="nunique",
            InputLabelStd="std",
            InputLabelMin="min",
            InputLabelMax="max",
        ).reset_index()
        cleaned = cleaned.merge(group_meta, on=PAIR_COLUMNS, how="left", validate="many_to_one")
        cleaned["InputLabelStd"] = cleaned["InputLabelStd"].fillna(0.0)
        cleaned["SameInputPolicy"] = "keep_grouped"
        keep_cols = [
            "Target ID", "Target Sequence", "SMILES", "Canonical SMILES", "SMILES raw",
            "Label", "Source Split", "Source Row", "InputGroupSize", "InputLabelNunique",
            "InputLabelStd", "InputLabelMin", "InputLabelMax", "SameInputPolicy",
        ]
        cleaned = cleaned[keep_cols].copy()
    else:
        agg_label = same_input_policy
        grouped = full.groupby(PAIR_COLUMNS, sort=False, as_index=False).agg({
            "Target ID": _join_unique,
            "SMILES raw": lambda s: next(iter(s), ""),
            "Label": agg_label,
            "Source Split": _join_unique,
        })
        group_meta = full.groupby(PAIR_COLUMNS, sort=False)["Label"].agg(
            InputGroupSize="size",
            InputLabelNunique="nunique",
            InputLabelStd="std",
            InputLabelMin="min",
            InputLabelMax="max",
        ).reset_index()
        grouped = grouped.merge(group_meta, on=PAIR_COLUMNS, how="left", validate="one_to_one")
        grouped["InputLabelStd"] = grouped["InputLabelStd"].fillna(0.0)
        grouped["SMILES"] = grouped["Canonical SMILES"]
        grouped["Source Row"] = -1
        grouped["SameInputPolicy"] = same_input_policy
        cleaned = grouped[[
            "Target ID", "Target Sequence", "SMILES", "Canonical SMILES", "SMILES raw",
            "Label", "Source Split", "Source Row", "InputGroupSize", "InputLabelNunique",
            "InputLabelStd", "InputLabelMin", "InputLabelMax", "SameInputPolicy",
        ]].copy()

    removed_by_policy = before_policy - len(cleaned)
    cleaned = cleaned.sort_values(
        ["Target Sequence", "Canonical SMILES", "Target ID", "Label"],
        kind="stable",
    ).reset_index(drop=True)

    seq_lengths = cleaned["Target Sequence"].str.len()
    stats = CleaningStats(
        dataset=dataset,
        same_input_policy=same_input_policy,
        source_train_rows=n_train,
        source_test_rows=n_test,
        source_total_rows=n_train + n_test,
        missing_required_rows=missing_required_rows,
        non_numeric_label_rows=non_numeric_label_rows,
        invalid_smiles_rows=invalid_smiles_rows,
        exact_duplicate_rows_removed=int(exact_removed),
        cleaned_rows=int(len(cleaned)),
        repeated_input_groups=repeated_input_groups,
        conflicting_label_groups=conflicting_label_groups,
        rows_in_repeated_input_groups=rows_in_repeated_input_groups,
        rows_in_conflicting_label_groups=rows_in_conflicting_label_groups,
        rows_removed_by_same_input_aggregation=int(removed_by_policy),
        unique_canonical_smiles=int(cleaned["Canonical SMILES"].nunique()),
        unique_target_ids=int(full["Target ID"].nunique()),
        unique_protein_sequences=int(cleaned["Target Sequence"].nunique()),
        protein_sequences_gt_1000=int(
            cleaned.loc[seq_lengths > 1000, "Target Sequence"].nunique()
        ),
        rows_with_protein_gt_1000=int((seq_lengths > 1000).sum()),
        source_sequence_overlap_count=int(len(source_overlap_sequences)),
        multi_id_sequence_count=int(len(alias_report)),
    )

    source_sets = {
        "source_train_sequences": source_train_sequences,
        "source_test_sequences": source_test_sequences,
        "source_overlap_sequences": source_overlap_sequences,
    }
    return cleaned, stats, source_sets, repeated_report, alias_report


def fit_train_minmax(train_df: pd.DataFrame) -> Dict[str, float | str]:
    y = train_df["Label"].to_numpy(dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot fit scaler on an empty training set")
    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_range = y_max - y_min
    if not np.isfinite([y_min, y_max, y_range]).all() or y_range <= 0:
        raise ValueError("Invalid training-label range")
    return {
        "method": "minmax",
        "fit_on": "train_only",
        "min": y_min,
        "max": y_max,
        "range": float(y_range),
    }


def apply_scaler(df: pd.DataFrame, scaler: Dict[str, float | str]) -> pd.DataFrame:
    out = df.copy()
    out["Label_norm"] = (
        out["Label"].astype(float) - float(scaler["min"])
    ) / float(scaler["range"])
    return out


def _input_group_keys(df: pd.DataFrame) -> List[Tuple[str, str]]:
    return list(zip(df["Canonical SMILES"].astype(str), df["Target Sequence"].astype(str)))


def split_warm_grouped(
    cleaned: pd.DataFrame,
    ratios: Sequence[float],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Warm split that NEVER separates identical model inputs across splits."""
    if len(ratios) != 3 or any(r <= 0 for r in ratios):
        raise ValueError("warm ratios must contain three positive values")
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()

    group_sizes = (
        cleaned.groupby(PAIR_COLUMNS, sort=False)
        .size()
        .reset_index(name="n_rows")
    )
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(group_sizes))
    shuffled = group_sizes.iloc[order].reset_index(drop=True)

    total_rows = int(shuffled["n_rows"].sum())
    train_target = total_rows * float(ratios[0])
    val_target_end = total_rows * float(ratios[0] + ratios[1])

    train_groups, val_groups, test_groups = [], [], []
    cumulative = 0
    # Use iloc because the group columns contain spaces.
    for i in range(len(shuffled)):
        smi = str(shuffled.iloc[i]["Canonical SMILES"])
        seq = str(shuffled.iloc[i]["Target Sequence"])
        n = int(shuffled.iloc[i]["n_rows"])
        if cumulative < train_target:
            train_groups.append((smi, seq))
        elif cumulative < val_target_end:
            val_groups.append((smi, seq))
        else:
            test_groups.append((smi, seq))
        cumulative += n

    # Defensive non-empty fallback for pathological tiny data.
    if not train_groups or not val_groups or not test_groups:
        raise ValueError("Warm grouped split produced an empty partition")

    assignment = {k: "train" for k in train_groups}
    assignment.update({k: "val" for k in val_groups})
    assignment.update({k: "test" for k in test_groups})

    keys = _input_group_keys(cleaned)
    split_name = pd.Series([assignment[k] for k in keys], index=cleaned.index)
    train = cleaned.loc[split_name == "train"].copy()
    val = cleaned.loc[split_name == "val"].copy()
    test = cleaned.loc[split_name == "test"].copy()
    return train, val, test


def split_cold_drug(
    cleaned: pd.DataFrame,
    ratios: Sequence[float],
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drug-disjoint split that keeps every canonical drug in exactly one partition.

    All interactions for the same canonical SMILES are assigned together, so
    train/validation/test have zero canonical-drug overlap. Protein identities
    may overlap, which is intentional: this setting measures generalization to
    unseen compounds while holding target familiarity unconstrained.
    """
    if len(ratios) != 3 or any(r <= 0 for r in ratios):
        raise ValueError("cold-drug ratios must contain three positive values")
    ratios = np.asarray(ratios, dtype=float)
    ratios = ratios / ratios.sum()

    drug_sizes = (
        cleaned.groupby("Canonical SMILES", sort=False)
        .size()
        .reset_index(name="n_rows")
    )
    if len(drug_sizes) < 3:
        raise ValueError("Cold-drug split requires at least three unique drugs")

    rng = np.random.RandomState(seed)
    shuffled = drug_sizes.iloc[rng.permutation(len(drug_sizes))].reset_index(drop=True)

    total_rows = int(shuffled["n_rows"].sum())
    train_target = total_rows * float(ratios[0])
    val_target_end = total_rows * float(ratios[0] + ratios[1])

    train_drugs, val_drugs, test_drugs = [], [], []
    cumulative = 0
    for i in range(len(shuffled)):
        drug = str(shuffled.iloc[i]["Canonical SMILES"])
        n = int(shuffled.iloc[i]["n_rows"])
        if cumulative < train_target:
            train_drugs.append(drug)
        elif cumulative < val_target_end:
            val_drugs.append(drug)
        else:
            test_drugs.append(drug)
        cumulative += n

    if not train_drugs or not val_drugs or not test_drugs:
        raise ValueError("Cold-drug split produced an empty partition")

    train_set, val_set, test_set = set(train_drugs), set(val_drugs), set(test_drugs)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("Cold-drug entity assignment is not disjoint")

    train = cleaned[cleaned["Canonical SMILES"].isin(train_set)].copy()
    val = cleaned[cleaned["Canonical SMILES"].isin(val_set)].copy()
    test = cleaned[cleaned["Canonical SMILES"].isin(test_set)].copy()
    if len(train) + len(val) + len(test) != len(cleaned):
        raise AssertionError("Cold-drug split did not preserve all cleaned rows")
    return train, val, test


def split_cold_target(
    cleaned: pd.DataFrame,
    source_sets: Dict[str, set[str]],
    val_protein_fraction: float,
    seed: int,
):
    if not (0.0 < val_protein_fraction < 1.0):
        raise ValueError("cold val protein fraction must be between 0 and 1")

    source_train_sequences = source_sets["source_train_sequences"]
    source_test_sequences = source_sets["source_test_sequences"]
    overlap = source_train_sequences & source_test_sequences

    # Any sequence seen in source train belongs to the train pool. This fixes the
    # DAVIS one-sequence source overlap while keeping final test sequences unseen.
    cold_test_sequences = source_test_sequences - source_train_sequences
    train_pool_sequences = source_train_sequences

    test = cleaned[cleaned["Target Sequence"].isin(cold_test_sequences)].copy()
    train_pool = cleaned[cleaned["Target Sequence"].isin(train_pool_sequences)].copy()

    covered = set(train_pool["Target Sequence"]) | set(test["Target Sequence"])
    all_sequences = set(cleaned["Target Sequence"])
    if covered != all_sequences:
        raise AssertionError(
            f"Cold-target sequence assignment missed {len(all_sequences-covered)} sequences"
        )

    proteins = np.array(sorted(set(train_pool["Target Sequence"])), dtype=object)
    rng = np.random.RandomState(seed)
    rng.shuffle(proteins)
    n_val = max(1, int(round(len(proteins) * val_protein_fraction)))
    n_val = min(n_val, len(proteins) - 1)
    val_sequences = set(proteins[:n_val].tolist())

    val = train_pool[train_pool["Target Sequence"].isin(val_sequences)].copy()
    train = train_pool[~train_pool["Target Sequence"].isin(val_sequences)].copy()

    notes = {
        "source_train_test_sequence_overlap": len(overlap),
        "overlap_sequences_assigned_to_train_pool": len(overlap),
        "cold_test_unique_sequences": len(cold_test_sequences),
        "cold_val_unique_sequences": len(val_sequences),
    }
    return train, val, test, notes


def pair_set(df: pd.DataFrame) -> set[Tuple[str, str]]:
    return set(_input_group_keys(df))


def protein_set(df: pd.DataFrame) -> set[str]:
    return set(df["Target Sequence"].astype(str))


def repeated_input_rows(df: pd.DataFrame) -> int:
    mask = df.duplicated(subset=PAIR_COLUMNS, keep=False)
    return int(mask.sum())


def validate_and_describe_split(
    dataset: str,
    setting: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler: Dict[str, float | str],
) -> SplitStats:
    for name, frame in (("train", train), ("val", val), ("test", test)):
        if frame.empty:
            raise AssertionError(f"{dataset} {setting}: {name} split is empty")
        if frame[["Canonical SMILES", "Target Sequence", "Label"]].isna().any().any():
            raise AssertionError(f"{dataset} {setting}: NaN remains in {name}")

    train_pairs, val_pairs, test_pairs = pair_set(train), pair_set(val), pair_set(test)
    train_drug = set(train["Canonical SMILES"].astype(str))
    val_drug = set(val["Canonical SMILES"].astype(str))
    test_drug = set(test["Canonical SMILES"].astype(str))
    train_prot, val_prot, test_prot = protein_set(train), protein_set(val), protein_set(test)

    pair_tv = len(train_pairs & val_pairs)
    pair_tt = len(train_pairs & test_pairs)
    pair_vt = len(val_pairs & test_pairs)
    if pair_tv or pair_tt or pair_vt:
        raise AssertionError(
            f"{dataset} {setting}: model-input leakage: "
            f"train-val={pair_tv}, train-test={pair_tt}, val-test={pair_vt}"
        )

    drug_tv = len(train_drug & val_drug)
    drug_tt = len(train_drug & test_drug)
    drug_vt = len(val_drug & test_drug)
    if setting == "cold_drug" and (drug_tv or drug_tt or drug_vt):
        raise AssertionError(
            f"{dataset} cold_drug: canonical-drug leakage: "
            f"train-val={drug_tv}, train-test={drug_tt}, val-test={drug_vt}"
        )

    prot_tv = len(train_prot & val_prot)
    prot_tt = len(train_prot & test_prot)
    prot_vt = len(val_prot & test_prot)
    if setting == "cold_target" and (prot_tv or prot_tt or prot_vt):
        raise AssertionError(
            f"{dataset} cold_target: protein leakage: "
            f"train-val={prot_tv}, train-test={prot_tt}, val-test={prot_vt}"
        )

    return SplitStats(
        dataset=dataset,
        setting=setting,
        train_rows=len(train), val_rows=len(val), test_rows=len(test),
        total_rows=len(train)+len(val)+len(test),
        train_unique_drugs=train["Canonical SMILES"].nunique(),
        val_unique_drugs=val["Canonical SMILES"].nunique(),
        test_unique_drugs=test["Canonical SMILES"].nunique(),
        train_unique_proteins=train["Target Sequence"].nunique(),
        val_unique_proteins=val["Target Sequence"].nunique(),
        test_unique_proteins=test["Target Sequence"].nunique(),
        repeated_input_rows_train=repeated_input_rows(train),
        repeated_input_rows_val=repeated_input_rows(val),
        repeated_input_rows_test=repeated_input_rows(test),
        pair_overlap_train_val=pair_tv,
        pair_overlap_train_test=pair_tt,
        pair_overlap_val_test=pair_vt,
        drug_overlap_train_val=drug_tv,
        drug_overlap_train_test=drug_tt,
        drug_overlap_val_test=drug_vt,
        protein_overlap_train_val=prot_tv,
        protein_overlap_train_test=prot_tt,
        protein_overlap_val_test=prot_vt,
        scaler_min=float(scaler["min"]),
        scaler_max=float(scaler["max"]),
    )


def write_split(
    ds_dir: Path,
    setting: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    scaler: Dict[str, float | str],
    metadata: Dict[str, object],
) -> None:
    out = ds_dir / setting
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = map(lambda x: apply_scaler(x, scaler), (train, val, test))

    sort_cols = ["Target Sequence", "Canonical SMILES", "Target ID", "Label"]
    train = train.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    val = val.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    test = test.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    train.to_csv(out / "train.csv", index=False)
    val.to_csv(out / "val.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    write_json(out / "scaler.json", scaler)
    write_json(out / "split_metadata.json", metadata)


def prepare_one_dataset(
    dataset: str,
    source_root: Path,
    output_root: Path,
    warm_seed: int,
    cold_seed: int,
    cold_drug_seed: int,
    warm_ratios: Sequence[float],
    cold_drug_ratios: Sequence[float],
    cold_val_protein_fraction: float,
    same_input_policy: str,
):
    print(f"\n=== {dataset}: loading complete source data ===")
    train_source, test_source = read_sources(source_root, dataset)
    print(
        f"source train={len(train_source):,}, source test={len(test_source):,}, "
        f"total={len(train_source)+len(test_source):,}"
    )

    cleaned, cleaning_stats, source_sets, repeat_report, alias_report = clean_full_dataset(
        dataset, train_source, test_source, same_input_policy
    )
    ds_out = output_root / dataset
    ds_out.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(ds_out / "cleaned_full.csv", index=False)
    repeat_report.to_csv(ds_out / "repeated_input_groups.csv", index=False)
    alias_report.to_csv(ds_out / "sequence_alias_report.csv", index=False)
    write_json(ds_out / "cleaning_stats.json", asdict(cleaning_stats))

    print(
        f"cleaned rows={len(cleaned):,}, unique drugs={cleaned['Canonical SMILES'].nunique():,}, "
        f"unique protein sequences={cleaned['Target Sequence'].nunique():,}, "
        f"same-input policy={same_input_policy}"
    )

    split_stats: List[SplitStats] = []

    warm_train, warm_val, warm_test = split_warm_grouped(cleaned, warm_ratios, warm_seed)
    warm_scaler = fit_train_minmax(warm_train)
    warm_stats = validate_and_describe_split(
        dataset, "warm", warm_train, warm_val, warm_test, warm_scaler
    )
    warm_meta = {
        "dataset": dataset,
        "setting": "warm",
        "description": (
            "Group-aware warm split: identical (Canonical SMILES, Target Sequence) "
            "inputs are kept in one partition; drug/protein entities may overlap."
        ),
        "split_seed": warm_seed,
        "ratios_requested": list(map(float, warm_ratios)),
        "label_scaler_fit": "train_only",
        "same_input_policy": same_input_policy,
    }
    write_split(ds_out, "warm", warm_train, warm_val, warm_test, warm_scaler, warm_meta)
    split_stats.append(warm_stats)

    cold_drug_train, cold_drug_val, cold_drug_test = split_cold_drug(
        cleaned, cold_drug_ratios, cold_drug_seed
    )
    cold_drug_scaler = fit_train_minmax(cold_drug_train)
    cold_drug_stats = validate_and_describe_split(
        dataset, "cold_drug", cold_drug_train, cold_drug_val, cold_drug_test, cold_drug_scaler
    )
    cold_drug_meta = {
        "dataset": dataset,
        "setting": "cold_drug",
        "description": (
            "Canonical-drug-disjoint split: each canonical SMILES appears in exactly "
            "one of train/validation/test; protein sequences may overlap."
        ),
        "split_seed": cold_drug_seed,
        "ratios_requested": list(map(float, cold_drug_ratios)),
        "label_scaler_fit": "train_only",
        "same_input_policy": same_input_policy,
    }
    write_split(
        ds_out, "cold_drug", cold_drug_train, cold_drug_val, cold_drug_test,
        cold_drug_scaler, cold_drug_meta
    )
    split_stats.append(cold_drug_stats)

    cold_train, cold_val, cold_test, cold_notes = split_cold_target(
        cleaned, source_sets, cold_val_protein_fraction, cold_seed
    )
    cold_scaler = fit_train_minmax(cold_train)
    cold_stats = validate_and_describe_split(
        dataset, "cold_target", cold_train, cold_val, cold_test, cold_scaler
    )
    cold_meta = {
        "dataset": dataset,
        "setting": "cold_target",
        "description": (
            "Protein-sequence-disjoint split. Source unseen-protein test intent is "
            "retained; any source sequence overlap is assigned to the train pool."
        ),
        "split_seed_for_validation_proteins": cold_seed,
        "validation_protein_fraction": float(cold_val_protein_fraction),
        "label_scaler_fit": "train_only",
        "same_input_policy": same_input_policy,
        **cold_notes,
    }
    write_split(ds_out, "cold_target", cold_train, cold_val, cold_test, cold_scaler, cold_meta)
    split_stats.append(cold_stats)

    return cleaning_stats, split_stats


def portable_manifest_path(path: Path, project_root: Path) -> str:
    """Store project-internal paths portably in provenance manifests."""
    path = Path(path).resolve()
    project_root = Path(project_root).resolve()
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        # External/custom source roots are still recorded explicitly.
        return str(path)


def write_readme(output_root: Path, source_root: Path, args: argparse.Namespace) -> None:
    text = f"""# ZhiYao-Graph final data

Generated by `scripts/prepare_final_datasets.py`. Do not edit generated CSV files manually.

## Full-data rule

- KIBA source rows: 118,254 before quality control.
- DAVIS source rows: 30,056 before quality control.
- No 50k/25k subsampling is performed.
- Exact duplicate records are removed.
- Default same-input policy: **{args.same_input_policy}**.

With `keep_grouped`, repeated identical model inputs are retained but forced into the same warm split, preventing train/val/test input leakage while preserving benchmark rows. This avoids collapsing DAVIS Target IDs that share the same sequence.

## Cleaning

1. Drop missing required fields / non-numeric labels.
2. Canonicalize SMILES with RDKit while retaining stereochemistry.
3. Remove invalid SMILES.
4. Remove exact duplicate records.
5. Diagnose repeated `(Canonical SMILES, Target Sequence)` inputs and conflicting labels.
6. Save `repeated_input_groups.csv` and `sequence_alias_report.csv` for audit.

No salt stripping, charge neutralization, stereochemistry removal, or affinity-label conversion is performed.

## Splits

- `warm/`: group-aware approximately {args.warm_ratios[0]:.2f}/{args.warm_ratios[1]:.2f}/{args.warm_ratios[2]:.2f}; identical model inputs never cross partitions; drug/protein entities may overlap; seed={args.warm_seed}.
- `cold_drug/`: canonical-drug-disjoint approximately {args.cold_drug_ratios[0]:.2f}/{args.cold_drug_ratios[1]:.2f}/{args.cold_drug_ratios[2]:.2f}; each canonical SMILES belongs to exactly one partition; protein sequences may overlap; seed={args.cold_drug_seed}.
- `cold_target/`: protein-sequence-disjoint. Validation proteins are sampled only from the source training protein pool; drug identities may overlap; seed={args.cold_seed}.

`cold_both` is intentionally not generated in the benchmark-full main pipeline because a highly connected drug-target bipartite graph cannot generally be partitioned into simultaneously drug-disjoint and protein-disjoint train/val/test sets while retaining every interaction. Such an experiment would require discarding cross-partition interactions and must be defined separately.

For every setting, min-max normalization is fit on **training labels only**. Validation/test normalized values are not clipped to [0,1].

## Downstream rule

All models/baselines must reuse these files. Training/evaluation code must not re-split, resample, or refit scalers.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output_root}\n"
                "Use --overwrite only when replacement is intended."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cleaning_rows: List[Dict[str, object]] = []
    split_rows: List[Dict[str, object]] = []

    for dataset in DATASETS:
        cleaning_stats, split_stats = prepare_one_dataset(
            dataset=dataset,
            source_root=source_root,
            output_root=output_root,
            warm_seed=args.warm_seed,
            cold_seed=args.cold_seed,
            cold_drug_seed=args.cold_drug_seed,
            warm_ratios=args.warm_ratios,
            cold_drug_ratios=args.cold_drug_ratios,
            cold_val_protein_fraction=args.cold_val_protein_fraction,
            same_input_policy=args.same_input_policy,
        )
        cleaning_rows.append(asdict(cleaning_stats))
        split_rows.extend(asdict(x) for x in split_stats)

    pd.DataFrame(cleaning_rows).to_csv(
        output_root / "dataset_cleaning_report.csv", index=False
    )
    pd.DataFrame(split_rows).to_csv(output_root / "split_report.csv", index=False)
    write_readme(output_root, source_root, args)

    project_root = Path(__file__).resolve().parents[1]
    source_manifest = {}
    for dataset in DATASETS:
        source_manifest[dataset] = {}
        for name, path in source_paths(source_root, dataset).items():
            source_manifest[dataset][name] = {
                "path": portable_manifest_path(path, project_root),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }

    manifest: Dict[str, object] = {
        "script": portable_manifest_path(Path(__file__), project_root),
        "source_root": portable_manifest_path(source_root, project_root),
        "output_root": portable_manifest_path(output_root, project_root),
        "warm_seed": args.warm_seed,
        "cold_seed": args.cold_seed,
        "cold_drug_seed": args.cold_drug_seed,
        "warm_ratios": list(map(float, args.warm_ratios)),
        "cold_drug_ratios": list(map(float, args.cold_drug_ratios)),
        "cold_val_protein_fraction": float(args.cold_val_protein_fraction),
        "same_input_policy": args.same_input_policy,
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "rdkit": rdBase.rdkitVersion,
        },
        "source_files": source_manifest,
        "files": {},
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(output_root))] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    write_json(output_root / "manifest.json", manifest)

    print("\n=== DONE ===")
    print(f"Output: {output_root}")
    print(pd.DataFrame(cleaning_rows).to_string(index=False))
    print("\nSplit checks:")
    print(pd.DataFrame(split_rows).to_string(index=False))


if __name__ == "__main__":
    main()
