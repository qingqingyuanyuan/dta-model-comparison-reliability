"""Fixed-split datasets for sequence-based controlled baselines.

DeepDTA-style experiments use canonical SMILES as a character sequence and the
same protein-sequence integer vocabulary as the canonical ZhiYao-Graph model.
No vocabulary is fitted from train/validation/test data: printable ASCII is a
fixed a priori SMILES character space, so tokenization cannot leak labels or
held-out distribution statistics.
"""

from __future__ import annotations

import random
from functools import partial
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.utils import (
    PAD_INDEX,
    load_fixed_split,
    load_fixed_train_val,
    protein_seq_to_indices,
)

SMILES_PAD_INDEX = 0
SMILES_UNK_INDEX = 1
# Fixed printable-ASCII vocabulary. Canonical RDKit SMILES are ASCII strings;
# using a fixed character table avoids data-dependent vocabulary construction.
PRINTABLE_ASCII = "".join(chr(i) for i in range(32, 127))
SMILES_CHAR_TO_INDEX = {
    ch: i + 2 for i, ch in enumerate(PRINTABLE_ASCII)
}
SMILES_VOCAB_SIZE = len(SMILES_CHAR_TO_INDEX) + 2


def smiles_to_indices(smiles: str, max_len: int) -> list[int]:
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    text = str(smiles)
    if not text:
        raise ValueError("SMILES cannot be empty")
    out = [
        SMILES_CHAR_TO_INDEX.get(ch, SMILES_UNK_INDEX)
        for ch in text[: int(max_len)]
    ]
    out.extend([SMILES_PAD_INDEX] * (int(max_len) - len(out)))
    return out


def sequence_truncation_report(
    frame: pd.DataFrame,
    *,
    max_smiles_len: int,
    max_protein_len: int,
) -> dict:
    smiles_len = frame["Canonical SMILES"].astype(str).str.len().to_numpy(int)
    protein_len = frame["Target Sequence"].astype(str).str.len().to_numpy(int)
    n = int(len(frame))
    return {
        "n_records": n,
        "max_smiles_len": int(max_smiles_len),
        "smiles_truncated_records": int((smiles_len > max_smiles_len).sum()),
        "smiles_truncated_fraction": (
            float((smiles_len > max_smiles_len).mean()) if n else 0.0
        ),
        "max_protein_len": int(max_protein_len),
        "protein_truncated_records": int((protein_len > max_protein_len).sum()),
        "protein_truncated_fraction": (
            float((protein_len > max_protein_len).mean()) if n else 0.0
        ),
    }


class SequenceDTADataset(Dataset):
    """Canonical-SMILES sequence + protein sequence + normalized affinity."""

    def __init__(
        self,
        smiles: Sequence[str],
        proteins: Sequence[str],
        affinities: Sequence[float],
        *,
        max_smiles_len: int,
        max_protein_len: int,
        protein_truncation: str = "right",
    ):
        if not (len(smiles) == len(proteins) == len(affinities)):
            raise ValueError("smiles/proteins/affinities length mismatch")
        self.smiles = [str(x) for x in smiles]
        self.proteins = [str(x) for x in proteins]
        self.affinities = [float(x) for x in affinities]
        self.max_smiles_len = int(max_smiles_len)
        self.max_protein_len = int(max_protein_len)
        self.protein_truncation = str(protein_truncation)

        self.smiles_cache = {
            s: torch.tensor(
                smiles_to_indices(s, self.max_smiles_len), dtype=torch.long
            )
            for s in dict.fromkeys(self.smiles)
        }
        self.protein_cache = {
            s: torch.tensor(
                protein_seq_to_indices(
                    s,
                    self.max_protein_len,
                    self.protein_truncation,
                ),
                dtype=torch.long,
            )
            for s in dict.fromkeys(self.proteins)
        }

    def __len__(self):
        return len(self.affinities)

    def __getitem__(self, idx):
        return (
            self.smiles_cache[self.smiles[idx]],
            self.protein_cache[self.proteins[idx]],
            torch.tensor(self.affinities[idx], dtype=torch.float32),
        )


def _collate_sequence(batch, *, trim_padding: bool):
    if not batch:
        raise ValueError("empty batch")
    smiles = torch.stack([x[0] for x in batch], dim=0)
    proteins = torch.stack([x[1] for x in batch], dim=0)
    labels = torch.stack([x[2] for x in batch], dim=0)

    if trim_padding:
        smi_nonpad = smiles.ne(SMILES_PAD_INDEX)
        pro_nonpad = proteins.ne(PAD_INDEX)
        if (~smi_nonpad).all(dim=1).any():
            raise ValueError("batch contains empty/all-pad SMILES")
        if (~pro_nonpad).all(dim=1).any():
            raise ValueError("batch contains empty/all-pad protein")
        smiles = smiles[:, : int(smi_nonpad.sum(dim=1).max().item())].contiguous()
        proteins = proteins[:, : int(pro_nonpad.sum(dim=1).max().item())].contiguous()
    return smiles, proteins, labels


def _seed_worker(worker_id: int):
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def create_sequence_loader_from_frame(
    frame: pd.DataFrame,
    *,
    batch_size: int,
    shuffle: bool,
    model_seed: int,
    max_smiles_len: int,
    max_protein_len: int,
    protein_truncation: str = "right",
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    trim_padding: bool = True,
):
    required = {
        "Canonical SMILES", "Target Sequence", "Label_norm"
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"frame missing columns: {sorted(missing)}")
    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    ds = SequenceDTADataset(
        frame["Canonical SMILES"].astype(str).tolist(),
        frame["Target Sequence"].astype(str).tolist(),
        frame["Label_norm"].astype(float).tolist(),
        max_smiles_len=max_smiles_len,
        max_protein_len=max_protein_len,
        protein_truncation=protein_truncation,
    )
    gen = torch.Generator()
    gen.manual_seed(int(model_seed))
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=partial(_collate_sequence, trim_padding=trim_padding),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=gen,
        worker_init_fn=_seed_worker if int(num_workers) > 0 else None,
        persistent_workers=bool(int(num_workers) > 0),
    )


def create_fixed_sequence_train_val_loaders(
    dataset_name: str,
    setting: str,
    *,
    data_root,
    batch_size: int,
    model_seed: int,
    max_smiles_len: int,
    max_protein_len: int,
    protein_truncation: str = "right",
    num_workers: int = 0,
):
    train_df, val_df, scaler = load_fixed_train_val(
        dataset_name, setting, data_root=data_root
    )
    common = dict(
        batch_size=batch_size,
        model_seed=model_seed,
        max_smiles_len=max_smiles_len,
        max_protein_len=max_protein_len,
        protein_truncation=protein_truncation,
        num_workers=num_workers,
        trim_padding=True,
    )
    train_loader = create_sequence_loader_from_frame(
        train_df, shuffle=True, **common
    )
    val_loader = create_sequence_loader_from_frame(
        val_df, shuffle=False, **common
    )
    return train_loader, val_loader, scaler


def create_fixed_sequence_test_loader(
    dataset_name: str,
    setting: str,
    *,
    data_root,
    batch_size: int,
    max_smiles_len: int,
    max_protein_len: int,
    protein_truncation: str = "right",
    num_workers: int = 0,
):
    train_df, val_df, test_df, scaler = load_fixed_split(
        dataset_name, setting, data_root=data_root
    )
    loader = create_sequence_loader_from_frame(
        test_df,
        batch_size=batch_size,
        shuffle=False,
        model_seed=0,
        max_smiles_len=max_smiles_len,
        max_protein_len=max_protein_len,
        protein_truncation=protein_truncation,
        num_workers=num_workers,
        trim_padding=True,
    )
    return train_df, val_df, test_df, scaler, loader
