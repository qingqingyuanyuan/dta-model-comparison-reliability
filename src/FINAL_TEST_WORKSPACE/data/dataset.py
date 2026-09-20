"""Dataset and DataLoader utilities for the final ZhiYao-Graph pipeline.

This module assumes that ``scripts/prepare_final_datasets.py`` has already
generated immutable train/val/test CSV files under ``data_final``.

Final-pipeline rules
--------------------
1. No random train/val/test split is created here.
2. Labels come from the prepared ``Label_norm`` column.
3. Molecular graphs are constructed from ``Canonical SMILES``.
4. Invalid SMILES must fail loudly; they are never silently replaced by "C".
5. PyG graphs are collated with ``Batch.from_data_list`` so downstream code can
   safely call ``batch.to(device)`` and use graph-level pooling.
6. The same fixed split files are reused by every model.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Dict, Iterable, Mapping, Optional, Sequence
import random
import warnings

import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from .build_graph import smiles_to_graph
from .utils import (
    DEFAULT_PROTEIN_MAX_LEN,
    PAD_INDEX,
    load_fixed_split,
    load_fixed_train_val,
    protein_seq_to_indices,
)


REQUIRED_SPLIT_COLUMNS = {
    "Canonical SMILES",
    "Target Sequence",
    "Label_norm",
}


def _validate_frame(df: pd.DataFrame, split_name: str) -> None:
    """Validate one already-prepared final split."""
    missing = REQUIRED_SPLIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{split_name} split is missing required columns: {sorted(missing)}"
        )

    required = sorted(REQUIRED_SPLIT_COLUMNS)
    if df[required].isna().any().any():
        raise ValueError(f"{split_name} split contains missing required values")

    # Repeated identical model inputs may be retained within one split under
    # the benchmark-full keep_grouped policy. They must never cross splits;
    # that invariant is checked by data.utils.load_fixed_split().


def _validate_graph(graph: Data, smiles: str) -> None:
    """Validate graph tensors expected by the drug encoder."""
    if graph is None:
        raise ValueError(f"Failed to build molecular graph for SMILES: {smiles}")

    if not hasattr(graph, "x") or graph.x is None or graph.x.ndim != 2:
        raise ValueError(f"Invalid node-feature tensor for SMILES: {smiles}")

    if not hasattr(graph, "edge_index") or graph.edge_index is None:
        raise ValueError(f"Missing edge_index for SMILES: {smiles}")

    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(
            f"edge_index must have shape [2, E] for SMILES: {smiles}; "
            f"got {tuple(graph.edge_index.shape)}"
        )

    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        raise ValueError(f"Missing edge_attr for SMILES: {smiles}")

    if graph.edge_attr.ndim != 2:
        raise ValueError(
            f"edge_attr must have shape [E, edge_dim] for SMILES: {smiles}; "
            f"got {tuple(graph.edge_attr.shape)}"
        )

    if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
        raise ValueError(
            f"Edge count mismatch for SMILES {smiles}: "
            f"edge_index has {graph.edge_index.shape[1]} edges, "
            f"edge_attr has {graph.edge_attr.shape[0]} rows"
        )


def build_graph_cache(
    smiles_list: Iterable[str],
    *,
    show_progress: bool = True,
) -> Dict[str, Data]:
    """Build each unique canonical-SMILES graph exactly once.

    This is deterministic, label-free preprocessing. Sharing this cache across
    fixed train/val/test files does not fit or learn anything from labels.
    """
    unique_smiles = list(OrderedDict.fromkeys(str(s) for s in smiles_list))
    iterator = tqdm(
        unique_smiles,
        desc="Building unique molecular graphs",
        disable=not show_progress,
    )

    graph_cache: Dict[str, Data] = {}
    failed = []

    for smi in iterator:
        graph = smiles_to_graph(smi)
        if graph is None:
            failed.append(smi)
            continue
        _validate_graph(graph, smi)
        graph_cache[smi] = graph

    if failed:
        examples = ", ".join(repr(s) for s in failed[:5])
        raise ValueError(
            f"Graph construction failed for {len(failed)} canonical SMILES. "
            f"Examples: {examples}. "
            "The final data-preparation step should have removed invalid SMILES; "
            "do not silently replace them with a fallback molecule."
        )

    return graph_cache


class DTADataset(Dataset):
    """Drug-target affinity dataset using fixed, prepared inputs."""

    def __init__(
        self,
        drug_smiles: Sequence[str],
        target_seqs: Sequence[str],
        affinities: Sequence[float],
        *,
        graph_cache: Mapping[str, Data],
        max_len: int = DEFAULT_PROTEIN_MAX_LEN,
        truncation: str = "right",
    ):
        if not (len(drug_smiles) == len(target_seqs) == len(affinities)):
            raise ValueError(
                "drug_smiles, target_seqs and affinities must have equal length"
            )

        if max_len <= 0:
            raise ValueError("max_len must be positive")

        self.drug_smiles = [str(s) for s in drug_smiles]
        self.target_seqs = [str(s) for s in target_seqs]
        self.affinities = [float(y) for y in affinities]
        self.graph_cache = graph_cache
        self.max_len = int(max_len)
        self.truncation = str(truncation)

        # Encode each unique protein sequence exactly once.  KIBA/Davis contain
        # only hundreds of unique proteins but tens of thousands of rows.
        self.protein_cache = {
            seq: torch.tensor(
                protein_seq_to_indices(seq, self.max_len, self.truncation),
                dtype=torch.long,
            )
            for seq in dict.fromkeys(self.target_seqs)
        }

        missing_graphs = sorted(
            {s for s in self.drug_smiles if s not in self.graph_cache}
        )
        if missing_graphs:
            examples = ", ".join(repr(s) for s in missing_graphs[:5])
            raise ValueError(
                f"Graph cache is missing {len(missing_graphs)} molecules. "
                f"Examples: {examples}"
            )

    def __len__(self) -> int:
        return len(self.affinities)

    def __getitem__(self, idx: int):
        drug_graph = self.graph_cache[self.drug_smiles[idx]]

        seq_indices = self.protein_cache[self.target_seqs[idx]]

        affinity = torch.tensor(
            self.affinities[idx],
            dtype=torch.float32,
        )

        return drug_graph, seq_indices, affinity


# Backward-compatible name for old project imports.
DTIDataset = DTADataset


def collate_fn(batch, *, trim_protein_padding: bool = True):
    """Collate graphs and optionally trim trailing all-PAD protein columns.

    Every sequence is encoded with the predeclared ``max_len`` before this
    point. Trimming removes only columns that are PAD for every sample in the
    current batch; it changes neither biological residues nor the predefined
    truncation rule. This substantially reduces CNN/cross-attention compute for
    short proteins while remaining prediction-equivalent to fixed right padding.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    drug_graphs = [item[0] for item in batch]
    target_seqs = torch.stack([item[1] for item in batch], dim=0)
    affinities = torch.stack([item[2] for item in batch], dim=0)

    if trim_protein_padding:
        nonpad = target_seqs.ne(PAD_INDEX)
        lengths = nonpad.sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("Protein batch contains an all-padding sequence")
        batch_max_len = int(lengths.max().item())
        target_seqs = target_seqs[:, :batch_max_len].contiguous()

    drug_batch = Batch.from_data_list(drug_graphs)
    return drug_batch, target_seqs, affinities


def _seed_worker(worker_id: int) -> None:
    """Deterministically seed Python/NumPy from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loader_from_frame(
    df: pd.DataFrame,
    *,
    split_name: str,
    graph_cache: Mapping[str, Data],
    batch_size: int = 64,
    shuffle: bool = False,
    model_seed: int = 42,
    max_len: int = DEFAULT_PROTEIN_MAX_LEN,
    truncation: str = "right",
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    trim_protein_padding: bool = True,
) -> DataLoader:
    """Create one DataLoader from an immutable prepared split DataFrame."""
    _validate_frame(df, split_name)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available())

    dataset = DTADataset(
        df["Canonical SMILES"].astype(str).tolist(),
        df["Target Sequence"].astype(str).tolist(),
        df["Label_norm"].astype(float).tolist(),
        graph_cache=graph_cache,
        max_len=max_len,
        truncation=truncation,
    )

    # Use an explicit generator for every loader so worker seeds are
    # reproducible even when shuffle=False.
    generator = torch.Generator()
    generator.manual_seed(int(model_seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=partial(
            collate_fn,
            trim_protein_padding=bool(trim_protein_padding),
        ),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
        generator=generator,
        worker_init_fn=_seed_worker if int(num_workers) > 0 else None,
        persistent_workers=bool(int(num_workers) > 0),
    )



def create_fixed_train_val_loaders(
    dataset_name: str,
    setting: str = "warm",
    *,
    data_root=None,
    batch_size: int = 64,
    model_seed: int = 42,
    max_len: int = DEFAULT_PROTEIN_MAX_LEN,
    truncation: str = "right",
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    show_progress: bool = True,
    trim_protein_padding: bool = True,
):
    """Create only train/validation loaders for the sealed-test training phase."""
    train_df, val_df, scaler = load_fixed_train_val(
        dataset_name, setting, data_root=data_root
    )
    all_smiles = pd.concat(
        [
            train_df["Canonical SMILES"],
            val_df["Canonical SMILES"],
        ],
        ignore_index=True,
    ).astype(str)
    graph_cache = build_graph_cache(
        all_smiles.tolist(), show_progress=show_progress
    )
    common = dict(
        graph_cache=graph_cache,
        batch_size=batch_size,
        model_seed=model_seed,
        max_len=max_len,
        truncation=truncation,
        num_workers=num_workers,
        pin_memory=pin_memory,
        trim_protein_padding=trim_protein_padding,
    )
    train_loader = create_loader_from_frame(
        train_df,
        split_name=f"{dataset_name}/{setting}/train",
        shuffle=True,
        **common,
    )
    val_loader = create_loader_from_frame(
        val_df,
        split_name=f"{dataset_name}/{setting}/val",
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, scaler

def create_fixed_dataloaders(
    dataset_name: str,
    setting: str = "warm",
    *,
    data_root=None,
    batch_size: int = 64,
    model_seed: int = 42,
    max_len: int = DEFAULT_PROTEIN_MAX_LEN,
    truncation: str = "right",
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    show_progress: bool = True,
    trim_protein_padding: bool = True,
):
    """Load fixed final splits and create train/val/test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, scaler

    Notes
    -----
    No sampling, re-splitting or scaler fitting is performed here.
    """
    train_df, val_df, test_df, scaler = load_fixed_split(
        dataset_name,
        setting,
        data_root=data_root,
    )

    all_smiles = pd.concat(
        [
            train_df["Canonical SMILES"],
            val_df["Canonical SMILES"],
            test_df["Canonical SMILES"],
        ],
        ignore_index=True,
    ).astype(str)

    graph_cache = build_graph_cache(
        all_smiles.tolist(),
        show_progress=show_progress,
    )

    train_loader = create_loader_from_frame(
        train_df,
        split_name=f"{dataset_name}/{setting}/train",
        graph_cache=graph_cache,
        batch_size=batch_size,
        shuffle=True,
        model_seed=model_seed,
        max_len=max_len,
        truncation=truncation,
        num_workers=num_workers,
        pin_memory=pin_memory,
        trim_protein_padding=trim_protein_padding,
    )
    val_loader = create_loader_from_frame(
        val_df,
        split_name=f"{dataset_name}/{setting}/val",
        graph_cache=graph_cache,
        batch_size=batch_size,
        shuffle=False,
        model_seed=model_seed,
        max_len=max_len,
        truncation=truncation,
        num_workers=num_workers,
        pin_memory=pin_memory,
        trim_protein_padding=trim_protein_padding,
    )
    test_loader = create_loader_from_frame(
        test_df,
        split_name=f"{dataset_name}/{setting}/test",
        graph_cache=graph_cache,
        batch_size=batch_size,
        shuffle=False,
        model_seed=model_seed,
        max_len=max_len,
        truncation=truncation,
        num_workers=num_workers,
        pin_memory=pin_memory,
        trim_protein_padding=trim_protein_padding,
    )

    return train_loader, val_loader, test_loader, scaler


def create_dataloaders(*args, **kwargs):
    """Disabled legacy entry point that used to create a new random split."""
    raise RuntimeError(
        "create_dataloaders(drug_smiles, target_seqs, affinities, ...) is "
        "disabled in the final pipeline because it creates a new random split. "
        "Use create_fixed_dataloaders(dataset_name, setting, ...) instead."
    )


def _build_graphs(smiles_list):
    """Legacy helper retained only for compatibility; no fallback molecules."""
    warnings.warn(
        "_build_graphs() is a legacy helper. Prefer build_graph_cache().",
        DeprecationWarning,
        stacklevel=2,
    )
    cache = build_graph_cache(smiles_list)
    return [cache[str(s)] for s in smiles_list]
