"""Public data API for the fixed-split ZhiYao-Graph pipeline.

Lightweight utilities are imported eagerly. PyTorch-Geometric-dependent graph
and DataLoader objects are imported lazily so planning, audit, scaler checks and
split validation can run on a host that does not yet have PyG installed.
"""

from .utils import (
    AA_TO_INDEX,
    DEFAULT_PROTEIN_MAX_LEN,
    PAD_INDEX,
    PROTEIN_VOCAB_SIZE,
    STANDARD_AMINO_ACIDS,
    UNK_INDEX,
    apply_affinity_scaler,
    create_dataset_from_csv,
    denormalize_affinity,
    load_fixed_split,
    load_fixed_train_val,
    load_raw_data,
    load_scaler,
    normalize_affinity,
    protein_seq_to_indices,
    split_data,
)

_GRAPH_EXPORTS = {
    "get_atom_feature_dim",
    "get_bond_feature_dim",
    "smiles_to_graph",
}

_DATASET_EXPORTS = {
    "DTADataset",
    "DTIDataset",
    "build_graph_cache",
    "collate_fn",
    "create_dataloaders",
    "create_fixed_dataloaders",
    "create_fixed_train_val_loaders",
    "create_loader_from_frame",
}


def __getattr__(name):
    if name in _GRAPH_EXPORTS:
        from . import build_graph as _build_graph
        return getattr(_build_graph, name)
    if name in _DATASET_EXPORTS:
        from . import dataset as _dataset
        return getattr(_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "STANDARD_AMINO_ACIDS",
    "AA_TO_INDEX",
    "PAD_INDEX",
    "UNK_INDEX",
    "PROTEIN_VOCAB_SIZE",
    "DEFAULT_PROTEIN_MAX_LEN",
    "DTADataset",
    "DTIDataset",
    "create_fixed_dataloaders",
    "create_fixed_train_val_loaders",
    "create_loader_from_frame",
    "load_fixed_split",
    "load_fixed_train_val",
    "load_scaler",
    "build_graph_cache",
    "collate_fn",
    "smiles_to_graph",
    "get_atom_feature_dim",
    "get_bond_feature_dim",
    "protein_seq_to_indices",
    "apply_affinity_scaler",
    "denormalize_affinity",
    "create_dataset_from_csv",
    # Legacy/provenance entry points remain exported but are blocked/deprecated.
    "create_dataloaders",
    "load_raw_data",
    "split_data",
    "normalize_affinity",
]
