"""RDKit molecular graph construction for ZhiYao-Graph V3.

The feature dimensions remain 41 (atoms) and 10 (bonds) so the scientific
feature definition stays stable, but unknown categorical values are no longer
silently mislabeled as SP3/single-bond/no-stereo classes.
"""

from __future__ import annotations

from typing import Optional

import torch
from rdkit import Chem
from torch_geometric.data import Data


ATOM_TYPES = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "B", "Si", "Se"]
DEGREE_BINS = list(range(11))
CHARGE_BINS = [-3, -2, -1, 0, 1, 2, 3]
H_BINS = list(range(5))
HYBRID_MAP = {
    Chem.HybridizationType.SP: 0,
    Chem.HybridizationType.SP2: 1,
    Chem.HybridizationType.SP3: 2,
}
BOND_MAP = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
STEREO_MAP = {
    Chem.BondStereo.STEREONONE: 0,
    Chem.BondStereo.STEREOANY: 1,
    Chem.BondStereo.STEREOZ: 2,
    Chem.BondStereo.STEREOE: 3,
}


def _one_hot_known(value, values):
    return [1.0 if value == candidate else 0.0 for candidate in values]


def get_atom_features(atom) -> list[float]:
    features: list[float] = []

    symbol = atom.GetSymbol()
    features.extend(_one_hot_known(symbol, ATOM_TYPES))
    features.append(1.0 if symbol not in ATOM_TYPES else 0.0)

    # Out-of-range values become all-zero within their block rather than being
    # incorrectly forced into one of the known categories.
    features.extend(_one_hot_known(atom.GetDegree(), DEGREE_BINS))
    features.extend(_one_hot_known(atom.GetFormalCharge(), CHARGE_BINS))
    features.extend(_one_hot_known(atom.GetTotalNumHs(), H_BINS))

    hybrid_onehot = [0.0, 0.0, 0.0]
    hybrid_idx = HYBRID_MAP.get(atom.GetHybridization())
    if hybrid_idx is not None:
        hybrid_onehot[hybrid_idx] = 1.0
    features.extend(hybrid_onehot)

    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(float(atom.GetMass()) * 0.01)

    if len(features) != 41:
        raise RuntimeError(f"Atom feature dimension drifted to {len(features)}")
    return features


def get_atom_feature_dim() -> int:
    return 41


def get_bond_features(bond) -> list[float]:
    features: list[float] = []

    bond_onehot = [0.0] * 4
    bond_idx = BOND_MAP.get(bond.GetBondType())
    if bond_idx is not None:
        bond_onehot[bond_idx] = 1.0
    features.extend(bond_onehot)

    stereo_onehot = [0.0] * 4
    stereo_idx = STEREO_MAP.get(bond.GetStereo())
    if stereo_idx is not None:
        stereo_onehot[stereo_idx] = 1.0
    features.extend(stereo_onehot)

    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)

    if len(features) != 10:
        raise RuntimeError(f"Bond feature dimension drifted to {len(features)}")
    return features


def get_bond_feature_dim() -> int:
    return 10


def smiles_to_graph(smiles: str) -> Optional[Data]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    if not atom_features:
        return None

    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edges.extend([[i, j], [j, i]])
        edge_features.extend([bf, bf])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, get_bond_feature_dim()), dtype=torch.float32)

    x = torch.tensor(atom_features, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
