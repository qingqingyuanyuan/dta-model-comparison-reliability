#!/usr/bin/env python3
"""Fast integrity checks for controlled baseline implementations.

Run in the same environment as the main project (PyTorch + PyG + RDKit):
    python scripts/baseline_smoke_test.py

These tests verify interfaces/gradients and strict checkpoint reconstruction,
not scientific performance or exact reproduction of external repositories.
"""

from __future__ import annotations

import copy
import tempfile
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from baselines.dataset import (
    SMILES_PAD_INDEX,
    SMILES_UNK_INDEX,
    SMILES_VOCAB_SIZE,
    smiles_to_indices,
)
from baselines.registry import (
    BASELINE_NAMES,
    build_baseline_model,
    get_baseline_config,
    load_baseline_checkpoint,
)
from data.build_graph import smiles_to_graph
from data.utils import protein_seq_to_indices
from torch_geometric.data import Batch


def test_smiles_vocab():
    idx = smiles_to_indices("CC(=O)O", 12)
    assert len(idx) == 12
    assert idx[-1] == SMILES_PAD_INDEX
    assert all(0 <= x < SMILES_VOCAB_SIZE for x in idx)
    assert SMILES_UNK_INDEX != SMILES_PAD_INDEX
    print("[PASS] fixed printable-ASCII SMILES vocabulary")


def test_deepdta_style():
    config = get_baseline_config("deepdta_style")
    config["BASELINE"]["max_smiles_len"] = 32
    config["BASELINE"]["max_protein_len"] = 40
    model = build_baseline_model("deepdta_style", config)
    smi = torch.tensor(
        [
            smiles_to_indices("CCO", 32),
            smiles_to_indices("c1ccccc1", 32),
        ],
        dtype=torch.long,
    )
    seq = torch.tensor(
        [
            protein_seq_to_indices("ACDEFGHIK", 40),
            protein_seq_to_indices("MKTLLVAGAA", 40),
        ],
        dtype=torch.long,
    )
    y = torch.tensor([0.2, 0.7])
    pred = model(smi, seq)
    assert pred.shape == (2,)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    print("[PASS] DeepDTA-style forward/backward")


def test_graphdta_style():
    config = get_baseline_config("graphdta_gcn_style")
    config["BASELINE"]["max_protein_len"] = 40
    model = build_baseline_model("graphdta_gcn_style", config)
    graphs = [smiles_to_graph("CCO"), smiles_to_graph("c1ccccc1")]
    batch = Batch.from_data_list(graphs)
    seq = torch.tensor(
        [
            protein_seq_to_indices("ACDEFGHIK", 40),
            protein_seq_to_indices("MKTLLVAGAA", 40),
        ],
        dtype=torch.long,
    )
    y = torch.tensor([0.2, 0.7])
    pred = model(batch, seq)
    assert pred.shape == (2,)
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    print("[PASS] GraphDTA-GCN-style forward/backward")


def test_strict_checkpoint_reload():
    # DeepDTA-style is sufficient to validate the registry/checkpoint protocol;
    # GraphDTA strict loading uses the same registry path.
    name = "deepdta_style"
    config = get_baseline_config(name)
    model = build_baseline_model(name, config)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "baseline.pt"
        torch.save(
            {
                "architecture_version": config["BASELINE"]["architecture_version"],
                "baseline_name": name,
                "model_state_dict": model.state_dict(),
                "config": config,
                "dataset": "KIBA",
                "setting": "warm",
                "model_seed": 42,
                "test_evaluated_during_training": False,
            },
            path,
        )
        loaded, ckpt = load_baseline_checkpoint(path)
        assert loaded.architecture_version == model.architecture_version
        for k, v in model.state_dict().items():
            if not torch.equal(v, loaded.state_dict()[k]):
                raise AssertionError(f"strict reload mismatch: {k}")
    print("[PASS] strict controlled-baseline checkpoint reload")


def main():
    print("Baselines:", BASELINE_NAMES)
    test_smiles_vocab()
    test_deepdta_style()
    test_graphdta_style()
    test_strict_checkpoint_reload()
    print("\nALL BASELINE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
