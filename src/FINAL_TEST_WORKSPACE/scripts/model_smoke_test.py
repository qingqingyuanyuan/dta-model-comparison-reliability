#!/usr/bin/env python3
"""Fast integrity tests for the canonical ZhiYao-Graph V3 model.

Run this before any expensive training:
    python scripts/model_smoke_test.py

The test intentionally uses tiny synthetic inputs.  It verifies interfaces and
critical invariants, not scientific performance.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from data.build_graph import get_atom_feature_dim, get_bond_feature_dim, smiles_to_graph
from data.utils import (
    PAD_INDEX,
    PROTEIN_VOCAB_SIZE,
    UNK_INDEX,
    protein_seq_to_indices,
)
from models.fusion import AttentionFusion, ConcatFusion, ProductFusion
from models.predictor import ARCHITECTURE_VERSION, DTAPredictor
from models.protein_encoder import ProteinCNNEncoder
from torch_geometric.data import Batch


def assert_close(a, b, tol=1e-5):
    if not torch.allclose(a, b, atol=tol, rtol=tol):
        raise AssertionError(f"Tensors differ beyond tolerance {tol}")


def test_vocab():
    idx = protein_seq_to_indices("ACDX?", max_len=8)
    assert idx[0] != PAD_INDEX
    assert idx[3] == UNK_INDEX  # X
    assert idx[4] == UNK_INDEX  # other unknown
    assert idx[5:] == [PAD_INDEX, PAD_INDEX, PAD_INDEX]
    assert PROTEIN_VOCAB_SIZE == 22
    print("[PASS] PAD/UNK protein vocabulary")


def test_graph_features():
    g = smiles_to_graph("CCO")
    assert g is not None
    assert g.x.shape[1] == 41 == get_atom_feature_dim()
    assert g.edge_attr.shape[1] == 10 == get_bond_feature_dim()
    assert g.edge_index.shape[1] == g.edge_attr.shape[0]
    print("[PASS] 41D atom / 10D bond graph features")


def test_padding_aware_cnn():
    enc = ProteinCNNEncoder(
        vocab_size=22,
        embed_dim=32,
        num_filters=8,
        hidden_dim=16,
        dropout=0.0,
        post_pool_layers=1,
    ).eval()

    # Same biological sequence with different amounts of trailing padding.
    seq_a = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]], dtype=torch.long)
    seq_b = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.long)
    out_a = enc(seq_a)
    out_b = enc(seq_b)
    assert out_a.shape == (1, 16)
    assert out_b.shape == (1, 16)
    assert_close(out_a, out_b, tol=1e-5)

    aux = enc.encode(seq_b, return_tokens=True)
    assert aux["tokens"].shape == (1, 12, 16)
    assert aux["mask"].sum().item() == 5
    assert torch.all(aux["tokens"][0, 5:] == 0)
    print("[PASS] padding-aware protein CNN pooling/token mask")


def test_fusion_parameter_match():
    concat = ConcatFusion(128, 128, 256, dropout=0.0)
    product = ProductFusion(
        128, 128, 256, align_dim=cfg.FUSION.get("align_dim", 128), dropout=0.0
    )
    attention = AttentionFusion(
        128,
        128,
        256,
        align_dim=cfg.FUSION.get("attention_align_dim", 80),
        num_heads=cfg.FUSION.get("num_heads", 4),
        dropout=0.0,
    )
    counts = {
        "concat": sum(p.numel() for p in concat.parameters()),
        "product": sum(p.numel() for p in product.parameters()),
        "attention": sum(p.numel() for p in attention.parameters()),
    }
    relative_span = (max(counts.values()) - min(counts.values())) / max(counts.values())
    if relative_span > 0.05:
        raise AssertionError(f"Fusion parameter span too large: {counts}")
    print(f"[PASS] fusion parameter control (<5% span): {counts}")


def test_true_cross_attention():
    torch.manual_seed(7)
    fusion = AttentionFusion(
        drug_dim=128,
        protein_dim=128,
        hidden_dim=256,
        align_dim=cfg.FUSION.get("attention_align_dim", 80),
        num_heads=cfg.FUSION.get("num_heads", 4),
        dropout=0.0,
    ).eval()

    b, n_atom, length = 2, 4, 7
    d_global = torch.randn(b, 128)
    p_global = torch.randn(b, 128)
    d_tokens = torch.randn(b, n_atom, 128)
    p_tokens = torch.randn(b, length, 128)
    d_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool
    )
    p_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool
    )

    out, attn = fusion(
        d_global,
        p_global,
        drug_tokens=d_tokens,
        drug_mask=d_mask,
        protein_tokens=p_tokens,
        protein_mask=p_mask,
        return_attention=True,
    )
    assert out.shape == (b, 256)
    assert attn.shape == (b, 4, n_atom, length)
    if attn.shape[-1] <= 1:
        raise AssertionError("Cross-attention key length is still 1")

    # Masked protein positions must receive zero attention.
    assert torch.allclose(attn[0, :, :, 5:], torch.zeros_like(attn[0, :, :, 5:]))
    assert torch.allclose(attn[1, :, :, 6:], torch.zeros_like(attn[1, :, :, 6:]))
    assert_close(attn[0, :, :, :5].sum(dim=-1), torch.ones_like(attn[0, :, :, 0]))
    assert_close(attn[1, :, :, :6].sum(dim=-1), torch.ones_like(attn[1, :, :, 0]))

    # Changing protein tokens must change the fused output for the same drug.
    p_tokens_2 = p_tokens.clone()
    p_tokens_2[:, :3] += 2.5
    out2 = fusion(
        d_global,
        p_global,
        drug_tokens=d_tokens,
        drug_mask=d_mask,
        protein_tokens=p_tokens_2,
        protein_mask=p_mask,
    )
    if torch.allclose(out, out2):
        raise AssertionError("Attention output did not respond to protein-token changes")
    print("[PASS] real token-level atom -> protein cross-attention")


def _small_config(fusion):
    c = {
        "DRUG_ENCODER": copy.deepcopy(cfg.DRUG_ENCODER),
        "PROTEIN_ENCODER": copy.deepcopy(cfg.PROTEIN_ENCODER),
        "FUSION": copy.deepcopy(cfg.FUSION),
        "PREDICTOR": copy.deepcopy(cfg.PREDICTOR),
        "TRAINING": copy.deepcopy(cfg.TRAINING),
    }
    c["FUSION"]["fusion_mode"] = fusion
    # Keep the exact feature dimensions but reduce only test sequence length.
    c["PROTEIN_ENCODER"]["max_len"] = 32
    return c


def test_end_to_end():
    graphs = [smiles_to_graph("CCO"), smiles_to_graph("c1ccccc1")]
    batch = Batch.from_data_list(graphs)
    seqs = torch.tensor(
        [
            protein_seq_to_indices("ACDEFGHIK", max_len=32),
            protein_seq_to_indices("MKTLLVAGAA", max_len=32),
        ],
        dtype=torch.long,
    )
    labels = torch.tensor([0.2, 0.7], dtype=torch.float32)

    for fusion in ("concat", "product", "attention"):
        model = DTAPredictor(_small_config(fusion))
        assert model.architecture_version == ARCHITECTURE_VERSION
        model.train()
        pred = model(batch, seqs)
        assert pred.shape == (2,)
        assert torch.isfinite(pred).all()
        loss = torch.nn.functional.mse_loss(pred, labels)
        loss.backward()
        # Protein encoder must actually receive gradient in every fusion.
        protein_grads = {
            name: p.grad
            for name, p in model.protein_encoder.named_parameters()
            if p.requires_grad
        }
        missing_grad = [
            name
            for name, grad in protein_grads.items()
            if grad is None or not torch.isfinite(grad).all()
        ]
        if missing_grad:
            raise AssertionError(
                f"Protein encoder parameters without finite gradients for {fusion}: "
                f"{missing_grad}"
            )
        if not any(grad.abs().sum() > 0 for grad in protein_grads.values()):
            raise AssertionError(f"Protein encoder received no nonzero gradient for {fusion}")

        model.eval()
        if fusion == "attention":
            aux = model(batch, seqs, return_aux=True)
            attn = aux["attention"]
            assert attn is not None and attn.shape[-1] == 32
        print(f"[PASS] end-to-end forward/backward: {fusion}")



def test_prespecified_sensitivity_configs():
    graphs = [smiles_to_graph("CCO"), smiles_to_graph("c1ccccc1")]
    batch = Batch.from_data_list(graphs)
    seqs = torch.tensor(
        [
            protein_seq_to_indices("ACDEFGHIK", max_len=32),
            protein_seq_to_indices("MKTLLVAGAA", max_len=32),
        ],
        dtype=torch.long,
    )
    for name, spec in cfg.SENSITIVITY_EXPERIMENTS.items():
        c = _small_config("concat")
        # Keep smoke inputs short while verifying the configured factor itself.
        if name == "protein_len_1000":
            c["PROTEIN_ENCODER"]["max_len"] = 32
        c["DRUG_ENCODER"]["pooling"] = spec["overrides"]["drug_pooling"]
        model = DTAPredictor(c).eval()
        pred = model(batch, seqs)
        assert pred.shape == (2,) and torch.isfinite(pred).all()
    print("[PASS] prespecified sensitivity model configurations build/forward")

def main():
    print("Architecture:", ARCHITECTURE_VERSION)
    test_vocab()
    test_graph_features()
    test_padding_aware_cnn()
    test_fusion_parameter_match()
    test_true_cross_attention()
    test_end_to_end()
    test_prespecified_sensitivity_configs()
    print("\nALL MODEL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
