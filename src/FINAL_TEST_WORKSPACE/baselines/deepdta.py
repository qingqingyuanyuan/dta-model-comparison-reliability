"""DeepDTA-style controlled baseline.

This model keeps the defining high-level idea of DeepDTA—character-level drug
sequence CNN + protein sequence CNN followed by concatenation—but uses a modern,
padding-aware implementation and the project's fixed split/scaler protocol.
It must therefore be reported as a *DeepDTA-style controlled reimplementation*,
not as an exact reproduction of a published repository result.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from data.utils import PAD_INDEX
from .common import CharacterCNNEncoder, RegressionHead
from .dataset import SMILES_PAD_INDEX, SMILES_VOCAB_SIZE


class DeepDTAStyle(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = copy.deepcopy(config)
        bc = self.config["BASELINE"]
        self.architecture_version = str(bc["architecture_version"])
        self.baseline_name = "deepdta_style"
        self.input_modality = str(bc["input_modality"])

        self.drug_encoder = CharacterCNNEncoder(
            vocab_size=SMILES_VOCAB_SIZE,
            padding_idx=SMILES_PAD_INDEX,
            embed_dim=bc["smiles_embed_dim"],
            conv_channels=bc["conv_channels"],
            kernel_sizes=bc["kernel_sizes"],
            output_dim=bc["encoder_output_dim"],
            dropout=bc["dropout"],
        )
        self.protein_encoder = CharacterCNNEncoder(
            vocab_size=bc["protein_vocab_size"],
            padding_idx=PAD_INDEX,
            embed_dim=bc["protein_embed_dim"],
            conv_channels=bc["conv_channels"],
            kernel_sizes=bc["kernel_sizes"],
            output_dim=bc["encoder_output_dim"],
            dropout=bc["dropout"],
        )
        self.predictor = RegressionHead(
            input_dim=2 * int(bc["encoder_output_dim"]),
            hidden_dims=bc["predictor_hidden_dims"],
            dropout=bc["dropout"],
        )

    def forward(self, smiles_indices: torch.Tensor, protein_indices: torch.Tensor):
        d = self.drug_encoder(smiles_indices)
        p = self.protein_encoder(protein_indices)
        return self.predictor(torch.cat([d, p], dim=-1))

    def parameter_summary(self):
        modules = {
            "drug_encoder": self.drug_encoder,
            "protein_encoder": self.protein_encoder,
            "predictor": self.predictor,
        }
        out = {
            name: sum(p.numel() for p in m.parameters() if p.requires_grad)
            for name, m in modules.items()
        }
        out["total"] = sum(out.values())
        return out
