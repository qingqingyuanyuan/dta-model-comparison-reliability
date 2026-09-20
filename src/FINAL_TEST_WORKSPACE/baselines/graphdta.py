"""GraphDTA-GCN-style controlled baseline.

The model uses a GCN molecular graph encoder, a protein sequence CNN,
concatenation and an MLP regression head on the exact fixed project splits.
It is intentionally labeled "GraphDTA-GCN-style" rather than an exact external
repository reproduction.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool

from data.utils import PAD_INDEX
from .common import CharacterCNNEncoder, RegressionHead


class GraphDTAStyleGCN(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = copy.deepcopy(config)
        bc = self.config["BASELINE"]
        self.architecture_version = str(bc["architecture_version"])
        self.baseline_name = "graphdta_gcn_style"
        self.input_modality = str(bc["input_modality"])
        self.dropout = float(bc["dropout"])

        hidden = int(bc["gcn_hidden_dim"])
        layers = int(bc["gcn_layers"])
        if layers <= 0:
            raise ValueError("gcn_layers must be positive")
        self.input_proj = nn.Linear(int(bc["node_dim"]), hidden)
        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden, hidden) for _ in range(layers)]
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
        self.drug_output = nn.Linear(hidden, int(bc["encoder_output_dim"]))
        self.predictor = RegressionHead(
            input_dim=2 * int(bc["encoder_output_dim"]),
            hidden_dims=bc["predictor_hidden_dims"],
            dropout=bc["dropout"],
        )

    def encode_drug(self, graph_batch):
        x = self.input_proj(graph_batch.x)
        for conv in self.gcn_layers:
            x = F.relu(conv(x, graph_batch.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        batch = getattr(graph_batch, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_max_pool(x, batch)
        return F.relu(self.drug_output(x))

    def forward(self, graph_batch, protein_indices):
        d = self.encode_drug(graph_batch)
        p = self.protein_encoder(protein_indices)
        return self.predictor(torch.cat([d, p], dim=-1))

    def parameter_summary(self):
        drug_params = (
            sum(p.numel() for p in self.input_proj.parameters() if p.requires_grad)
            + sum(
                p.numel()
                for layer in self.gcn_layers
                for p in layer.parameters()
                if p.requires_grad
            )
            + sum(p.numel() for p in self.drug_output.parameters() if p.requires_grad)
        )
        out = {
            "drug_encoder": int(drug_params),
            "protein_encoder": sum(
                p.numel() for p in self.protein_encoder.parameters() if p.requires_grad
            ),
            "predictor": sum(
                p.numel() for p in self.predictor.parameters() if p.requires_grad
            ),
        }
        out["total"] = sum(out.values())
        return out
