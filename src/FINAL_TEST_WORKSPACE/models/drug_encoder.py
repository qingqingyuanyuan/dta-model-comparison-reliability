"""Canonical molecular graph encoder for ZhiYao-Graph V3.

The encoder exposes both graph-level and atom-level representations so all
fusion strategies share exactly the same drug encoder.  Token-level outputs are
used by the real cross-attention module; Concat/Product use the graph-level
representation.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    global_add_pool,
    global_mean_pool,
)
from torch_geometric.utils import to_dense_batch


class DrugGNNEncoder(nn.Module):
    """Edge-aware GAT/GCN drug encoder.

    Parameters
    ----------
    pooling:
        ``"add"`` preserves the historical project behavior; ``"mean"`` is
        available for a later sensitivity analysis.  The selected value is
        stored in the checkpoint config.
    """

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        encoder_type: str = "gat",
        gat_heads: int = 4,
        edge_dim: int | None = 10,
        pooling: str = "add",
    ):
        super().__init__()

        if hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if encoder_type not in {"gat", "gcn"}:
            raise ValueError("encoder_type must be 'gat' or 'gcn'")
        if pooling not in {"add", "mean"}:
            raise ValueError("pooling must be 'add' or 'mean'")
        if encoder_type == "gat" and hidden_dim % gat_heads != 0:
            raise ValueError("hidden_dim must be divisible by gat_heads")

        self.node_dim = int(node_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_type = encoder_type
        self.dropout = float(dropout)
        self.edge_dim = edge_dim
        self.pooling = pooling

        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if encoder_type == "gat":
                self.convs.append(
                    GATConv(
                        hidden_dim,
                        hidden_dim // gat_heads,
                        heads=gat_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                    )
                )
            else:
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.output_dim = hidden_dim

        # Compatibility-only caches for the Web visualizer.  They are populated
        # only in eval mode, so training does not retain computation graphs.
        self.node_feat = None
        self.node_mask = None

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "add":
            return global_add_pool(x, batch)
        return global_mean_pool(x, batch)

    def encode(self, data, *, return_tokens: bool = False) -> Dict[str, torch.Tensor]:
        """Return graph-level features and, optionally, padded atom tokens."""
        if data.x.ndim != 2:
            raise ValueError(f"data.x must be [num_nodes, node_dim], got {data.x.shape}")
        if data.x.size(-1) != self.node_dim:
            raise ValueError(
                f"Expected node feature dim {self.node_dim}, got {data.x.size(-1)}"
            )

        x = data.x
        edge_index = data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        edge_attr = getattr(data, "edge_attr", None)
        if self.encoder_type == "gat" and self.edge_dim is not None:
            if edge_attr is None:
                raise ValueError(
                    "GAT is configured with edge_dim but edge_attr is missing"
                )
            if edge_attr.ndim != 2 or edge_attr.size(-1) != self.edge_dim:
                raise ValueError(
                    f"Expected edge_attr [E,{self.edge_dim}], got {edge_attr.shape}"
                )
            if edge_attr.size(0) != edge_index.size(1):
                raise ValueError("edge_attr row count must equal edge_index edge count")

        x = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            if self.encoder_type == "gat":
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        global_feat = self._pool(x, batch)
        result: Dict[str, torch.Tensor] = {"global": global_feat}

        if return_tokens:
            tokens, mask = to_dense_batch(x, batch=batch)
            result["tokens"] = tokens
            result["mask"] = mask
        else:
            tokens = mask = None

        if not self.training:
            # Detach compatibility caches used by legacy Web visualizations.
            self.node_feat = x.detach()
            if return_tokens:
                self.node_mask = mask.detach()

        return result

    def forward(self, data) -> torch.Tensor:
        """Backward-compatible graph-level forward."""
        return self.encode(data, return_tokens=False)["global"]
