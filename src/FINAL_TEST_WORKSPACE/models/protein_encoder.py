"""Canonical protein encoders for ZhiYao-Graph V3.

Key fixes relative to the historical implementation:
- PAD (0) and unknown residue (21) are different tokens;
- CNN padding positions are masked before global pooling;
- token-level protein representations are exposed for true cross-attention;
- pooled representations no longer depend on padded sequence length.
"""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or mask.ndim != 2:
        raise ValueError("tokens must be [B,L,D] and mask must be [B,L]")
    if tokens.shape[:2] != mask.shape:
        raise ValueError("token/mask shapes are incompatible")
    if (~mask).all(dim=1).any():
        raise ValueError("At least one protein sequence contains only padding")

    min_value = torch.finfo(tokens.dtype).min
    masked = tokens.masked_fill(~mask.unsqueeze(-1), min_value)
    return masked.max(dim=1).values


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if (~mask).all(dim=1).any():
        raise ValueError("At least one protein sequence contains only padding")
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ProteinCNNEncoder(nn.Module):
    """Multi-scale 1D-CNN with padding-aware pooling and token outputs."""

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 128,
        kernel_sizes: Sequence[int] | None = None,
        num_filters: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        padding_idx: int = 0,
        post_pool_layers: int = 1,
    ):
        super().__init__()
        kernel_sizes = list(kernel_sizes or [3, 5, 7])
        if not kernel_sizes or any(k <= 0 or k % 2 == 0 for k in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive odd integers")
        if post_pool_layers < 0:
            raise ValueError("post_pool_layers must be >= 0")

        self.padding_idx = int(padding_idx)
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.padding_idx,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    embed_dim,
                    num_filters,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        local_dim = num_filters * len(kernel_sizes)
        self.token_proj = nn.Linear(local_dim, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

        blocks = []
        for _ in range(post_pool_layers):
            blocks.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
        self.global_mlp = nn.Sequential(*blocks) if blocks else nn.Identity()

        self.output_dim = hidden_dim
        self.token_dim = hidden_dim

        # Compatibility-only cache for the Web visualizer; populated in eval.
        self.pos_feat = None
        self.pos_mask = None

    def encode(
        self,
        sequences: torch.Tensor,
        *,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if sequences.ndim != 2:
            raise ValueError(f"sequences must be [B,L], got {sequences.shape}")

        mask = sequences.ne(self.padding_idx)
        if (~mask).all(dim=1).any():
            raise ValueError("Protein sequence cannot be empty/all-padding")

        x = self.embedding(sequences)  # [B,L,E]
        x = x.transpose(1, 2)          # [B,E,L]

        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        local = torch.cat(conv_outs, dim=1).transpose(1, 2)  # [B,L,KF]

        tokens = self.token_proj(local)
        tokens = self.token_norm(tokens)
        tokens = F.relu(tokens)
        tokens = F.dropout(tokens, p=self.dropout, training=self.training)

        # Explicitly zero padded token rows and ignore them in global max.
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        global_feat = _masked_max(tokens, mask)
        global_feat = self.global_mlp(global_feat)

        result: Dict[str, torch.Tensor] = {"global": global_feat}
        if return_tokens:
            result["tokens"] = tokens
            result["mask"] = mask

        if not self.training:
            self.pos_feat = tokens.detach()
            self.pos_mask = mask.detach()

        return result

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.encode(sequences, return_tokens=False)["global"]


class ProteinTransformerEncoder(nn.Module):
    """Optional lightweight Transformer with padding-aware masked pooling."""

    def __init__(
        self,
        vocab_size: int = 22,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 1500,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.max_len = int(max_len)
        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=self.padding_idx,
        )
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim * 2, embed_dim * 2),
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.output_proj = nn.Linear(embed_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim
        self.token_dim = hidden_dim
        self.pos_feat = None
        self.pos_mask = None

    def encode(
        self,
        sequences: torch.Tensor,
        *,
        return_tokens: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if sequences.ndim != 2:
            raise ValueError("sequences must be [B,L]")
        if sequences.size(1) > self.max_len:
            raise ValueError(
                f"Input length {sequences.size(1)} exceeds max_len={self.max_len}"
            )

        mask = sequences.ne(self.padding_idx)
        if (~mask).all(dim=1).any():
            raise ValueError("Protein sequence cannot be empty/all-padding")

        x = self.embedding(sequences)
        x = x + self.pos_encoding[:, : x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=~mask)
        tokens = self.output_norm(self.output_proj(x))
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        global_feat = _masked_mean(tokens, mask)

        result: Dict[str, torch.Tensor] = {"global": global_feat}
        if return_tokens:
            result["tokens"] = tokens
            result["mask"] = mask

        if not self.training:
            self.pos_feat = tokens.detach()
            self.pos_mask = mask.detach()
        return result

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.encode(sequences, return_tokens=False)["global"]
