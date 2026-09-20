"""Neural building blocks shared by controlled baseline models."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_max(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or mask.ndim != 2:
        raise ValueError("tokens=[B,L,D], mask=[B,L] required")
    if tokens.shape[:2] != mask.shape:
        raise ValueError("token/mask shape mismatch")
    if (~mask).all(dim=1).any():
        raise ValueError("all-padding sequence")
    floor = torch.finfo(tokens.dtype).min
    return tokens.masked_fill(~mask.unsqueeze(-1), floor).max(dim=1).values


class CharacterCNNEncoder(nn.Module):
    """Padding-aware character CNN used by the controlled DeepDTA-style model."""

    def __init__(
        self,
        *,
        vocab_size: int,
        padding_idx: int,
        embed_dim: int,
        conv_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        conv_channels = list(conv_channels)
        kernel_sizes = list(kernel_sizes)
        if len(conv_channels) != len(kernel_sizes) or not conv_channels:
            raise ValueError("conv_channels/kernel_sizes must be equal non-empty lists")
        if any(int(k) <= 0 or int(k) % 2 == 0 for k in kernel_sizes):
            raise ValueError("controlled baseline kernels must be positive odd integers")

        self.padding_idx = int(padding_idx)
        self.embedding = nn.Embedding(
            int(vocab_size), int(embed_dim), padding_idx=self.padding_idx
        )
        layers = []
        in_ch = int(embed_dim)
        for out_ch, kernel in zip(conv_channels, kernel_sizes):
            layers.append(
                nn.Conv1d(
                    in_ch,
                    int(out_ch),
                    kernel_size=int(kernel),
                    padding=int(kernel) // 2,
                )
            )
            in_ch = int(out_ch)
        self.convs = nn.ModuleList(layers)
        self.proj = nn.Linear(in_ch, int(output_dim))
        self.norm = nn.LayerNorm(int(output_dim))
        self.dropout = float(dropout)
        self.output_dim = int(output_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2:
            raise ValueError("sequence indices must be [B,L]")
        mask = indices.ne(self.padding_idx)
        if (~mask).all(dim=1).any():
            raise ValueError("empty/all-pad sequence")
        x = self.embedding(indices).transpose(1, 2)
        for conv in self.convs:
            x = F.relu(conv(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        tokens = x.transpose(1, 2)
        tokens = tokens * mask.unsqueeze(-1).to(tokens.dtype)
        pooled = masked_max(tokens, mask)
        pooled = self.norm(self.proj(pooled))
        return F.relu(pooled)


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, dropout: float):
        super().__init__()
        layers = []
        d = int(input_dim)
        for h in hidden_dims:
            h = int(h)
            layers.extend(
                [
                    nn.Linear(d, h),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
