"""Cross-modal fusion modules for ZhiYao-Graph V3.

ConcatFusion, ProductFusion and AttentionFusion are parameter-controlled so
the fusion comparison does not simply compare a tiny baseline with a much
larger attention block. AttentionFusion is a *real token-level cross-attention*:
drug atom tokens are queries and protein residue-position tokens are keys/values.
It refuses to run without token features, preventing regression to the
historical length-1 attention bug.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Parameter-controlled concatenation baseline."""

    requires_token_features = False

    def __init__(
        self,
        drug_dim: int = 128,
        protein_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        self.proj = nn.Linear(drug_dim + protein_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, drug_vec, protein_vec, **kwargs):
        x = torch.cat([drug_vec, protein_vec], dim=-1)
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        return self.norm(x)


class ProductFusion(nn.Module):
    """Element-wise interaction with parameter count matched to ConcatFusion."""

    requires_token_features = False

    def __init__(
        self,
        drug_dim: int = 128,
        protein_dim: int = 128,
        hidden_dim: int = 256,
        align_dim: int = 128,
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, align_dim)
        self.protein_proj = nn.Linear(protein_dim, align_dim)
        self.output_proj = nn.Linear(align_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, drug_vec, protein_vec, **kwargs):
        d = self.drug_proj(drug_vec)
        p = self.protein_proj(protein_vec)
        x = d * p
        x = F.relu(self.output_proj(x))
        x = self.dropout(x)
        return self.norm(x)


class AttentionFusion(nn.Module):
    """True atom-to-protein-position multi-head cross-attention.

    Query  : drug atom tokens ``[B, N_atom, D]``
    Key/Val: protein position tokens ``[B, L, P]``

    The historical implementation first pooled each modality to one vector and
    then added a length-1 dimension.  Softmax therefore operated over one key
    and was always 1.  This implementation requires sequence/token tensors and
    masks, so that failure mode cannot occur silently.
    """

    requires_token_features = True

    def __init__(
        self,
        drug_dim: int = 128,
        protein_dim: int = 128,
        hidden_dim: int = 256,
        align_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.2,
        **_,
    ):
        super().__init__()
        if align_dim % num_heads != 0:
            raise ValueError("align_dim must be divisible by num_heads")

        self.drug_token_proj = (
            nn.Identity() if drug_dim == align_dim else nn.Linear(drug_dim, align_dim)
        )
        self.protein_token_proj = (
            nn.Identity()
            if protein_dim == align_dim
            else nn.Linear(protein_dim, align_dim)
        )
        self.query_norm = nn.LayerNorm(align_dim)
        self.key_value_norm = nn.LayerNorm(align_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=align_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(align_dim)
        self.output_proj = nn.Linear(align_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.output_dim = hidden_dim
        self.num_heads = num_heads
        self.last_attn = None

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(tokens.dtype)
        denom = weights.sum(dim=1)
        if (denom == 0).any():
            raise ValueError("drug_mask contains an all-padding graph")
        return (tokens * weights).sum(dim=1) / denom

    def forward(
        self,
        drug_vec,
        protein_vec,
        *,
        drug_tokens=None,
        drug_mask=None,
        protein_tokens=None,
        protein_mask=None,
        return_attention: bool = False,
    ):
        if drug_tokens is None or protein_tokens is None:
            raise ValueError(
                "AttentionFusion requires atom-level drug_tokens and "
                "position-level protein_tokens; pooled vectors are insufficient."
            )
        if drug_mask is None or protein_mask is None:
            raise ValueError("AttentionFusion requires drug_mask and protein_mask")
        if drug_tokens.ndim != 3 or protein_tokens.ndim != 3:
            raise ValueError("token tensors must be [B,N,D] and [B,L,D]")
        if drug_mask.shape != drug_tokens.shape[:2]:
            raise ValueError("drug_mask shape does not match drug_tokens")
        if protein_mask.shape != protein_tokens.shape[:2]:
            raise ValueError("protein_mask shape does not match protein_tokens")
        if (~protein_mask).all(dim=1).any():
            raise ValueError("protein_mask contains an all-padding sequence")

        q = self.query_norm(self.drug_token_proj(drug_tokens))
        kv = self.key_value_norm(self.protein_token_proj(protein_tokens))

        cross, attn = self.cross_attention(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=~protein_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        cross = self.cross_norm(q + self.dropout(cross))
        pooled = self._masked_mean(cross, drug_mask)

        # Reuse the same alignment projections for global residual context.
        # This keeps the shared global encoder paths trainable in the attention
        # model while the selective interaction itself remains token-level.
        d_global = self.drug_token_proj(drug_vec)
        p_global = self.protein_token_proj(protein_vec)
        pooled = pooled + 0.5 * (d_global + p_global)

        fused = F.relu(self.output_proj(pooled))
        fused = self.dropout(fused)
        fused = self.output_norm(fused)

        if return_attention:
            self.last_attn = attn.detach()
            return fused, attn
        self.last_attn = None
        return fused


# More explicit scientific alias.  The public experiment name remains
# "attention" for continuity in command-line arguments and tables.
TokenCrossAttentionFusion = AttentionFusion


def get_fusion_module(
    fusion_mode,
    drug_dim,
    protein_dim,
    hidden_dim,
    num_heads=4,
    dropout=0.2,
    align_dim=128,
    attention_align_dim=80,
):
    mode = str(fusion_mode).lower()
    common = dict(
        drug_dim=drug_dim,
        protein_dim=protein_dim,
        hidden_dim=hidden_dim,
        align_dim=align_dim,
        num_heads=num_heads,
        dropout=dropout,
    )
    if mode == "concat":
        return ConcatFusion(**common)
    if mode == "product":
        return ProductFusion(**common)
    if mode == "attention":
        attention_common = dict(common)
        attention_common["align_dim"] = int(attention_align_dim)
        return AttentionFusion(**attention_common)
    raise ValueError(f"Unknown fusion_mode: {fusion_mode}")


def fusion_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
