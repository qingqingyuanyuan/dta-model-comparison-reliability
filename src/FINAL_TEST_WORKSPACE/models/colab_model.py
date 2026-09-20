"""Checkpoint loading and historical-model compatibility.

New experiments use the canonical classes in ``models.predictor``.  Historical
V1/V2 classes remain here only so old checkpoints can be inspected in the Web
app; they must not be used to produce new paper results.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_add_pool

from .predictor import ARCHITECTURE_VERSION, DTAPredictor as CanonicalDTAPredictor


def _trusted_torch_load(path):
    """Load a user/project-owned checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")



# ---------------------------------------------------------------------------
# Historical V2 architecture — frozen for legacy checkpoint inspection only.
# ---------------------------------------------------------------------------
class LegacyDrugGNNEncoder(nn.Module):
    def __init__(
        self,
        node_dim=41,
        hidden_dim=128,
        num_layers=3,
        dropout=0.2,
        encoder_type="gat",
        gat_heads=4,
        edge_dim=None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim // gat_heads,
                    heads=gat_heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.node_feat = None

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, "edge_attr", None)
        x = self.input_proj(x)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        self.node_feat = x
        return global_add_pool(x, batch)


class LegacyProteinCNNEncoder(nn.Module):
    def __init__(self, vocab_size=26, embed_dim=128, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, 64, k, padding=k // 2) for k in [3, 5, 7]]
        )
        self.fc = nn.Linear(192, hidden_dim)
        self.dropout = dropout
        self.pos_feat = None

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conv_out = torch.cat([F.relu(c(x)) for c in self.convs], dim=1)
        self.pos_feat = conv_out.permute(0, 2, 1)
        x = conv_out.max(dim=2)[0]
        return F.dropout(F.relu(self.fc(x)), p=self.dropout, training=self.training)


class LegacyConcatFusion(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, dv, pv):
        return self.ln(torch.cat([dv, pv], dim=-1))


class LegacyProductFusion(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.dp = nn.Linear(128, hidden_dim)
        self.pp = nn.Linear(128, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, dv, pv):
        return self.fc(self.dp(dv) * self.pp(pv))


class LegacyAttentionFusion(nn.Module):
    """Historical length-1 attention, retained only to load old checkpoints."""

    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.drug_proj = nn.Linear(128, hidden_dim)
        self.protein_proj = nn.Linear(128, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.last_attn = None

    def forward(self, drug_vec, protein_vec):
        b = drug_vec.size(0)
        d = self.drug_proj(drug_vec).unsqueeze(1)
        p = self.protein_proj(protein_vec).unsqueeze(1)
        q = self.q_proj(d).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(p).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(p).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn.detach()
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(
            b, -1, self.num_heads * self.head_dim
        )
        return self.layer_norm(self.output_proj(out.squeeze(1)) + d.squeeze(1))


LEGACY_FUSION_MAP = {
    "concat": LegacyConcatFusion,
    "product": LegacyProductFusion,
    "attention": LegacyAttentionFusion,
}


class LegacyDTIPredictor(nn.Module):
    """Long-name historical model."""

    def __init__(self, encoder_type="gat", fusion_mode="concat", edge_dim=None):
        super().__init__()
        self.drug_encoder = LegacyDrugGNNEncoder(
            encoder_type=encoder_type, edge_dim=edge_dim
        )
        self.protein_encoder = LegacyProteinCNNEncoder()
        self.fusion = LEGACY_FUSION_MAP[fusion_mode]()
        self.predictor = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, drug_graphs, protein_seqs):
        batch = (
            Batch.from_data_list(drug_graphs)
            if isinstance(drug_graphs, list)
            else drug_graphs
        )
        return self.predictor(
            self.fusion(
                self.drug_encoder(batch),
                self.protein_encoder(protein_seqs),
            )
        ).squeeze(-1)


class LegacyModel(nn.Module):
    """Short-name historical V2 notebook model."""

    def __init__(self, fm="concat", edge_dim=None):
        super().__init__()
        self.de = LegacyDrugGNNEncoder(edge_dim=edge_dim)
        self.pe = LegacyProteinCNNEncoder()
        self.fu = LEGACY_FUSION_MAP[fm]()
        self.pr = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.fm = fm

    def forward(self, dg, ps):
        if isinstance(dg, list):
            dg = Batch.from_data_list(dg)
        return self.pr(self.fu(self.de(dg), self.pe(ps))).squeeze(-1)


def _remap_notebook_keys(sd):
    """Map historical V2 short notebook encoder names to legacy class names."""
    if not any(k.startswith("de.ip.") for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("de.ip."):
            k2 = "de.input_proj." + k2[len("de.ip.") :]
        elif k2.startswith("de.c."):
            k2 = "de.convs." + k2[len("de.c.") :]
        elif k2.startswith("de.b."):
            k2 = "de.bns." + k2[len("de.b.") :]
        elif k2.startswith("pe.emb."):
            k2 = "pe.embedding." + k2[len("pe.emb.") :]
        elif k2.startswith("pe.cs."):
            k2 = "pe.convs." + k2[len("pe.cs.") :]
        out[k2] = v
    return out


def _load_legacy_state_strict_enough(model, state_dict):
    """Block the historical silent ``strict=False`` failure mode."""
    model_state = model.state_dict()
    compatible = {
        k: v
        for k, v in state_dict.items()
        if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    key_coverage = len(compatible) / max(1, len(model_state))
    param_coverage = sum(v.numel() for v in compatible.values()) / max(
        1, sum(v.numel() for v in model_state.values())
    )
    if key_coverage < 0.90 or param_coverage < 0.95:
        raise RuntimeError(
            "Legacy checkpoint compatibility too low; refusing silent partial load. "
            f"key coverage={key_coverage:.1%}, parameter coverage={param_coverage:.1%}"
        )
    model.load_state_dict(state_dict, strict=False)


def load_legacy_model(ckpt_path):
    """Load a historical checkpoint for inspection only."""
    ckpt = _trusted_torch_load(ckpt_path)
    sd = _remap_notebook_keys(ckpt["model_state_dict"])
    fm = str(ckpt.get("fusion_mode", "concat")).lower()
    edge_dim = ckpt.get("edge_dim")

    if any(k.startswith("de.") for k in sd):
        model = LegacyModel(fm=fm, edge_dim=edge_dim)
    else:
        model = LegacyDTIPredictor(fusion_mode=fm, edge_dim=edge_dim)
    _load_legacy_state_strict_enough(model, sd)
    model.eval()
    return model, ckpt


def load_model(ckpt_path, *, allow_legacy=True):
    """Load V3 checkpoints strictly; optionally fall back to frozen legacy code."""
    ckpt = _trusted_torch_load(ckpt_path)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Unrecognized checkpoint format")

    if ckpt.get("architecture_version") == ARCHITECTURE_VERSION:
        config = ckpt.get("config")
        if not isinstance(config, dict):
            raise ValueError("V3 checkpoint is missing its exact model config")
        model = CanonicalDTAPredictor(config)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        return model, ckpt

    if not allow_legacy:
        raise RuntimeError(
            "Checkpoint is not a ZhiYao-Graph V3 canonical model. "
            "Legacy loading was not permitted."
        )

    warnings.warn(
        "Loading a historical V1/V2 checkpoint. This is for inspection/Web "
        "demonstration only and must not be reported as a new V3 experiment.",
        RuntimeWarning,
        stacklevel=2,
    )
    return load_legacy_model(ckpt_path)


# Old import names kept only so the historical Web code does not crash.
DrugGNNEncoder = LegacyDrugGNNEncoder
ProteinCNNEncoder = LegacyProteinCNNEncoder
ConcatFusion = LegacyConcatFusion
ProductFusion = LegacyProductFusion
AttentionFusion = LegacyAttentionFusion
FUSION_MAP = LEGACY_FUSION_MAP
DTIPredictor = LegacyDTIPredictor
Model = LegacyModel
