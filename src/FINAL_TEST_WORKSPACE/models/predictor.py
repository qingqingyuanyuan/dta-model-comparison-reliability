"""Canonical ZhiYao-Graph drug-target affinity predictor (V3)."""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from data.build_graph import get_atom_feature_dim
from .drug_encoder import DrugGNNEncoder
from .fusion import get_fusion_module
from .protein_encoder import ProteinCNNEncoder, ProteinTransformerEncoder


ARCHITECTURE_VERSION = "zhiy_graph_v3_token_cross_attention"


class DTAPredictor(nn.Module):
    """Canonical continuous drug-target affinity model.

    All fusion strategies share the same encoders and predictor.  Only the
    fusion module changes.  AttentionFusion receives atom/residue token features
    from those shared encoders.
    """

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = copy.deepcopy(config)
        dc = config["DRUG_ENCODER"]
        pc = config["PROTEIN_ENCODER"]
        fc = config["FUSION"]
        prc = config["PREDICTOR"]

        node_dim = get_atom_feature_dim()
        configured_node_dim = dc.get("node_dim")
        if configured_node_dim is not None and int(configured_node_dim) != node_dim:
            raise ValueError(
                f"Config node_dim={configured_node_dim} but graph builder emits {node_dim}"
            )

        self.drug_encoder = DrugGNNEncoder(
            node_dim=node_dim,
            hidden_dim=dc["hidden_dim"],
            num_layers=dc["num_layers"],
            dropout=dc["dropout"],
            encoder_type=dc["encoder_type"],
            gat_heads=dc["gat_heads"],
            edge_dim=dc.get("edge_dim"),
            pooling=dc.get("pooling", "add"),
        )

        encoder_type = pc.get("encoder_type", "cnn")
        if encoder_type == "transformer":
            self.protein_encoder = ProteinTransformerEncoder(
                vocab_size=pc["vocab_size"],
                embed_dim=pc["embed_dim"],
                hidden_dim=pc["hidden_dim"],
                num_heads=pc["transformer_heads"],
                num_layers=pc["transformer_layers"],
                dropout=pc["dropout"],
                max_len=pc["max_len"],
                padding_idx=pc.get("padding_idx", 0),
            )
        elif encoder_type == "cnn":
            self.protein_encoder = ProteinCNNEncoder(
                vocab_size=pc["vocab_size"],
                embed_dim=pc["embed_dim"],
                kernel_sizes=pc["kernel_sizes"],
                num_filters=pc["num_filters"],
                hidden_dim=pc["hidden_dim"],
                dropout=pc["dropout"],
                padding_idx=pc.get("padding_idx", 0),
                post_pool_layers=pc.get("post_pool_layers", 1),
            )
        else:
            raise ValueError(f"Unknown protein encoder_type: {encoder_type}")

        self.fusion_mode = str(fc["fusion_mode"]).lower()
        self.fusion = get_fusion_module(
            fusion_mode=self.fusion_mode,
            drug_dim=self.drug_encoder.output_dim,
            protein_dim=self.protein_encoder.output_dim,
            hidden_dim=fc["hidden_dim"],
            align_dim=fc.get("align_dim", 128),
            attention_align_dim=fc.get("attention_align_dim", 80),
            num_heads=fc["num_heads"],
            dropout=fc["dropout"],
        )

        layers = []
        in_dim = fc["hidden_dim"]
        for hidden in prc["hidden_dims"]:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Dropout(prc["dropout"]),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(in_dim, prc["output_dim"]))
        self.predictor = nn.Sequential(*layers)
        self.task = prc.get("task", "regression")

    def forward(
        self,
        drug_graphs,
        protein_seqs: torch.Tensor,
        *,
        return_aux: bool = False,
    ):
        if isinstance(drug_graphs, list):
            drug_graphs = Batch.from_data_list(drug_graphs)
        if not isinstance(drug_graphs, Batch) and not hasattr(drug_graphs, "x"):
            raise TypeError("drug_graphs must be a PyG Batch/Data or list of Data")

        need_tokens = bool(self.fusion.requires_token_features or return_aux)
        drug = self.drug_encoder.encode(
            drug_graphs,
            return_tokens=need_tokens,
        )
        protein = self.protein_encoder.encode(
            protein_seqs,
            return_tokens=need_tokens,
        )

        attention = None
        if self.fusion.requires_token_features:
            fused_out = self.fusion(
                drug["global"],
                protein["global"],
                drug_tokens=drug["tokens"],
                drug_mask=drug["mask"],
                protein_tokens=protein["tokens"],
                protein_mask=protein["mask"],
                return_attention=return_aux,
            )
            if return_aux:
                fused, attention = fused_out
            else:
                fused = fused_out
        else:
            fused = self.fusion(drug["global"], protein["global"])

        # Important: no sigmoid here.  BCEWithLogitsLoss, if used in a separate
        # classification experiment, expects raw logits.  The paper task is DTA
        # regression and uses this raw scalar directly.
        prediction = self.predictor(fused).squeeze(-1)

        if not return_aux:
            return prediction

        return {
            "prediction": prediction,
            "fused": fused,
            "drug_global": drug["global"],
            "protein_global": protein["global"],
            "drug_tokens": drug.get("tokens"),
            "drug_mask": drug.get("mask"),
            "protein_tokens": protein.get("tokens"),
            "protein_mask": protein.get("mask"),
            "attention": attention,
        }

    def parameter_summary(self) -> Dict[str, int]:
        modules = {
            "drug_encoder": self.drug_encoder,
            "protein_encoder": self.protein_encoder,
            "fusion": self.fusion,
            "predictor": self.predictor,
        }
        result = {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in modules.items()
        }
        result["total"] = sum(result.values())
        return result


# Temporary backward-compatible alias; the scientific task is DTA regression.
DTIPredictor = DTAPredictor
