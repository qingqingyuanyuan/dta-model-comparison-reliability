"""Canonical model API for ZhiYao-Graph V3."""

from .drug_encoder import DrugGNNEncoder
from .protein_encoder import ProteinCNNEncoder, ProteinTransformerEncoder
from .fusion import (
    AttentionFusion,
    TokenCrossAttentionFusion,
    ConcatFusion,
    ProductFusion,
    fusion_parameter_count,
    get_fusion_module,
)
from .predictor import ARCHITECTURE_VERSION, DTAPredictor, DTIPredictor
from .colab_model import load_model, load_legacy_model

__all__ = [
    "ARCHITECTURE_VERSION",
    "DrugGNNEncoder",
    "ProteinCNNEncoder",
    "ProteinTransformerEncoder",
    "ConcatFusion",
    "ProductFusion",
    "AttentionFusion",
    "TokenCrossAttentionFusion",
    "fusion_parameter_count",
    "get_fusion_module",
    "DTAPredictor",
    "DTIPredictor",
    "load_model",
    "load_legacy_model",
]
