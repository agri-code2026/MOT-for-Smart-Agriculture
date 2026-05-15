"""
MRTC-Net Model Components
=========================
Multi-modal Real-time Tracking and Counting Network for Tomato Yield Estimation.

Architecture:
    - DGM-Encoder : Depth-Gated Multimodal Encoder (DGA + M-AIFI)
    - UCL-Decoder : Uncertainty-aware Closed-Loop Decoder
    - GCS-Head    : Geometry-Constrained Structured Head
"""

from .dgm_encoder import DGMEncoder, DepthGatedAttention, MAIFI
from .ucl_decoder import UCLDecoder, UncertaintyDualQuerySet
from .gcs_head import GCSHead, GeometricConstraintModule
from .mrtcnet import MRTCNet

__all__ = [
    "MRTCNet",
    "DGMEncoder",
    "DepthGatedAttention",
    "MAIFI",
    "UCLDecoder",
    "UncertaintyDualQuerySet",
    "GCSHead",
    "GeometricConstraintModule",
]
