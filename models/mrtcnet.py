"""
MRTC-Net: Multi-modal Real-time Tracking and Counting Network
=============================================================
Paper: "MRTC-Net: A Multi-modal Real-time Tracking and Counting Network
        for Tomato Yield Estimation in Greenhouse"

Full model assembling DGM-Encoder + UCL-Decoder + GCS-Head on top of
a pre-trained RT-DETR-L (ResNet50) backbone.

Usage:
    from models import MRTCNet

    model = MRTCNet(num_classes=4, num_plants=210)
    model.load_state_dict(torch.load("checkpoints/mrtcnet_best.pt"))
    model.eval()

    # Inference (single frame)
    with torch.no_grad():
        outputs = model(
            rgb   = rgb_tensor,      # (B, 3, H, W) normalized
            depth = depth_tensor,    # (B, 1, H, W) [0,1]
            prev_state = None,       # or dict from previous frame
        )
        boxes       = outputs["boxes"]        # (B, N, 4)
        class_ids   = outputs["logits"].argmax(-1)
        track_ids   = outputs["track_embed"]  # for ByteTrack / identity association
        plant_ids   = outputs["plant_logits"].argmax(-1)
        depth_m     = outputs["depth"]
        prev_state  = outputs["state"]        # pass to next frame for closed-loop

    # Training: see scripts/train.py
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dgm_encoder import DGMEncoder
from .ucl_decoder import UCLDecoder
from .gcs_head    import GCSHead


class RTDETRBackbone(nn.Module):
    """
    Lightweight wrapper to load a pre-trained RT-DETR-L backbone via ultralytics.

    The backbone extracts multi-scale feature maps from RGB images.
    We take the C5 feature map (stride=32) as the primary encoder input,
    matching RT-DETR's architecture.

    In the full paper implementation, this wraps the official RT-DETR-L
    (ResNet50) encoder. For standalone use, a simple CNN stub is provided.
    """

    def __init__(self, pretrained_path: Optional[str] = None, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim

        if pretrained_path is not None:
            self._load_rtdetr(pretrained_path)
        else:
            # Lightweight stub for testing without pre-trained weights
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                nn.Conv2d(256, embed_dim, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
            )
            self._use_stub = True
        # else:
        self._use_stub = False if pretrained_path else True

    def _load_rtdetr(self, path: str):
        """Load RT-DETR-L backbone weights from ultralytics checkpoint."""
        try:
            from ultralytics import RTDETR
            rtdetr = RTDETR(path)
            self.backbone = rtdetr.model.model  # access internal backbone
            self._use_stub = False
        except Exception as e:
            print(f"[MRTC-Net] Warning: could not load RT-DETR from {path}: {e}")
            print("[MRTC-Net] Falling back to lightweight CNN stub.")
            self._init_stub()
            self._use_stub = True

    def _init_stub(self):
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.embed_dim), nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb : (B, 3, H, W)
        Returns:
            feat : (B, embed_dim, H/32, W/32)
        """
        if self._use_stub:
            return self.stem(rgb)
        else:
            # In the full implementation, extract C5 feature from RT-DETR backbone
            feats = self.backbone(rgb)
            return feats[-1] if isinstance(feats, (list, tuple)) else feats


# ─────────────────────────────────────────────────────────────────────────────
# MRTC-Net
# ─────────────────────────────────────────────────────────────────────────────

class MRTCNet(nn.Module):
    """
    MRTC-Net: Full model for tomato tracking, fruit–plant association,
    and plant-level yield estimation.

    Architecture (Fig. 3 in paper):
        RGB + Depth → Backbone → DGM-Encoder → UCL-Decoder → GCS-Head
                                                    ↑
                              closed-loop feedback (prev frame state)

    Args:
        embed_dim       : feature embedding dimension (256 for RT-DETR-L scale)
        num_classes     : number of detection categories (4: unripe/semi/ripe/stem)
        num_plants      : number of plant IDs (210 in GH-Tomato-MOTC)
        num_queries     : number of initial detection queries
        num_dec_layers  : number of UCL-Decoder transformer layers
        max_depth_m     : effective harvesting depth range in meters (0.75 m)
        pretrained_rtdetr: path to pre-trained RT-DETR-L weights (.pt)
        tau_low         : uncertainty threshold for stable tracking state
        tau_high        : uncertainty threshold for lost tracking state
    """

    def __init__(
        self,
        embed_dim:        int   = 256,
        num_classes:      int   = 4,
        num_plants:       int   = 210,
        num_queries:      int   = 300,
        num_dec_layers:   int   = 6,
        max_depth_m:      float = 0.75,
        pretrained_rtdetr: Optional[str] = None,
        tau_low:          float = 0.3,
        tau_high:         float = 0.7,
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_classes = num_classes

        # ── Modules ─────────────────────────────────────────────────────────
        self.backbone = RTDETRBackbone(pretrained_rtdetr, embed_dim)

        self.dgm_encoder = DGMEncoder(
            in_channels=embed_dim,
            num_heads=8,
            num_layers=1,  # single layer as in paper
        )

        self.ucl_decoder = UCLDecoder(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=num_dec_layers,
            num_queries=num_queries,
            tau_low=tau_low,
            tau_high=tau_high,
        )

        self.gcs_head = GCSHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_plants=num_plants,
            max_depth_m=max_depth_m,
        )

        # Project flat encoder output to sequence tokens for decoder
        self.enc_proj = nn.Sequential(
            nn.Flatten(2),  # (B, C, H*W)
        )

    def forward(
        self,
        rgb:        torch.Tensor,               # (B, 3, H, W)
        depth:      torch.Tensor,               # (B, 1, H, W) normalized [0,1]
        prev_state: Optional[Dict] = None,      # closed-loop feedback from prev frame
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb        : RGB image batch, values in [0,1] (normalized)
            depth      : single-channel depth map, values in [0,1]
            prev_state : dict with keys 'queries', 'motion', 'uncertainty'
                         (returned by this function on previous frame)
                         Pass None for the first frame or when resetting tracking.

        Returns a dict:
            'boxes'        : (B, N, 4)            — predicted bounding boxes (cx,cy,w,h)
            'logits'       : (B, N, num_classes)  — classification logits
            'depth'        : (B, N, 1)            — depth in meters
            'depth_conf'   : (B, N, 1)            — depth confidence
            'track_embed'  : (B, N, embed_dim)    — tracking embedding (L2-normalized)
            'plant_logits' : (B, N, num_plants)   — plant ID logits
            'uncertainty'  : (B, N, 1)            — per-query uncertainty
            'query_types'  : (B, N)               — 0=init, 1=tracking, 2=recovery
            'state'        : dict for next frame's prev_state
        """
        B = rgb.size(0)

        # ── 1. Backbone: extract RGB feature map ────────────────────────────
        rgb_feat = self.backbone(rgb)  # (B, C, H', W')

        # ── 2. DGM-Encoder: depth-gated multimodal encoding ─────────────────
        enc_feat = self.dgm_encoder(rgb_feat, depth)  # (B, C, H', W')

        # Flatten spatial dims to sequence tokens for decoder
        B, C, Hf, Wf = enc_feat.shape
        memory = enc_feat.flatten(2).permute(0, 2, 1)  # (B, H'*W', C)

        # ── 3. UCL-Decoder: uncertainty-driven closed-loop decoding ──────────
        prev_queries = prev_state.get("queries") if prev_state else None
        prev_motion  = prev_state.get("motion")  if prev_state else None
        prev_uncert  = prev_state.get("uncertainty") if prev_state else None

        decoded, uncertainty, motion_state, query_types = self.ucl_decoder(
            memory=memory,
            prev_queries=prev_queries,
            prev_motion=prev_motion,
            prev_uncert=prev_uncert,
        )  # decoded: (B, N_q, C)

        # ── 4. GCS-Head: multi-task structured prediction ────────────────────
        head_out = self.gcs_head(decoded)

        # ── 5. Package next-frame state for closed-loop feedback ─────────────
        next_state = {
            "queries"    : decoded.detach(),
            "motion"     : motion_state.detach(),
            "uncertainty": uncertainty.detach(),
        }

        return {
            **head_out,
            "uncertainty"  : uncertainty,
            "query_types"  : query_types,
            "state"        : next_state,
        }

    def compute_loss(
        self,
        outputs:     Dict[str, torch.Tensor],
        targets:     Dict[str, torch.Tensor],
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total training loss.

        Args:
            outputs : dict from forward()
            targets : dict with keys:
                'boxes'    : (B, M, 4) GT boxes
                'classes'  : (B, M) GT class indices
                'depth'    : (B, M, 1) GT depth values in meters
                'plant_ids': (B, M) GT plant IDs per fruit
                'track_ids': (B, M) GT tracking IDs (for embedding loss)
            loss_weights : optional per-loss scaling dict

        Returns:
            losses : dict with 'total' and individual loss terms
        """
        if loss_weights is None:
            loss_weights = {
                "det_cls"   : 1.0,
                "det_box"   : 5.0,
                "depth"     : 1.0,
                "track"     : 1.0,
                "plant"     : 1.0,
                "geo"       : 0.5,
                "uncertainty": 0.1,
            }

        losses = {}

        # ── Detection losses (GIoU + classification focal loss) ─────────────
        # (Simplified; full implementation uses Hungarian matching as in RT-DETR)
        pred_boxes  = outputs["boxes"]           # (B, N, 4)
        pred_logits = outputs["logits"]          # (B, N, num_classes)
        gt_boxes    = targets["boxes"]           # (B, M, 4)
        gt_cls      = targets["classes"]         # (B, M)

        # Classification loss (cross-entropy on matched queries)
        # Note: proper implementation requires set-prediction matching (DETR-style)
        losses["det_cls"] = F.cross_entropy(
            pred_logits.reshape(-1, self.num_classes),
            gt_cls.flatten().clamp(0, self.num_classes - 1),
            ignore_index=-1,
        )

        # Box regression loss (L1 as proxy; replace with GIoU in full training)
        min_n = min(pred_boxes.size(1), gt_boxes.size(1))
        losses["det_box"] = F.l1_loss(pred_boxes[:, :min_n], gt_boxes[:, :min_n])

        # ── Depth regression loss (smooth L1) ───────────────────────────────
        if "depth" in targets:
            gt_depth = targets["depth"]  # (B, M, 1)
            pred_depth = outputs["depth"][:, :gt_depth.size(1)]
            losses["depth"] = F.smooth_l1_loss(pred_depth, gt_depth)

        # ── Tracking embedding loss (triplet loss approximation) ─────────────
        # (Simplified to embedding L2 regularization here)
        track_embed = outputs["track_embed"]
        losses["track"] = (1.0 - (track_embed ** 2).sum(-1).mean())

        # ── Plant ID classification loss ─────────────────────────────────────
        if "plant_ids" in targets:
            gt_plant = targets["plant_ids"]  # (B, M)
            pred_plant = outputs["plant_logits"][:, :gt_plant.size(1)]
            losses["plant"] = F.cross_entropy(
                pred_plant.reshape(-1, pred_plant.size(-1)),
                gt_plant.flatten(),
            )

        # ── Geometric constraint loss ────────────────────────────────────────
        if "plant_ids" in targets:
            losses["geo"] = self.gcs_head.compute_geo_loss(
                pred_boxes, pred_logits, targets["plant_ids"], self.num_classes
            )

        # ── Uncertainty calibration loss (entropy regularization) ────────────
        u = outputs["uncertainty"]
        losses["uncertainty"] = -(u * torch.log(u.clamp(1e-6))
                                  + (1-u) * torch.log((1-u).clamp(1e-6))).mean()

        # ── Total weighted loss ───────────────────────────────────────────────
        total = sum(loss_weights.get(k, 1.0) * v for k, v in losses.items())
        losses["total"] = total

        return losses
