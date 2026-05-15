"""
GCS-Head: Geometry-Constrained Structured Head
===============================================
Paper Section 2.5

Simultaneously handles:
    1. Object detection (bounding box regression + multi-class classification)
    2. Depth regression  (continuous depth value per fruit, normalized [0,1])
    3. Tracking embedding (L2-normalized feature for identity association)
    4. Plant identification (classify which plant stem each fruit belongs to)

Geometric constraint mechanism (§2.5.2):
    Builds a dynamic relational graph between fruits and stems.
    Computes geometric compatibility score S = w1*S_adj + w2*S_cluster + w3*S_dir
    for each (fruit, stem) candidate pair.
    Loss: negative log-likelihood that penalizes geometrically implausible associations.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Feature processing backbone of GCS-Head (§2.5.1)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureNormMLP(nn.Module):
    """LayerNorm → two-layer MLP with GELU → residual connection (§2.5.1)."""

    def __init__(self, dim: int, expand_ratio: float = 2.0):
        super().__init__()
        hidden = int(dim * expand_ratio)
        self.norm = nn.LayerNorm(dim)
        self.mlp  = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, dim). Returns (B, N, dim) with residual."""
        return x + self.mlp(self.norm(x))


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific prediction branches
# ─────────────────────────────────────────────────────────────────────────────

class DetectionBranch(nn.Module):
    """Predicts bounding box (cx, cy, w, h) and class logits."""

    def __init__(self, in_dim: int, num_classes: int = 4):
        """
        Args:
            num_classes: 4 by default (unripe, semi-mature, mature tomatoes, stems)
        """
        super().__init__()
        self.num_classes = num_classes
        # Box regression: 4 coordinates
        self.bbox_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, 4),
            nn.Sigmoid(),  # normalized box coordinates
        )
        # Classification: logits for each category
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x : (B, N, in_dim)
        Returns:
            boxes  : (B, N, 4)            — (cx, cy, w, h) normalized
            logits : (B, N, num_classes)  — raw classification logits
        """
        return self.bbox_head(x), self.cls_head(x)


class DepthRegressionBranch(nn.Module):
    """
    Regresses a single depth value per query.
    Architecture: 512 → 128 → 32 → 1, with Sigmoid to [0,1] then
    linear mapping to the effective operating range (default 0–0.75 m).

    Optional confidence estimation branch shares feature layers but has
    a separate output head (reliability score for depth predictions).
    """

    def __init__(self, in_dim: int, max_depth_m: float = 0.75,
                 with_confidence: bool = True):
        super().__init__()
        self.max_depth_m = max_depth_m
        self.with_confidence = with_confidence

        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
        )
        self.depth_out = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        if with_confidence:
            self.conf_out = nn.Sequential(
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x : (B, N, in_dim)
        Returns:
            depth : (B, N, 1) in meters [0, max_depth_m]
            conf  : (B, N, 1) ∈ [0,1] reliability score (or None)
        """
        feat  = self.shared(x)
        depth = self.depth_out(feat) * self.max_depth_m
        conf  = self.conf_out(feat) if self.with_confidence else None
        return depth, conf


class TrackingEmbeddingBranch(nn.Module):
    """
    Produces L2-normalized feature embeddings for identity association.
    Cosine distance between embeddings serves as the similarity metric.

    Architecture (paper §2.5.1):
        256 → 512 → LayerNorm + ReLU → 256 → L2-normalize
    """

    def __init__(self, in_dim: int = 256, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, in_dim)
        Returns:
            embed : (B, N, embed_dim)  L2-normalized, unit hypersphere
        """
        feat = self.net(x)
        return F.normalize(feat, p=2, dim=-1)


class PlantIDHead(nn.Module):
    """
    Plant identification: classify which plant each fruit belongs to.

    Architecture (paper §2.5.1):
        128-dim plant ID features → 32 → spatial attention → classification
    """

    def __init__(self, in_dim: int, num_plants: int = 210):
        """
        Args:
            num_plants: total plant IDs (210 in GH-Tomato-MOTC dataset)
        """
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
        )
        # Spatial attention: weighted association between feature points and positions
        self.spatial_attn = nn.Sequential(
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1),
        )
        self.cls = nn.Linear(32, num_plants)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, N, in_dim)
        Returns:
            logits : (B, N, num_plants)
        """
        feat = self.feat_net(x)                     # (B, N, 32)
        attn = self.spatial_attn(feat)              # (B, N, 1)
        feat = feat * attn                          # weighted by spatial attention
        return self.cls(feat)                       # (B, N, num_plants)


# ─────────────────────────────────────────────────────────────────────────────
# Geometric Constraint Module (§2.5.2)
# ─────────────────────────────────────────────────────────────────────────────

class GeometricConstraintModule(nn.Module):
    """
    Computes geometric compatibility scores between detected fruits and stems.

    S = w_adj * S_adj + w_cluster * S_cluster + w_dir * S_dir

    Three components (paper §2.5.2):
        S_adj     : spatial adjacency (Gaussian of center-to-center distance)
        S_cluster : cluster-wise distribution (how tightly fruits cluster around a stem)
        S_dir     : cosine of angular alignment (stem direction vs. fruit position)

    The geometric constraint loss:  L_geo = -log(softmax(S)[correct_plant])

    Args:
        w_adj     : weight for adjacency score
        w_cluster : weight for cluster score
        w_dir     : weight for directional alignment score
        sigma     : distance scale for S_adj Gaussian kernel
        temperature : softmax temperature τ (paper uses τ=1.0)
    """

    def __init__(self, w_adj: float = 0.4, w_cluster: float = 0.3,
                 w_dir: float = 0.3, sigma: float = 0.15, temperature: float = 1.0):
        super().__init__()
        self.w_adj     = w_adj
        self.w_cluster = w_cluster
        self.w_dir     = w_dir
        self.sigma     = sigma
        self.temperature = temperature

    def compute_scores(
        self,
        fruit_boxes: torch.Tensor,   # (B, N_f, 4) fruits (cx,cy,w,h) normalized
        stem_boxes:  torch.Tensor,   # (B, N_s, 4) stems  (cx,cy,w,h) normalized
    ) -> torch.Tensor:
        """
        Returns:
            scores : (B, N_f, N_s) — compatibility score for each (fruit, stem) pair
        """
        B, N_f, _ = fruit_boxes.shape
        N_s = stem_boxes.size(1)

        # Centers
        f_cxy = fruit_boxes[..., :2]   # (B, N_f, 2)
        s_cxy = stem_boxes[..., :2]    # (B, N_s, 2)

        # Expand for broadcasting: (B, N_f, N_s, 2)
        f_exp = f_cxy.unsqueeze(2).expand(-1, -1, N_s, -1)
        s_exp = s_cxy.unsqueeze(1).expand(-1, N_f, -1, -1)

        diff = f_exp - s_exp  # (B, N_f, N_s, 2)
        dist = diff.norm(dim=-1)  # (B, N_f, N_s)

        # S_adj: Gaussian of distance (smaller distance → higher score)
        s_adj = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))

        # S_cluster: measure how tightly the fruit clusters around the stem
        # Approximated as inverse of fruit size * stem size ratio
        f_area = (fruit_boxes[..., 2] * fruit_boxes[..., 3]).unsqueeze(2)  # (B, N_f, 1)
        s_area = (stem_boxes[..., 2] * stem_boxes[..., 3]).unsqueeze(1)    # (B, 1, N_s)
        area_ratio = f_area / (s_area.clamp(min=1e-6))
        s_cluster = torch.exp(-torch.abs(area_ratio - 0.5))                 # centered at 0.5

        # S_dir: cosine of angular alignment between stem direction and fruit-to-stem vector
        # Stem direction approximated as vertical (0, -1) for vine tomatoes
        stem_dir = torch.tensor([0.0, -1.0], device=fruit_boxes.device)    # (2,)
        vec_norm = F.normalize(diff, p=2, dim=-1)                          # (B, N_f, N_s, 2)
        cos_angle = (vec_norm * stem_dir).sum(dim=-1)                       # (B, N_f, N_s)
        s_dir = cos_angle.clamp(min=0.0)  # only positive alignment counts

        # Weighted sum
        scores = (self.w_adj     * s_adj
                + self.w_cluster * s_cluster
                + self.w_dir     * s_dir)   # (B, N_f, N_s)
        return scores

    def loss(
        self,
        fruit_boxes: torch.Tensor,   # (B, N_f, 4)
        stem_boxes:  torch.Tensor,   # (B, N_s, 4)
        plant_ids:   torch.Tensor,   # (B, N_f)  ground-truth plant ID for each fruit (index into stem dim)
    ) -> torch.Tensor:
        """
        Geometric constraint loss: negative log-likelihood.
        L_geo = - mean_{b,i} log( softmax(S[b,i,:] / τ)[plant_ids[b,i]] )
        """
        scores = self.compute_scores(fruit_boxes, stem_boxes)       # (B, N_f, N_s)
        log_probs = F.log_softmax(scores / self.temperature, dim=-1) # (B, N_f, N_s)

        # Gather correct plant log-prob
        plant_ids_clamped = plant_ids.clamp(0, stem_boxes.size(1) - 1)
        correct_log_prob = log_probs.gather(
            dim=-1, index=plant_ids_clamped.unsqueeze(-1)
        ).squeeze(-1)  # (B, N_f)

        return -correct_log_prob.mean()


# ─────────────────────────────────────────────────────────────────────────────
# GCS-Head (full)
# ─────────────────────────────────────────────────────────────────────────────

class GCSHead(nn.Module):
    """
    GCS-Head: Geometry-Constrained Structured Head.

    Processes UCL-Decoder output features into structured multi-task outputs:
        - Detection (box + class)
        - Depth regression
        - Tracking embedding
        - Plant identification

    Also applies majority-voting on trajectory history to smooth plant association,
    and integrates the geometric constraint loss during training.

    Args:
        embed_dim   : input feature dimension from UCL-Decoder (default 256)
        num_classes : number of object categories (default 4: unripe/semi/ripe/stem)
        num_plants  : number of plant IDs in the dataset (default 210)
        max_depth_m : effective operating depth range in meters (default 0.75)
    """

    def __init__(self, embed_dim: int = 256, num_classes: int = 4,
                 num_plants: int = 210, max_depth_m: float = 0.75):
        super().__init__()
        self.embed_dim = embed_dim

        # Shared feature processing (LayerNorm + MLP + residual, §2.5.1)
        self.shared_mlp = FeatureNormMLP(embed_dim, expand_ratio=2.0)

        # Task-specific feature projections (split embed_dim into task-specific sub-dims)
        self.proj_det   = nn.Linear(embed_dim, 256)   # detection
        self.proj_depth = nn.Linear(embed_dim, 256)   # depth regression
        self.proj_track = nn.Linear(embed_dim, 256)   # tracking embedding
        self.proj_plant = nn.Linear(embed_dim, 128)   # plant identification

        # Task-specific heads
        self.det_branch   = DetectionBranch(256, num_classes)
        self.depth_branch = DepthRegressionBranch(256, max_depth_m, with_confidence=True)
        self.track_branch = TrackingEmbeddingBranch(256, embed_dim)
        self.plant_head   = PlantIDHead(128, num_plants)

        # Geometric constraint module
        self.geo_constraint = GeometricConstraintModule()

        # Majority-voting window for plant association smoothing (length=5 frames)
        self.majority_vote_window = 5

    def forward(
        self,
        decoded: torch.Tensor,  # (B, N, embed_dim) from UCL-Decoder
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            decoded : (B, N, embed_dim) — UCL-Decoder output

        Returns a dict with:
            'boxes'       : (B, N, 4)           — (cx, cy, w, h) normalized
            'logits'      : (B, N, num_classes) — class logits
            'depth'       : (B, N, 1)           — depth in meters
            'depth_conf'  : (B, N, 1)           — depth reliability
            'track_embed' : (B, N, embed_dim)   — L2-norm tracking features
            'plant_logits': (B, N, num_plants)  — plant ID logits
        """
        # 1. Shared feature enhancement
        x = self.shared_mlp(decoded)  # (B, N, embed_dim)

        # 2. Task-specific projections
        det_feat   = self.proj_det(x)    # (B, N, 256)
        depth_feat = self.proj_depth(x)  # (B, N, 256)
        track_feat = self.proj_track(x)  # (B, N, 256)
        plant_feat = self.proj_plant(x)  # (B, N, 128)

        # 3. Task-specific heads
        boxes, logits     = self.det_branch(det_feat)
        depth, depth_conf = self.depth_branch(depth_feat)
        track_embed       = self.track_branch(track_feat)
        plant_logits      = self.plant_head(plant_feat)

        return {
            "boxes"        : boxes,
            "logits"       : logits,
            "depth"        : depth,
            "depth_conf"   : depth_conf,
            "track_embed"  : track_embed,
            "plant_logits" : plant_logits,
        }

    def compute_geo_loss(
        self,
        pred_boxes: torch.Tensor,  # (B, N, 4) all predictions
        pred_cls:   torch.Tensor,  # (B, N, num_classes) class logits
        plant_ids:  torch.Tensor,  # (B, N_f) GT plant association for fruits
        num_classes: int = 4,
    ) -> torch.Tensor:
        """
        Compute geometric constraint loss for fruit–stem association.

        Separates fruit and stem predictions by argmax class, then evaluates
        how well predicted boxes satisfy the geometric spatial priors.
        """
        cls_pred = pred_cls.argmax(dim=-1)  # (B, N)
        stem_cls_idx = num_classes - 1      # stem is the last category

        # Separate fruit and stem detections
        B = pred_boxes.size(0)
        geo_losses = []
        for b in range(B):
            is_stem  = (cls_pred[b] == stem_cls_idx)
            is_fruit = ~is_stem

            fruit_boxes = pred_boxes[b][is_fruit].unsqueeze(0)  # (1, N_f, 4)
            stem_boxes  = pred_boxes[b][is_stem].unsqueeze(0)   # (1, N_s, 4)
            gt_pid      = plant_ids[b].unsqueeze(0)             # (1, N_f)

            if fruit_boxes.size(1) == 0 or stem_boxes.size(1) == 0:
                continue

            loss = self.geo_constraint.loss(fruit_boxes, stem_boxes, gt_pid)
            geo_losses.append(loss)

        if not geo_losses:
            return pred_boxes.sum() * 0.0  # return zero-grad tensor
        return torch.stack(geo_losses).mean()
