"""
DGM-Encoder: Depth-Gated Multimodal Encoder
============================================
Paper Section 2.3

Contains:
  - DepthGatedAttention (DGA): suppresses far-field noise via depth-guided spatial gating
  - MAIFI: Multi-modal Attention-based Intra-scale Feature Interaction for illumination adaptation
  - DGMEncoder: full encoder that replaces the original AIFI module in RT-DETR

The DGM-Encoder is designed as a plug-and-play replacement for RT-DETR's AIFI encoder,
maintaining identical input/output dimensions to allow seamless integration.

Depth images are loaded as single-channel inputs and processed via a 1x1 convolution,
introducing only minimal additional computational overhead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Depth-Gated Attention (DGA)
# ─────────────────────────────────────────────────────────────────────────────

class DepthGatedAttention(nn.Module):
    """
    DGA: Depth-Gated Attention for far-field suppression.

    Core principle (Paper §2.3.1):
        Rather than hard thresholding (non-differentiable), DGA applies a
        learnable spatial gating weight derived from depth maps.  The gate is
        inversely proportional to the depth value, so near-field pixels receive
        high weights and far-field pixels are suppressed — all end-to-end.

    Architecture:
        depth (B,1,H,W) → 1×1 conv → 3×3 conv → Sigmoid → depth_mask (B,1,H,W)
        rgb_feat (B,C,H,W) × depth_mask → gated_feat (B,C,H,W)
        gated_feat undergoes layer norm + learnable scale α for adaptive optimization

    Input dims are identical to output dims (plug-and-play).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        # Depth branch: 1×1 conv → 3×3 conv → Sigmoid
        self.depth_conv1 = nn.Conv2d(1, in_channels // 4, kernel_size=1, bias=False)
        self.depth_conv2 = nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1, bias=False)
        self.depth_bn1   = nn.BatchNorm2d(in_channels // 4)
        self.sigmoid     = nn.Sigmoid()

        # Learnable scale for adaptive optimization (α in the paper)
        self.alpha = nn.Parameter(torch.ones(1))

        # Layer norm for feature stabilization
        self.norm = nn.GroupNorm(1, in_channels)

    def forward(self, rgb_feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat : (B, C, H, W) — RGB feature map from backbone
            depth    : (B, 1, H, W) — single-channel depth map (0–1 normalized)
        Returns:
            gated_feat : (B, C, H, W) — depth-gated RGB features
        """
        # Resize depth to match feature spatial dimensions if needed
        if depth.shape[-2:] != rgb_feat.shape[-2:]:
            depth = F.interpolate(depth, size=rgb_feat.shape[-2:], mode="bilinear", align_corners=False)

        # Depth gate: values closer to camera (small depth) get higher weights
        # Invert depth: gate = 1 - sigmoid(depth_processed) so near→1, far→0
        d = self.depth_conv1(depth)
        d = self.depth_bn1(d)
        d = F.relu(d, inplace=True)
        d = self.depth_conv2(d)
        depth_mask = self.sigmoid(-d)  # invert: large depth → small gate

        # Apply spatial gating
        gated = rgb_feat * (1.0 + self.alpha * depth_mask)
        gated = self.norm(gated)
        return gated


# ─────────────────────────────────────────────────────────────────────────────
# M-AIFI: Multi-modal Attention-based Intra-scale Feature Interaction
# ─────────────────────────────────────────────────────────────────────────────

class MAIFI(nn.Module):
    """
    M-AIFI: Multi-modal Attention-based Intra-scale Feature Interaction.

    Paper §2.3.2:
        Enhances robustness to illumination changes by fusing RGB and depth
        features within the self-attention mechanism.

    Modification over standard AIFI (RT-DETR's encoder):
        - Query (Q): from RGB features
        - Key (K) and Value (V): concatenation of RGB + depth features
        This ensures that similarity scores between Q and K incorporate
        both RGB semantic and spatial depth information.

    A lightweight two-layer MLP with Sigmoid outputs adaptive fusion weights,
    yielding the final multimodal-enriched feature.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        self.scale = head_dim ** -0.5

        # Q from RGB; K,V from [RGB || depth] projected to embed_dim
        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.kv_proj  = nn.Linear(embed_dim * 2, embed_dim * 2)  # concat(RGB, depth) → K, V
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Adaptive fusion MLP: concat(RGB_attn, depth_feat) → Sigmoid weight β
        mlp_hidden = int(embed_dim * mlp_ratio * 0.5)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Sigmoid(),
        )

        # Feed-forward network (same as standard Transformer FFN)
        ffn_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, rgb_tokens: torch.Tensor, depth_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_tokens   : (B, N, C) — flattened RGB feature tokens
            depth_tokens : (B, N, C) — flattened depth feature tokens (same spatial resolution)
        Returns:
            out : (B, N, C) — fused multimodal feature tokens
        """
        B, N, C = rgb_tokens.shape
        H = self.num_heads
        Hd = C // H

        # ── Multi-head attention ────────────────────────────────────────────
        q = self.q_proj(rgb_tokens)  # (B, N, C) from RGB only

        kv_input = torch.cat([rgb_tokens, depth_tokens], dim=-1)  # (B, N, 2C)
        kv = self.kv_proj(kv_input)  # (B, N, 2C)
        k, v = kv.chunk(2, dim=-1)   # each (B, N, C)

        # Reshape for multi-head
        def reshape(x):
            return x.reshape(B, N, H, Hd).permute(0, 2, 1, 3)

        q, k, v = reshape(q), reshape(k), reshape(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        attn_out = self.out_proj(attn_out)

        # ── Adaptive fusion weight β ────────────────────────────────────────
        fusion_input = torch.cat([attn_out, depth_tokens], dim=-1)
        beta = self.fusion_mlp(fusion_input)  # (B, N, C), values in [0,1]

        # When depth is reliable, β→1 and attn_out dominates
        fused = beta * attn_out + (1.0 - beta) * rgb_tokens

        # ── Residual + LayerNorm ────────────────────────────────────────────
        x = self.norm1(rgb_tokens + fused)
        x = self.norm2(x + self.ffn(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# DGM-Encoder (full module)
# ─────────────────────────────────────────────────────────────────────────────

class DGMEncoder(nn.Module):
    """
    DGM-Encoder: Depth-Gated Multimodal Encoder.

    Replaces RT-DETR's AIFI encoder module. Maintains identical input/output
    channel dimensions as a plug-and-play component.

    Processing pipeline:
        1. Depth image (1-ch) → 1×1 conv → depth feature tokens
        2. RGB feature map → DGA (spatial gating with depth) → gated RGB features
        3. Gated RGB tokens + depth tokens → M-AIFI (multimodal attention)
        4. Output: enriched multimodal feature tokens (same dim as input RGB)

    Args:
        in_channels  : number of input RGB feature channels (e.g. 256 for RT-DETR-L)
        num_heads    : number of attention heads in M-AIFI
        num_layers   : number of stacked M-AIFI layers
        mlp_ratio    : FFN hidden dim expansion ratio
        dropout      : dropout probability
    """

    def __init__(self, in_channels: int = 256, num_heads: int = 8,
                 num_layers: int = 1, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels

        # Depth branch: map 1-channel depth to embed_dim tokens
        self.depth_stem = nn.Sequential(
            nn.Conv2d(1, in_channels // 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # DGA: far-field suppression via depth gating
        self.dga = DepthGatedAttention(in_channels)

        # M-AIFI layers: multimodal attention for illumination robustness
        self.maifi_layers = nn.ModuleList([
            MAIFI(in_channels, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Positional encoding (sine-based, same as RT-DETR)
        self._register_pos_enc_cache = {}

    @staticmethod
    def _build_2d_sincos_pos_enc(h: int, w: int, embed_dim: int,
                                  device: torch.device) -> torch.Tensor:
        """Build 2D sine-cosine positional encoding of shape (1, h*w, embed_dim)."""
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        y_pos = torch.arange(h, device=device).float()
        x_pos = torch.arange(w, device=device).float()
        y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")  # (h, w)

        dim_half = embed_dim // 4
        omega = torch.arange(dim_half, device=device).float() / dim_half
        omega = 1.0 / (10000 ** omega)  # (dim_half,)

        out_y = y_pos.flatten().unsqueeze(-1) * omega.unsqueeze(0)  # (h*w, dim_half)
        out_x = x_pos.flatten().unsqueeze(-1) * omega.unsqueeze(0)

        pos_enc = torch.cat([
            torch.sin(out_y), torch.cos(out_y),
            torch.sin(out_x), torch.cos(out_x),
        ], dim=-1).unsqueeze(0)  # (1, h*w, embed_dim)
        return pos_enc

    def forward(self, rgb_feat: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat : (B, C, H, W) — RGB feature map (e.g. from ResNet50 backbone)
            depth    : (B, 1, H, W) — single-channel depth map, pixel values normalized [0,1]
                       Invalid/missing depth pixels should be set to 0.
        Returns:
            out : (B, C, H, W) — enriched multimodal feature map (same spatial dims)
        """
        B, C, H, W = rgb_feat.shape

        # Resize depth to match feature spatial dimensions
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=False)

        # 1. DGA: suppress far-field noise in RGB features
        gated_rgb = self.dga(rgb_feat, depth)  # (B, C, H, W)

        # 2. Depth tokens for M-AIFI
        depth_feat = self.depth_stem(depth)  # (B, C, H, W)

        # 3. Flatten to token sequences
        rgb_tokens   = gated_rgb.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        depth_tokens = depth_feat.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        # Add positional encoding
        pos_enc = self._build_2d_sincos_pos_enc(H, W, C, rgb_feat.device)
        rgb_tokens   = rgb_tokens   + pos_enc
        depth_tokens = depth_tokens + pos_enc

        # 4. M-AIFI layers
        x = rgb_tokens
        for layer in self.maifi_layers:
            x = layer(x, depth_tokens)

        # Reshape back to spatial feature map
        out = x.permute(0, 2, 1).reshape(B, C, H, W)
        return out
