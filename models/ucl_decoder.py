"""
UCL-Decoder: Uncertainty-aware Closed-Loop Decoder
===================================================
Paper Section 2.4

Key innovations over standard RT-DETR decoder:
    1. Uncertainty-Driven Dual Query Set
       - Tracking queries   : for stable, visible targets
       - Recovery queries   : for occluded / low-confidence targets
       - Initialization queries : for newly detected targets
    2. Closed-loop feedback mechanism
       - Historical trajectory states feed back into the decoder each frame
       - Transforms passive frame-by-frame matching into active predictive search
    3. Majority-voting trajectory state classifier
       - Tracks classified as STABLE / UNCERTAIN / LOST based on uncertainty score

Target lifecycle (paper Fig. 5):
    init → stable tracking → occlusion-induced transition → identity recovery → stable tracking
"""

import math
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Track state enum
# ─────────────────────────────────────────────────────────────────────────────

class TrackState(IntEnum):
    """Trajectory state categories used for routing query types (paper §2.4)."""
    STABLE      = 0  # uncertainty ≤ τ_low  → tracking query
    UNCERTAIN   = 1  # τ_low < uncertainty ≤ τ_high → recovery query
    LOST        = 2  # uncertainty > τ_high → removed from active pool


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty estimation head
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyHead(nn.Module):
    """
    Lightweight MLP that estimates per-query uncertainty score ∈ [0, 1].
    High uncertainty → target likely occluded or drifting.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query : (B, N, embed_dim)
        Returns:
            uncertainty : (B, N, 1) ∈ [0, 1]
        """
        return self.net(query)


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty-Driven Dual Query Set
# ─────────────────────────────────────────────────────────────────────────────

class UncertaintyDualQuerySet(nn.Module):
    """
    Dynamically constructs a dual query pool comprising:
      - Tracking queries   : for stable targets (uncertainty ≤ τ_low)
      - Recovery queries   : for occluded targets (τ_low < uncertainty ≤ τ_high)

    The recovery queries encode historical appearance and motion priors,
    enabling the model to re-associate targets after prolonged occlusion
    — analogous to how a human observer memorizes and later re-identifies a fruit.

    Args:
        embed_dim   : query embedding dimension
        num_queries : initial number of detection queries (like RT-DETR)
        tau_low     : uncertainty threshold for stable tracking (paper default ≈ 0.3)
        tau_high    : uncertainty threshold for lost tracks (paper default ≈ 0.7)
    """

    def __init__(self, embed_dim: int = 256, num_queries: int = 300,
                 tau_low: float = 0.3, tau_high: float = 0.7):
        super().__init__()
        self.embed_dim   = embed_dim
        self.num_queries = num_queries
        self.tau_low     = tau_low
        self.tau_high    = tau_high

        # Learnable init queries (detection queries for new instances)
        self.init_queries = nn.Embedding(num_queries, embed_dim)

        # Recovery query encoder: takes [last_appearance_feat || motion_prior] → recovery query
        self.recovery_encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

        # Uncertainty estimator
        self.uncertainty_head = UncertaintyHead(embed_dim)

    def classify_states(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """
        Map per-query uncertainty scores to TrackState labels.

        Args:
            uncertainty : (B, N, 1) ∈ [0, 1]
        Returns:
            states : (B, N) int tensor with values in TrackState
        """
        u = uncertainty.squeeze(-1)  # (B, N)
        states = torch.full_like(u, TrackState.STABLE, dtype=torch.long)
        states[u > self.tau_low]  = TrackState.UNCERTAIN
        states[u > self.tau_high] = TrackState.LOST
        return states

    def build_query_pool(
        self,
        prev_queries:   Optional[torch.Tensor],   # (B, N_prev, C) or None
        prev_motion:    Optional[torch.Tensor],   # (B, N_prev, C) predicted motion embedding
        uncertainty:    Optional[torch.Tensor],   # (B, N_prev, 1)
        batch_size:     int,
        device:         torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the full query pool for the current frame.

        Returns:
            queries : (B, N_total, C)  — concatenated query embeddings
            query_types : (B, N_total) — 0=init, 1=tracking, 2=recovery
        """
        # --- Initialization queries (constant learned embeddings) ----------
        init_q = self.init_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (B, Q, C)
        init_types = torch.zeros(batch_size, self.num_queries, dtype=torch.long, device=device)

        if prev_queries is None:
            return init_q, init_types

        # --- Classify previous queries by uncertainty ----------------------
        states = self.classify_states(uncertainty)  # (B, N_prev)

        # --- Tracking queries: stable targets (uncertainty ≤ τ_low) --------
        stable_mask = (states == TrackState.STABLE)  # (B, N_prev)

        # --- Recovery queries: occluded targets (τ_low < u ≤ τ_high) ------
        recover_mask = (states == TrackState.UNCERTAIN)

        # Build per-sample query lists (variable-length; pad to max)
        tracking_qs  = []
        recovery_qs  = []
        t_types_list = []
        r_types_list = []

        for b in range(batch_size):
            tq = prev_queries[b][stable_mask[b]]   # (n_stable, C)
            tracking_qs.append(tq)
            t_types_list.append(torch.ones(tq.size(0), dtype=torch.long, device=device))

            # Recovery query: fuse appearance feature + motion prior
            rq_feats = prev_queries[b][recover_mask[b]]   # (n_occluded, C)
            if prev_motion is not None:
                rm_feats = prev_motion[b][recover_mask[b]]    # (n_occluded, C)
                rq = self.recovery_encoder(torch.cat([rq_feats, rm_feats], dim=-1))
            else:
                dummy_motion = torch.zeros_like(rq_feats)
                rq = self.recovery_encoder(torch.cat([rq_feats, dummy_motion], dim=-1))
            recovery_qs.append(rq)
            r_types_list.append(torch.full((rq.size(0),), 2, dtype=torch.long, device=device))

        # Pad and concatenate along query dimension
        def pad_and_stack(lst, max_n=None):
            if max_n is None:
                max_n = max(x.size(0) for x in lst) if lst else 0
            C = lst[0].size(-1) if lst and lst[0].size(0) > 0 else self.embed_dim
            padded = []
            for x in lst:
                if x.size(0) < max_n:
                    pad = torch.zeros(max_n - x.size(0), C, device=device)
                    x   = torch.cat([x, pad], dim=0)
                padded.append(x)
            return torch.stack(padded, dim=0) if padded else torch.zeros(batch_size, 0, C, device=device)

        def pad_types(lst, max_n=None, fill=0):
            if max_n is None:
                max_n = max(x.size(0) for x in lst) if lst else 0
            padded = []
            for x in lst:
                if x.size(0) < max_n:
                    pad = torch.full((max_n - x.size(0),), fill, dtype=torch.long, device=device)
                    x   = torch.cat([x, pad], dim=0)
                padded.append(x)
            return torch.stack(padded, dim=0) if padded else torch.zeros(batch_size, 0, dtype=torch.long, device=device)

        track_q = pad_and_stack(tracking_qs)
        recov_q = pad_and_stack(recovery_qs)
        track_types = pad_types(t_types_list, max_n=track_q.size(1), fill=1)
        recov_types = pad_types(r_types_list, max_n=recov_q.size(1), fill=2)

        # Concatenate: [init | tracking | recovery]
        all_queries = torch.cat([init_q, track_q, recov_q], dim=1)
        all_types   = torch.cat([init_types, track_types, recov_types], dim=1)

        return all_queries, all_types


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Decoder Layer (standard, shared with RT-DETR)
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """Single transformer decoder layer with cross-attention + self-attention + FFN."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        ffn_dim = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        tgt:      torch.Tensor,   # (B, N_q, C)  queries
        memory:   torch.Tensor,   # (B, N_m, C)  encoder output (keys/values)
        tgt_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.drop(tgt2))

        # Cross-attention (query attends to encoder memory)
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=mem_mask)
        tgt = self.norm2(tgt + self.drop(tgt2))

        # FFN
        tgt = self.norm3(tgt + self.drop(self.ffn(tgt)))
        return tgt


# ─────────────────────────────────────────────────────────────────────────────
# UCL-Decoder
# ─────────────────────────────────────────────────────────────────────────────

class UCLDecoder(nn.Module):
    """
    UCL-Decoder: Uncertainty-aware Closed-Loop Decoder.

    Paper §2.4:
        End-to-end multi-object tracking decoder.  Mitigates serial error
        accumulation of conventional TBD frameworks through:
          - Hybrid query pool (init + tracking + recovery queries)
          - Tracking-priority mechanism (uncertainty-based query routing)
          - Closed-loop feedback: previous frame's output fed back as queries

    Args:
        embed_dim    : decoder embedding dimension (default 256 for RT-DETR-L)
        num_heads    : number of attention heads
        num_layers   : number of decoder transformer layers
        num_queries  : number of initial detection queries
        tau_low      : uncertainty threshold for stable state
        tau_high     : uncertainty threshold for lost state
        mlp_ratio    : FFN expansion ratio
        dropout      : dropout probability
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 num_layers: int = 6, num_queries: int = 300,
                 tau_low: float = 0.3, tau_high: float = 0.7,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim

        # Dual query set
        self.query_set = UncertaintyDualQuerySet(
            embed_dim=embed_dim, num_queries=num_queries,
            tau_low=tau_low, tau_high=tau_high,
        )

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Motion predictor: generates motion priors for recovery queries
        self.motion_predictor = nn.GRUCell(embed_dim, embed_dim)

        # Uncertainty head (applied after decoding)
        self.uncertainty_head = UncertaintyHead(embed_dim)

        # Output projection
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        memory:         torch.Tensor,              # (B, N_mem, C) from DGM-Encoder
        prev_queries:   Optional[torch.Tensor],    # (B, N_prev, C) last frame's output
        prev_motion:    Optional[torch.Tensor],    # (B, N_prev, C) last frame's motion state
        prev_uncert:    Optional[torch.Tensor],    # (B, N_prev, 1) last frame's uncertainty
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            memory       : encoder output (B, N_mem, C)
            prev_queries : previous frame's decoded features (for closed-loop feedback)
            prev_motion  : motion embedding from GRU (for recovery queries)
            prev_uncert  : per-query uncertainty from previous frame

        Returns:
            decoded      : (B, N_q, C) — decoded query features
            uncertainty  : (B, N_q, 1) — per-query uncertainty scores
            motion_state : (B, N_q, C) — updated motion embeddings (pass to next frame)
            query_types  : (B, N_q)    — 0=init, 1=tracking, 2=recovery
        """
        B = memory.size(0)
        device = memory.device

        # 1. Build query pool for this frame
        queries, query_types = self.query_set.build_query_pool(
            prev_queries=prev_queries,
            prev_motion=prev_motion,
            uncertainty=prev_uncert,
            batch_size=B,
            device=device,
        )  # (B, N_q, C)

        N_q = queries.size(1)

        # 2. Closed-loop feedback: update queries with prev_queries via cross-attention
        #    (only for tracking + recovery queries that have history)
        # This is handled implicitly since prev_queries are directly embedded in query_set.

        # 3. Transformer decoding (queries cross-attend to encoder memory)
        x = queries
        for layer in self.layers:
            x = layer(x, memory)

        # 4. Uncertainty estimation on decoded features
        uncertainty = self.uncertainty_head(x)  # (B, N_q, 1)

        # 5. Update motion state via GRU for next frame's recovery queries
        x_flat = x.reshape(B * N_q, -1)
        if prev_motion is not None and prev_motion.size(1) == N_q:
            hx_flat = prev_motion.reshape(B * N_q, -1)
        else:
            hx_flat = torch.zeros(B * N_q, self.embed_dim, device=device)
        motion_state = self.motion_predictor(x_flat, hx_flat).reshape(B, N_q, -1)

        decoded = self.out_norm(x)
        return decoded, uncertainty, motion_state, query_types
