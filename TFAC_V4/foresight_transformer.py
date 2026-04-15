"""
V4 Foresight Transformer — Innovations C1 (spatial tokens) + C2 (latent prediction).

Key changes from V3:
  1. Input: 9 spatial tactile tokens (not 1 global), more vision-tactile interaction
  2. Output: z_hat_latent (B, H, 9, D) in latent space + t_hat_obs (B, H, 9, 9, 2) decoded
  3. PatchLevelDecoder: (B, 9, D) → per-patch MLP → reassemble → (B, 9, 9, 2)
  4. ForesightContrastiveV4: spatial + temporal + GT 3-level contrastive

ForesightLayer from V3 is reused (works with any n_vt count).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ForesightLayer(nn.Module):
    """
    Single foresight layer (from V3, unchanged interface).
    SpatialSelfAttn -> TemporalSelfAttn -> CrossAttn(A1) -> FFN

    Works with any n_vt (V3: N_v+1, V4: N_v+9+1).
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, vt: torch.Tensor, a1: torch.Tensor,
                k: int, n_vt: int) -> torch.Tensor:
        """
        Args:
            vt:   (k * N_vt, B, D)
            a1:   (S_a, B, D)
            k:    int — number of history frames
            n_vt: int — N_v + N_t per frame
        Returns:
            vt: (k * N_vt, B, D)
        """
        B, D = vt.shape[1], vt.shape[2]

        if k == 1:
            vt2 = self.spatial_attn(vt, vt, vt)[0]
            vt = vt + self.dropout1(vt2)
            vt = self.norm1(vt)
        else:
            # 1. Spatial self-attention: within each timestep
            vt_4d = vt.view(k, n_vt, B, D)
            vt_spatial = vt_4d.permute(1, 0, 2, 3).reshape(n_vt, k * B, D)
            vt2 = self.spatial_attn(vt_spatial, vt_spatial, vt_spatial)[0]
            vt_spatial = vt_spatial + self.dropout1(vt2)
            vt_spatial = self.norm1(vt_spatial)
            vt_4d = vt_spatial.view(n_vt, k, B, D).permute(1, 0, 2, 3)

            # 2. Temporal self-attention: across timesteps for each spatial position
            vt_temporal = vt_4d.reshape(k, n_vt * B, D)
            vt2 = self.temporal_attn(vt_temporal, vt_temporal, vt_temporal)[0]
            vt_temporal = vt_temporal + self.dropout2(vt2)
            vt_temporal = self.norm2(vt_temporal)
            vt = vt_temporal.view(k, n_vt, B, D).reshape(k * n_vt, B, D)

        # 3. Cross-Attention: Q=all [V;T], K/V=A1
        vt2 = self.cross_attn(vt, a1, a1)[0]
        vt = vt + self.dropout3(vt2)
        vt = self.norm3(vt)

        # 4. FFN
        vt2 = self.ffn(vt)
        vt = vt + self.dropout4(vt2)
        vt = self.norm4(vt)

        return vt


class PatchLevelDecoder(nn.Module):
    """
    Decode 9 patch tokens back to (B, 9, 9, 2) marker displacement.

    Each patch token (D-dim) generates a 3×3×2 = 18-dim output via MLP,
    then patches are reassembled into the full 9×9×2 grid.

    This is much more faithful than V3's SpatialTactileDecoder which tried
    to reconstruct the entire 9×9 from a single global vector.
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.d_model = d_model
        # Per-patch MLP: (D) → (18) = 3×3×2
        self.patch_mlp = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 18),  # 3×3×2
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, 9, D) — 9 spatial patch tokens
        Returns:
            marker_offset: (B, 9, 9, 2)
        """
        B = patch_tokens.shape[0]
        # Per-patch decode
        patch_out = self.patch_mlp(patch_tokens)  # (B, 9, 18)
        patch_out = patch_out.view(B, 9, 3, 3, 2)  # (B, 9, 3, 3, 2)

        # Reassemble 3×3 patches into 9×9 grid
        # patch layout: row-major 3×3 grid of patches
        output = torch.zeros(B, 9, 9, 2, device=patch_tokens.device, dtype=patch_tokens.dtype)
        for pid in range(9):
            pr = pid // 3  # patch row (0,1,2)
            pc = pid % 3   # patch col (0,1,2)
            r_start = pr * 3
            c_start = pc * 3
            output[:, r_start:r_start+3, c_start:c_start+3, :] = patch_out[:, pid]

        return output


class ForesightTransformerV4(nn.Module):
    """
    V4 Foresight Transformer with spatial token support.

    Key differences from V3:
      - Input tactile: 9 spatial tokens (not 1 global)
      - Dual output: latent z_hat (B,H,9,D) + observation t_hat (B,H,9,9,2)
      - PatchLevelDecoder instead of SpatialTactileDecoder
      - embed_predictor operates on all 9 patch tokens

    predict_horizon=1: single frame prediction (backward compatible)
    predict_horizon>1: multi-frame with causal future queries
    """

    def __init__(self, d_model: int = 512, action_dim: int = 7,
                 num_layers: int = 2, nhead: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_history: int = 8,
                 predict_horizon: int = 1,
                 state_dim: int = 7,
                 n_tac_tokens: int = 9):
        super().__init__()
        self.d_model = d_model
        self.predict_horizon = predict_horizon
        self.n_tac_tokens = n_tac_tokens

        # Action projection
        self.action_proj = nn.Linear(action_dim, d_model)

        # Proprio projection (robot state → 1 token)
        self.proprio_proj = nn.Linear(state_dim, d_model)

        # Temporal position embedding
        self.temporal_pos_embed = nn.Embedding(max_history, d_model)

        # Foresight layers
        self.layers = nn.ModuleList([
            ForesightLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Multi-frame: future query tokens + cross-attention + causal self-attention
        if predict_horizon > 1:
            # One set of queries per future frame, with n_tac_tokens per frame
            self.future_queries = nn.Parameter(
                torch.randn(predict_horizon, n_tac_tokens, 1, d_model) * 0.02)
            self.future_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.future_norm = nn.LayerNorm(d_model)
            self.future_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.future_self_norm = nn.LayerNorm(d_model)

        # Embed predictor: latent space output for contrastive + fusion
        # Per-token predictor: (D) → (D) for each of the 9 tokens
        self.embed_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Patch-level observation decoder: 9 tokens → (9, 9, 2)
        self.tactile_decoder = PatchLevelDecoder(d_model)

        # Vision output projection
        self.vision_out = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v_tokens: torch.Tensor, t_tokens: torch.Tensor,
                a1: torch.Tensor, n_v: int,
                proprio: torch.Tensor = None) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            v_tokens: (k, N_v, B, D) or (N_v, B, D)
            t_tokens: (k, N_t, B, D) or (N_t, B, D) — N_t=9 for spatial encoder
            a1:       (B, chunk_size, action_dim)
            n_v:      int — number of vision tokens per frame
            proprio:  (B, state_dim) optional
        Returns:
            t_hat_obs:      (B, H, 9, 9, 2) or (B, 9, 9, 2) — decoded observation
            v_hat_future:   (B, D)
            t_embed_future: (B, H, 9, D) or (B, 9, D) — latent embeddings per patch
        """
        # Handle single-frame input
        if v_tokens.dim() == 3:
            v_tokens = v_tokens.unsqueeze(0)
            t_tokens = t_tokens.unsqueeze(0)

        k = v_tokens.shape[0]
        N_v = v_tokens.shape[1]
        N_t = t_tokens.shape[1]
        B = v_tokens.shape[2]
        D = v_tokens.shape[3]
        n_vt = N_v + N_t

        # Project action
        a1_emb = self.action_proj(a1).permute(1, 0, 2)  # (chunk_size, B, D)

        # Concat V and T per timestep
        vt = torch.cat([v_tokens, t_tokens], dim=1)  # (k, N_vt, B, D)

        # Add proprio token
        if proprio is not None:
            proprio_token = self.proprio_proj(proprio).unsqueeze(0)  # (1, B, D)
            proprio_expanded = proprio_token.unsqueeze(0).expand(k, -1, -1, -1)  # (k, 1, B, D)
            vt = torch.cat([vt, proprio_expanded], dim=1)
            n_vt = n_vt + 1

        # Add temporal position embedding
        temporal_pos = self.temporal_pos_embed.weight[:k].view(k, 1, 1, D)
        vt = vt + temporal_pos

        # Flatten
        vt = vt.reshape(k * n_vt, B, D)

        # Foresight layers
        for layer in self.layers:
            vt = layer(vt, a1_emb, k, n_vt)

        # --- Vision output (last timestep, vision tokens only) ---
        vt_4d = vt.view(k, n_vt, B, D)
        vt_last = vt_4d[-1]  # (N_vt, B, D)
        v_out = vt_last[:N_v]  # (N_v, B, D)
        v_pooled = v_out.mean(dim=0)
        v_hat_future = self.vision_out(v_pooled)  # (B, D)

        # --- Tactile output ---
        H = self.predict_horizon
        if H > 1:
            # Multi-frame: future queries (H*N_t, B, D)
            queries = self.future_queries.expand(H, N_t, B, D)  # (H, N_t, B, D)
            queries_flat = queries.reshape(H * N_t, B, D)

            # Cross-attend to all vt features
            q_out = self.future_cross_attn(queries_flat, vt, vt)[0]
            q_out = self.future_norm(queries_flat + q_out)

            # Causal self-attention: frame h can attend to frames 1..h only
            # Build block-causal mask: (H*N_t, H*N_t) — tokens within same or earlier frame
            total_q = H * N_t
            causal_mask = torch.ones(total_q, total_q, device=q_out.device, dtype=torch.bool)
            for h in range(H):
                start = h * N_t
                end = (h + 1) * N_t
                # frame h can attend to frames 0..h
                causal_mask[start:end, :end] = False
            # True = blocked
            q_out2 = self.future_self_attn(q_out, q_out, q_out, attn_mask=causal_mask)[0]
            q_out = self.future_self_norm(q_out + q_out2)

            # Reshape: (H*N_t, B, D) → (H, N_t, B, D) → (B, H, N_t, D)
            q_out = q_out.view(H, N_t, B, D).permute(2, 0, 1, 3)  # (B, H, N_t, D)

            # Latent embeddings: per-patch
            t_embed_future = self.embed_predictor(q_out)  # (B, H, N_t, D)

            # Observation decode: per frame
            B_cur = q_out.shape[0]
            q_flat = q_out.reshape(B_cur * H, N_t, D)  # (B*H, N_t, D)
            t_hat_obs = self.tactile_decoder(q_flat)  # (B*H, 9, 9, 2)
            t_hat_obs = t_hat_obs.view(B_cur, H, 9, 9, 2)
        else:
            # Single-frame: use last timestep tactile tokens
            t_out = vt_last[N_v:N_v + N_t]  # (N_t, B, D)
            t_out = t_out.permute(1, 0, 2)  # (B, N_t, D)

            # Latent embeddings
            t_embed_future = self.embed_predictor(t_out)  # (B, N_t, D)

            # Observation decode
            t_hat_obs = self.tactile_decoder(t_out)  # (B, 9, 9, 2)

        return t_hat_obs, v_hat_future, t_embed_future


class ForesightContrastiveV4(nn.Module):
    """
    V4 Multi-level Contrastive Learning (Innovation C5).

    Level 1 — Spatial: per-patch contrastive (9 pairs per sample, fine-grained alignment)
    Level 2 — Temporal: adjacent-frame similarity (smooth trajectory constraint)
    Level 3 — GT: GT tactile vs GT vision alignment (no prediction noise)

    All levels use InfoNCE with learnable temperature.
    """

    def __init__(self, feat_dim: int = 512, proj_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
        # Shared projection heads (V and T project independently)
        self.v_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        self.t_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        # Learnable temperature
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward_spatial(self, v_hat: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Spatial contrastive: per-patch V-T alignment.

        Args:
            v_hat: (B, D) — predicted future vision (global)
            t_embed: (B, 9, D) — predicted future tactile embeddings (per patch)
        Returns:
            loss: scalar
        """
        B, N_t, D = t_embed.shape
        temp = self.log_temp.exp()

        # Project vision: expand to match patch count
        v_proj = F.normalize(self.v_proj(v_hat), dim=-1)  # (B, proj_dim)

        # Project each tactile patch
        t_flat = t_embed.reshape(B * N_t, D)
        t_proj = F.normalize(self.t_proj(t_flat), dim=-1)  # (B*9, proj_dim)
        t_proj = t_proj.view(B, N_t, -1)  # (B, 9, proj_dim)

        # Spatial InfoNCE: for each patch, treat same-sample vision as positive
        # Logits: (B, 9, B) — each patch has B potential vision matches
        logits = temp * torch.einsum('bnd,cd->bnc', t_proj, v_proj)  # (B, 9, B)

        labels = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, N_t)  # (B, 9)
        logits_flat = logits.reshape(B * N_t, B)
        labels_flat = labels.reshape(B * N_t)

        loss = F.cross_entropy(logits_flat, labels_flat)
        return loss

    def forward_global(self, v_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        """
        Standard global InfoNCE (V3 compatible, used for GT contrastive).

        Args:
            v_hat: (B, D) — vision feature
            t_hat: (B, D) — tactile feature
        Returns:
            loss: scalar
        """
        v = F.normalize(self.v_proj(v_hat), dim=-1)
        t = F.normalize(self.t_proj(t_hat), dim=-1)

        temp = self.log_temp.exp()
        logits = temp * v @ t.T

        labels = torch.arange(v.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)) / 2.0
        return loss

    def forward_temporal(self, t_embed_seq: torch.Tensor) -> torch.Tensor:
        """
        Temporal contrastive: encourage smooth trajectory in latent space.
        Adjacent frames should be more similar than distant frames.

        Args:
            t_embed_seq: (B, H, D) — tactile embeddings across H future frames
        Returns:
            loss: scalar — margin ranking loss encouraging t[h] closer to t[h+1] than t[h+2]
        """
        B, H, D = t_embed_seq.shape
        if H < 3:
            return torch.tensor(0.0, device=t_embed_seq.device)

        t_norm = F.normalize(t_embed_seq, dim=-1)  # (B, H, D)

        losses = []
        for h in range(H - 2):
            # sim(h, h+1) should be > sim(h, h+2) by margin
            sim_close = (t_norm[:, h] * t_norm[:, h + 1]).sum(dim=-1)   # (B,)
            sim_far = (t_norm[:, h] * t_norm[:, h + 2]).sum(dim=-1)     # (B,)
            # margin ranking: want sim_close > sim_far by margin 0.1
            loss_h = F.relu(0.1 - (sim_close - sim_far)).mean()
            losses.append(loss_h)

        return torch.stack(losses).mean()
