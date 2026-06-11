"""
Multi-step tactile latent foresight modules.

This file intentionally does not modify the existing single-step
ForesightTransformer.  The multi-step head predicts a full spatial tactile
latent sequence with shape (B, H, 144) = (B, H, 9 spatial tokens * 16 channels).
"""

import torch
import torch.nn as nn

from TFAC_V5.foresight_transformer import ForesightLayer


class MultiStepSpatialForesightTransformer(nn.Module):
    """
    Transformer world model for direct multi-step tactile latent prediction.

    Inputs are the same token representation used by the V5 single-step model:
    current vision tokens, current tactile latent spatial tokens, proprio state,
    and a future action/state chunk.  The output is a direct future tactile
    latent sequence, not an autoregressive rollout.
    """

    def __init__(self, d_model=512, action_dim=7, num_layers=3, nhead=8,
                 dim_feedforward=2048, dropout=0.1, latent_dim=16,
                 n_tactile_spatial=9, predict_horizon=16,
                 max_history=8, state_dim=7, max_action_len=30):
        super().__init__()
        if n_tactile_spatial <= 0:
            raise ValueError("n_tactile_spatial must be positive for spatial multi-step prediction")
        if predict_horizon <= 1:
            raise ValueError("MultiStepSpatialForesightTransformer expects predict_horizon > 1")

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_tactile_spatial = n_tactile_spatial
        self.predict_horizon = predict_horizon

        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_pos_embed = nn.Embedding(max_action_len, d_model)
        self.proprio_proj = nn.Linear(state_dim, d_model)
        self.temporal_pos_embed = nn.Embedding(max_history, d_model)

        self.layers = nn.ModuleList([
            ForesightLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Query one token per future time and tactile spatial cell.
        self.future_queries = nn.Parameter(
            torch.randn(predict_horizon, n_tactile_spatial, 1, d_model) * 0.02)
        self.future_time_embed = nn.Parameter(
            torch.randn(predict_horizon, 1, 1, d_model) * 0.02)
        self.future_spatial_embed = nn.Parameter(
            torch.randn(1, n_tactile_spatial, 1, d_model) * 0.02)

        self.future_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.future_cross_norm = nn.LayerNorm(d_model)

        # Causal temporal attention is applied independently for each tactile
        # spatial cell.  This preserves future order without mixing with unknown
        # later timesteps.
        self.future_temporal_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.future_temporal_norm = nn.LayerNorm(d_model)

        # Spatial attention within each future frame couples the 3x3 tactile map.
        self.future_spatial_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.future_spatial_norm = nn.LayerNorm(d_model)

        self.future_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.future_ffn_norm = nn.LayerNorm(d_model)

        self.tactile_out = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, latent_dim),
        )
        self.vision_out = nn.Linear(d_model, d_model)
        self.embed_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v_tokens, t_tokens, a1, n_v, proprio=None):
        """
        Args:
            v_tokens: (N_v, B, D) or (k, N_v, B, D)
            t_tokens: (N_t, B, D) or (k, N_t, B, D)
            a1:       (B, chunk_size, action_dim)
            n_v:      number of vision tokens per frame
            proprio:  (B, state_dim), optional

        Returns:
            t_hat_future: (B, H, N_t * latent_dim)
            v_hat_future: (B, D)
            t_embed_future: (B, H, D)
        """
        if v_tokens.dim() == 3:
            v_tokens = v_tokens.unsqueeze(0)
            t_tokens = t_tokens.unsqueeze(0)

        k = v_tokens.shape[0]
        n_tactile = t_tokens.shape[1]
        B = v_tokens.shape[2]
        D = v_tokens.shape[3]
        if n_tactile != self.n_tactile_spatial:
            raise ValueError(
                f"Expected {self.n_tactile_spatial} tactile spatial tokens, got {n_tactile}")

        a1_emb = self.action_proj(a1).permute(1, 0, 2)
        action_len = a1_emb.size(0)
        a1_emb = a1_emb + self.action_pos_embed.weight[:action_len].unsqueeze(1)

        vt = torch.cat([v_tokens, t_tokens], dim=1)
        n_vt = vt.shape[1]

        if proprio is not None:
            proprio_token = self.proprio_proj(proprio).unsqueeze(0)
            proprio_token = proprio_token.unsqueeze(0).expand(k, -1, -1, -1)
            vt = torch.cat([vt, proprio_token], dim=1)
            n_vt += 1

        temporal_pos = self.temporal_pos_embed.weight[:k].view(k, 1, 1, D)
        vt = vt + temporal_pos
        vt = vt.reshape(k * n_vt, B, D)

        for layer in self.layers:
            vt = layer(vt, a1_emb, k, n_vt)

        vt_4d = vt.view(k, n_vt, B, D)
        vt_last = vt_4d[-1]
        if n_v > 0:
            v_tokens_last = vt_last[:n_v]
            v_hat_future = self.vision_out(v_tokens_last.mean(dim=0))
        else:
            v_hat_future = torch.zeros(B, D, device=vt.device, dtype=vt.dtype)

        H = self.predict_horizon
        S = self.n_tactile_spatial
        queries = self.future_queries.expand(H, S, B, D)
        queries = queries + self.future_time_embed + self.future_spatial_embed
        q = queries.reshape(H * S, B, D)

        q2 = self.future_cross_attn(q, vt, vt)[0]
        q = self.future_cross_norm(q + q2)

        q_hs = q.view(H, S, B, D)

        q_temporal = q_hs.permute(0, 1, 2, 3).reshape(H, S * B, D)
        causal_mask = torch.triu(
            torch.ones(H, H, device=q.device), diagonal=1).bool()
        q2 = self.future_temporal_attn(
            q_temporal, q_temporal, q_temporal, attn_mask=causal_mask)[0]
        q_temporal = self.future_temporal_norm(q_temporal + q2)
        q_hs = q_temporal.view(H, S, B, D)

        q_spatial = q_hs.permute(1, 0, 2, 3).reshape(S, H * B, D)
        q2 = self.future_spatial_attn(q_spatial, q_spatial, q_spatial)[0]
        q_spatial = self.future_spatial_norm(q_spatial + q2)
        q_hs = q_spatial.view(S, H, B, D).permute(1, 0, 2, 3)

        q = q_hs.reshape(H * S, B, D)
        q2 = self.future_ffn(q)
        q = self.future_ffn_norm(q + q2)
        q_hs = q.view(H, S, B, D)

        per_token = self.tactile_out(q_hs.reshape(H * S * B, D))
        per_token = per_token.view(H, S, B, self.latent_dim)
        t_hat_future = per_token.permute(2, 0, 1, 3).reshape(
            B, H, S * self.latent_dim)

        pooled = q_hs.mean(dim=1)  # (H, B, D)
        t_embed_future = self.embed_predictor(pooled).permute(1, 0, 2)
        return t_hat_future, v_hat_future, t_embed_future
