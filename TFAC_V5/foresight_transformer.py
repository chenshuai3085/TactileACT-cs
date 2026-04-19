"""
Foresight Transformer V2: 支持时序输入 (过去 k 帧 V/T 特征)。
借鉴 VT-WM (Higuera 2026) 的 factorized attention:
  SpatialSelfAttn → TemporalSelfAttn → CrossAttn(A1) → FFN

ForesightContrastive: 不变, V̂_future 和 T̂_future 投影到低维空间做 InfoNCE。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ForesightLayer(nn.Module):
    """
    单层: SpatialSelfAttn → TemporalSelfAttn → CrossAttn(K/V=A1) → FFN

    Spatial: 同一时刻内的 V/T tokens 交互
    Temporal: 同一空间位置跨时间步交互
    Cross: 所有 tokens 与 action 交互
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
            vt:   (k * N_vt, B, D) — all V/T tokens across k timesteps
            a1:   (S_a, B, D)      — action tokens (projected A1)
            k:    int               — number of history frames
            n_vt: int               — N_v + N_t per frame
        Returns:
            vt: (k * N_vt, B, D)
        """
        B, D = vt.shape[1], vt.shape[2]

        if k == 1:
            # No temporal dimension — fall back to simple self-attention
            vt2 = self.spatial_attn(vt, vt, vt)[0]
            vt = vt + self.dropout1(vt2)
            vt = self.norm1(vt)

            # Skip temporal (identity) — 不再 apply norm2, 避免 double LayerNorm
        else:
            # 1. Spatial self-attention: within each timestep
            # Reshape: (k * N_vt, B, D) → (k, N_vt, B, D)
            vt_4d = vt.view(k, n_vt, B, D)
            # Process each timestep: merge k into batch → (N_vt, k*B, D)
            vt_spatial = vt_4d.permute(1, 0, 2, 3).reshape(n_vt, k * B, D)
            vt2 = self.spatial_attn(vt_spatial, vt_spatial, vt_spatial)[0]
            vt_spatial = vt_spatial + self.dropout1(vt2)
            vt_spatial = self.norm1(vt_spatial)
            # Reshape back: (N_vt, k*B, D) → (k, N_vt, B, D)
            vt_4d = vt_spatial.view(n_vt, k, B, D).permute(1, 0, 2, 3)

            # 2. Temporal self-attention: across timesteps for each spatial position
            # (k, N_vt, B, D) → merge N_vt into batch → (k, N_vt*B, D)
            vt_temporal = vt_4d.reshape(k, n_vt * B, D)
            vt2 = self.temporal_attn(vt_temporal, vt_temporal, vt_temporal)[0]
            vt_temporal = vt_temporal + self.dropout2(vt2)
            vt_temporal = self.norm2(vt_temporal)
            # Reshape back: (k, N_vt*B, D) → (k * N_vt, B, D)
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


class SpatialTactileDecoder(nn.Module):
    """
    ConvTranspose2d 上采样解码器: (B, D) → (B, 162)
    从 pooled feature 重建 9×9×2 的 marker_offset 空间场。

    num_layers=2 (旧版):
        Linear(D, 128*3*3) → ConvT(128,64, 3→5) → ConvT(64,2, 5→9)
    num_layers=3 (新版):
        Linear(D, 256*3*3) → ConvT(256,128, 3→5) → ConvT(128,64, 5→7) → ConvT(64,2, 7→9)
    """

    def __init__(self, d_model: int = 512, num_layers: int = 3):
        super().__init__()
        if num_layers == 2:
            self.fc = nn.Linear(d_model, 128 * 3 * 3)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),  # 3→5
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=0),  # 5→9
            )
            self._fc_channels = 128
        else:
            self.fc = nn.Linear(d_model, 256 * 3 * 3)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=0),  # 3→5
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),   # 5→7
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 2, kernel_size=3, stride=1, padding=0),     # 7→9
            )
            self._fc_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B, 162)"""
        B = x.size(0)
        h = F.relu(self.fc(x))
        h = h.view(B, self._fc_channels, 3, 3)
        h = self.deconv(h)                  # (B, 2, 9, 9)
        return h.permute(0, 2, 3, 1).reshape(B, -1)  # (B, 9, 9, 2) → (B, 162)


class ForesightTransformer(nn.Module):
    """
    V3: 支持时序输入 + 多帧预测的前瞻 Transformer。

    输入: V_hist, T_hist (过去 k 帧 backbone 输出), A1 (draft action chunk)
    输出: T̂_future(B, H, tactile_out_dim), V̂_future(B, D)

    predict_horizon=1 时退化为 V2 单帧行为。
    k=1 时退化为 V1 行为。
    """

    def __init__(self, d_model: int = 512, action_dim: int = 7,
                 num_layers: int = 2, nhead: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 tactile_out_dim: int = None,
                 tactile_decoder_type: str = "linear",
                 spatial_tac_dec_layers: int = 3,
                 max_history: int = 8,
                 predict_horizon: int = 1,
                 state_dim: int = 7):
        super().__init__()
        self.d_model = d_model
        self.tactile_out_dim = tactile_out_dim if tactile_out_dim is not None else d_model
        self.predict_horizon = predict_horizon

        # 将 action chunk 投影到 d_model
        self.action_proj = nn.Linear(action_dim, d_model)

        # 将本体状态投影为 1 个 token
        self.proprio_proj = nn.Linear(state_dim, d_model)

        # Temporal position embedding (learnable)
        self.temporal_pos_embed = nn.Embedding(max_history, d_model)

        # Foresight layers (with spatial + temporal attention)
        self.layers = nn.ModuleList([
            ForesightLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 多帧预测: future query tokens + cross-attention + causal self-attention
        if predict_horizon > 1:
            self.future_queries = nn.Parameter(torch.randn(predict_horizon, 1, d_model) * 0.02)
            self.future_cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.future_norm = nn.LayerNorm(d_model)
            # P0: causal self-attention among future queries
            self.future_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.future_self_norm = nn.LayerNorm(d_model)

        # P1: embed_predictor — 直接输出 embedding (给 contrastive + fusion)
        self.embed_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 输出投影
        if tactile_decoder_type == "spatial" and self.tactile_out_dim == 9 * 9 * 2:
            self.tactile_out = SpatialTactileDecoder(d_model, num_layers=spatial_tac_dec_layers)
        else:
            self.tactile_out = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, self.tactile_out_dim),
            )
        self.vision_out = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v_tokens: torch.Tensor, t_tokens: torch.Tensor,
                a1: torch.Tensor, n_v: int,
                proprio: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_tokens: (k, N_v, B, D) or (N_v, B, D) — vision tokens (k frames or single)
            t_tokens: (k, N_t, B, D) or (N_t, B, D) — tactile tokens
            a1:       (B, chunk_size, action_dim) — draft action (detached)
            n_v:      int — number of vision tokens per frame
            proprio:  (B, state_dim) — robot proprioceptive state (optional)
        Returns:
            t_hat_future: (B, H, tactile_out_dim) if predict_horizon>1, else (B, tactile_out_dim)
            v_hat_future: (B, D)
        """
        # Handle single-frame input (backward compatible)
        if v_tokens.dim() == 3:
            v_tokens = v_tokens.unsqueeze(0)  # (1, N_v, B, D)
            t_tokens = t_tokens.unsqueeze(0)  # (1, N_t, B, D)

        k = v_tokens.shape[0]
        N_v = v_tokens.shape[1]
        N_t = t_tokens.shape[1]
        B = v_tokens.shape[2]
        D = v_tokens.shape[3]
        n_vt = N_v + N_t

        # Project action to d_model: (chunk_size, B, D)
        a1_emb = self.action_proj(a1).permute(1, 0, 2)

        # Concat V and T per timestep: (k, N_vt, B, D)
        vt = torch.cat([v_tokens, t_tokens], dim=1)  # (k, N_vt, B, D)

        # 将本体状态作为额外 token concat 到每帧: [V; T; P]
        if proprio is not None:
            proprio_token = self.proprio_proj(proprio).unsqueeze(0)  # (1, B, D)
            proprio_expanded = proprio_token.unsqueeze(0).expand(k, -1, -1, -1)  # (k, 1, B, D)
            vt = torch.cat([vt, proprio_expanded], dim=1)  # (k, N_vt+1, B, D)
            n_vt = n_vt + 1

        # Add temporal position embedding
        # temporal_pos: (k, 1, 1, D) → broadcast to (k, N_vt, B, D)
        temporal_pos = self.temporal_pos_embed.weight[:k].view(k, 1, 1, D)
        vt = vt + temporal_pos

        # Flatten to (k * N_vt, B, D) for layer processing
        vt = vt.reshape(k * n_vt, B, D)

        # Pass through foresight layers
        for layer in self.layers:
            vt = layer(vt, a1_emb, k, n_vt)

        # --- Vision output (always single-frame: last timestep) ---
        vt_4d = vt.view(k, n_vt, B, D)
        vt_last = vt_4d[-1]  # (N_vt, B, D)
        v_out = vt_last[:N_v]  # (N_v, B, D)
        v_pooled = v_out.mean(dim=0)
        v_hat_future = self.vision_out(v_pooled)  # (B, D)

        # --- Tactile output ---
        H = self.predict_horizon
        if H > 1:
            # Multi-frame: use future queries cross-attending to vt features
            queries = self.future_queries.expand(H, B, D)  # (H, B, D)
            q_out = self.future_cross_attn(queries, vt, vt)[0]  # (H, B, D)
            q_out = self.future_norm(queries + q_out)  # (H, B, D)

            # P0: causal self-attention among future queries
            # t+k can attend to t+1..t+k-1, but not t+k+1..t+H
            causal_mask = torch.triu(
                torch.ones(H, H, device=q_out.device), diagonal=1).bool()
            q_out2 = self.future_self_attn(
                q_out, q_out, q_out, attn_mask=causal_mask)[0]
            q_out = self.future_self_norm(q_out + q_out2)  # (H, B, D)

            # Raw marker output (保留, 给 foresight_tac 像素级 loss)
            # Batch 所有帧一次性通过 tactile_out, 避免 H 次串行调用
            q_flat = q_out.reshape(H * B, D)           # (H*B, D)
            t_flat = self.tactile_out(q_flat)           # (H*B, 162)
            t_hat_future = t_flat.view(H, B, -1).permute(1, 0, 2)  # (B, H, 162)

            # P1: embedding output (给 contrastive + fusion, 无需 roundtrip)
            t_embed_future = self.embed_predictor(q_out)  # (H, B, D)
            t_embed_future = t_embed_future.permute(1, 0, 2)  # (B, H, D)
        else:
            # Single-frame (backward compatible, ignore proprio token at the end)
            t_out = vt_last[N_v:N_v + N_t]  # (N_t, B, D)
            t_pooled = t_out.mean(dim=0)
            t_hat_future = self.tactile_out(t_pooled)  # (B, 162)
            t_embed_future = self.embed_predictor(
                t_pooled.unsqueeze(0)).squeeze(0)  # (B, D)

        return t_hat_future, v_hat_future, t_embed_future


class ForesightContrastive(nn.Module):
    """
    在前瞻输出空间做 V-T 跨模态 InfoNCE 对比学习。
    各自独立的 projection head 投影到低维空间。
    """

    def __init__(self, feat_dim: int = 512, proj_dim: int = 128,
                 temperature: float = 0.07):
        super().__init__()
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
        # 可学习的 temperature (log-scale)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, v_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_hat: (B, feat_dim) — predicted future vision feature
            t_hat: (B, feat_dim) — predicted future tactile feature
        Returns:
            loss: scalar — symmetric InfoNCE loss
        """
        v = F.normalize(self.v_proj(v_hat), dim=-1)  # (B, proj_dim)
        t = F.normalize(self.t_proj(t_hat), dim=-1)  # (B, proj_dim)

        temp = self.log_temp.exp()
        logits = temp * v @ t.T  # (B, B)

        labels = torch.arange(v.size(0), device=logits.device)
        loss = (F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)) / 2.0

        return loss
