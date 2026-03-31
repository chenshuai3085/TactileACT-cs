"""
Foresight Transformer: 给定当前 V/T 特征和 draft action A1，预测未来触觉/视觉特征。
ForesightContrastive: 预测的 V̂_future 和 T̂_future 投影到低维空间做 InfoNCE。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ForesightLayer(nn.Module):
    """单层: SelfAttn([V;T]) → CrossAttn(Q=[V;T], K/V=A1) → FFN"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, vt: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vt: (S_vt, B, D) — concat of V and T tokens
            a1: (S_a, B, D)  — action tokens (projected A1)
        Returns:
            vt: (S_vt, B, D)
        """
        # 1. Self-Attention on [V;T]
        vt2 = self.self_attn(vt, vt, vt)[0]
        vt = vt + self.dropout1(vt2)
        vt = self.norm1(vt)

        # 2. Cross-Attention: Q=[V;T], K/V=A1
        vt2 = self.cross_attn(vt, a1, a1)[0]
        vt = vt + self.dropout2(vt2)
        vt = self.norm2(vt)

        # 3. FFN
        vt2 = self.ffn(vt)
        vt = vt + self.dropout3(vt2)
        vt = self.norm3(vt)

        return vt


class SpatialTactileDecoder(nn.Module):
    """
    ConvTranspose2d 上采样解码器: (B, D) → (B, 162)
    从 pooled feature 重建 9×9×2 的 marker_offset 空间场。

    Architecture:
        Linear(D, 128*3*3) → ReLU → reshape (B, 128, 3, 3)
        → ConvTranspose2d(128, 64, k=3, s=1, p=0) + ReLU → (B, 64, 5, 5)
        → ConvTranspose2d(64, 2, k=3, s=2, p=1, output_padding=0) → (B, 2, 9, 9)
        → permute → flatten → (B, 162)
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.fc = nn.Linear(d_model, 128 * 3 * 3)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=0),  # 3→5
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 2, kernel_size=3, stride=2, padding=1, output_padding=0),  # 5→9
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D) → (B, 162)"""
        B = x.size(0)
        h = F.relu(self.fc(x))             # (B, 128*3*3)
        h = h.view(B, 128, 3, 3)           # (B, 128, 3, 3)
        h = self.deconv(h)                  # (B, 2, 9, 9)
        return h.permute(0, 2, 3, 1).reshape(B, -1)  # (B, 9, 9, 2) → (B, 162)


class ForesightTransformer(nn.Module):
    """
    输入: V_feat, T_feat (backbone 输出, 已 flatten), A1 (draft action chunk)
    输出: T̂_future(B, tactile_out_dim), V̂_future(B, D)

    tactile_out_dim:
      - "image" mode: d_model (512), 预测 embedding
      - "marker" mode: 9*9*2 = 162, 预测 raw marker_offset
    """

    def __init__(self, d_model: int = 512, action_dim: int = 7,
                 num_layers: int = 2, nhead: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 tactile_out_dim: int = None,
                 tactile_decoder_type: str = "linear"):
        super().__init__()
        self.d_model = d_model
        self.tactile_out_dim = tactile_out_dim if tactile_out_dim is not None else d_model

        # 将 action chunk 投影到 d_model
        self.action_proj = nn.Linear(action_dim, d_model)

        # Foresight layers
        self.layers = nn.ModuleList([
            ForesightLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # 输出投影
        # tactile: d_model → tactile_out_dim (162 for marker, d_model for image)
        if tactile_decoder_type == "spatial" and self.tactile_out_dim == 9 * 9 * 2:
            self.tactile_out = SpatialTactileDecoder(d_model)
        else:
            self.tactile_out = nn.Linear(d_model, self.tactile_out_dim)
        # vision: d_model → d_model (always embedding space)
        self.vision_out = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v_tokens: torch.Tensor, t_tokens: torch.Tensor,
                a1: torch.Tensor, n_v: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_tokens: (N_v, B, D) — vision spatial tokens from backbone
            t_tokens: (N_t, B, D) — tactile spatial tokens from backbone
            a1:       (B, chunk_size, action_dim) — draft action (should be detached)
            n_v:      int — number of vision tokens, for splitting output
        Returns:
            t_hat_future: (B, tactile_out_dim) — predicted future tactile
                          image mode: (B, D) embedding; marker mode: (B, 162) raw offset
            v_hat_future: (B, D) — predicted future vision feature
        """
        # Project action to d_model and permute to (chunk_size, B, D)
        a1_emb = self.action_proj(a1).permute(1, 0, 2)  # (chunk_size, B, D)

        # Concat V and T tokens: (N_v + N_t, B, D)
        vt = torch.cat([v_tokens, t_tokens], dim=0)

        # Pass through foresight layers
        for layer in self.layers:
            vt = layer(vt, a1_emb)

        # Split back to V and T
        v_out = vt[:n_v]   # (N_v, B, D)
        t_out = vt[n_v:]   # (N_t, B, D)

        # Mean-pool over spatial dimension → (B, D)
        v_pooled = v_out.mean(dim=0)
        t_pooled = t_out.mean(dim=0)

        # Output projection
        v_hat_future = self.vision_out(v_pooled)   # (B, D)
        t_hat_future = self.tactile_out(t_pooled)  # (B, D)

        return t_hat_future, v_hat_future


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
