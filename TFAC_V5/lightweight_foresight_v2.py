"""
Lightweight Tactile Foresight V2: FiLM-conditioned action for better sensitivity.

V1 problem: concatenating action with state → action signal gets drowned out
V2 solution: FiLM conditioning — action produces scale+bias that modulate features

Also adds action-contrastive auxiliary loss: same state, different actions
should produce detectably different predictions.

Input:  qpos(7) + eef(6) + marker_cur(162) = 175 dim (state branch)
        action_chunk(20*7=140) dim (action branch, FiLM conditioning)
Output: marker_future(162) = predicted marker_offset at t+h
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation block."""

    def __init__(self, feature_dim, cond_dim, dropout=0.1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.film = nn.Linear(cond_dim, feature_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, cond):
        h = self.main(x)
        gamma_beta = self.film(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        h = gamma * h + beta
        return self.relu(x + self.dropout(h))


class LightweightForesightV2(nn.Module):
    """FiLM-conditioned foresight with separated state and action branches."""

    def __init__(self, tac_dim=162, action_dim=7, chunk_size=20,
                 qpos_dim=7, eef_dim=6, hidden=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.tac_dim = tac_dim

        state_dim = tac_dim + qpos_dim + eef_dim
        action_flat_dim = action_dim * chunk_size

        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_flat_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.film_blocks = nn.ModuleList([
            FiLMBlock(hidden, hidden, dropout) for _ in range(n_layers - 1)
        ])

        self.head = nn.Linear(hidden, tac_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, qpos, eef, action_chunk, marker_cur):
        B = qpos.shape[0]
        action_flat = action_chunk.reshape(B, -1)

        state = torch.cat([qpos, eef, marker_cur], dim=-1)
        h = self.state_proj(state)

        action_cond = self.action_encoder(action_flat)

        for block in self.film_blocks:
            h = block(h, action_cond)

        delta = self.head(h)
        return marker_cur + self.residual_scale * delta

    def get_action_embedding(self, action_chunk):
        """Get action embedding for contrastive loss."""
        B = action_chunk.shape[0]
        return self.action_encoder(action_chunk.reshape(B, -1))


class ForesightV2Loss(nn.Module):
    """Combined reconstruction + delta-aware + action-contrastive loss."""

    def __init__(self, delta_weight=0.3, contrastive_weight=0.1, contrastive_margin=1.0):
        super().__init__()
        self.delta_weight = delta_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_margin = contrastive_margin

    def forward(self, pred, target, marker_cur, pred_perturbed=None, action_diff_norm=None):
        loss_recon = F.smooth_l1_loss(pred, target)

        pred_delta = pred - marker_cur
        target_delta = target - marker_cur
        loss_delta = F.smooth_l1_loss(pred_delta, target_delta)

        loss = loss_recon + self.delta_weight * loss_delta

        loss_contrastive = torch.tensor(0.0, device=pred.device)
        if pred_perturbed is not None and action_diff_norm is not None:
            pred_diff = (pred - pred_perturbed).norm(dim=-1)
            target_diff = action_diff_norm * self.contrastive_margin
            loss_contrastive = F.smooth_l1_loss(pred_diff, target_diff)
            loss = loss + self.contrastive_weight * loss_contrastive

        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            delta_mse = F.mse_loss(pred_delta, target_delta)
            cosine = F.cosine_similarity(pred_delta, target_delta, dim=-1).mean()

        return loss, {
            "loss": loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_delta": loss_delta.item(),
            "loss_contrastive": loss_contrastive.item(),
            "mse": mse.item(),
            "delta_mse": delta_mse.item(),
            "delta_cosine": cosine.item(),
        }


if __name__ == "__main__":
    model = LightweightForesightV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 16
    qpos = torch.randn(B, 7)
    eef = torch.randn(B, 6)
    action = torch.randn(B, 20, 7)
    marker_cur = torch.randn(B, 162)

    pred = model(qpos, eef, action, marker_cur)
    print(f"Output shape: {pred.shape}")

    action2 = action + torch.randn_like(action) * 0.5
    pred2 = model(qpos, eef, action2, marker_cur)
    diff = (pred - pred2).norm(dim=-1).mean()
    print(f"Pred diff with noisy action: {diff.item():.4f}")

    target = torch.randn(B, 162)
    action_diff = (action - action2).reshape(B, -1).norm(dim=-1)
    criterion = ForesightV2Loss()
    loss, metrics = criterion(pred, target, marker_cur, pred2, action_diff)
    print(f"Loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
