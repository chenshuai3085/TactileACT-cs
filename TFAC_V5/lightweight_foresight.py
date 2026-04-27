"""
Lightweight Tactile Foresight: MLP-based raw marker prediction.

Predicts future marker_offset from (state, action_chunk, marker_cur).
Designed for CQF reranking pipeline — runs K=16 times per inference step.

Input:  qpos(7) + eef(6) + action_chunk(20*7=140) + marker_cur(162) = 315 dim
Output: marker_future(162) = predicted marker_offset at t+h

Architecture: 4-layer MLP with residual connections.
~400K params, <0.5ms per forward pass for K=16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightForesight(nn.Module):
    """MLP-based tactile foresight for marker_offset prediction."""

    def __init__(self, tac_dim=162, action_dim=7, chunk_size=20,
                 qpos_dim=7, eef_dim=6, hidden=512, n_layers=4, dropout=0.1):
        super().__init__()
        self.tac_dim = tac_dim

        input_dim = tac_dim + action_dim * chunk_size + qpos_dim + eef_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        layers.append(nn.LayerNorm(hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 2):
            layers.append(ResBlock(hidden, dropout))

        layers.append(nn.Linear(hidden, tac_dim))

        self.net = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, qpos, eef, action_chunk, marker_cur):
        """
        Args:
            qpos:         (B, 7)
            eef:          (B, 6)
            action_chunk: (B, chunk_size, 7) or (B, chunk_size*7)
            marker_cur:   (B, tac_dim)
        Returns:
            marker_pred: (B, tac_dim) predicted future marker_offset
        """
        B = qpos.shape[0]
        action_flat = action_chunk.reshape(B, -1)
        x = torch.cat([qpos, eef, action_flat, marker_cur], dim=-1)
        delta = self.net(x)
        return marker_cur + self.residual_scale * delta


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class ForesightLoss(nn.Module):
    """Combined MSE + delta-aware loss for foresight training."""

    def __init__(self, delta_weight=0.3):
        super().__init__()
        self.delta_weight = delta_weight

    def forward(self, pred, target, marker_cur):
        loss_recon = F.smooth_l1_loss(pred, target)

        pred_delta = pred - marker_cur
        target_delta = target - marker_cur
        loss_delta = F.smooth_l1_loss(pred_delta, target_delta)

        loss = loss_recon + self.delta_weight * loss_delta

        with torch.no_grad():
            mse = F.mse_loss(pred, target)
            delta_mse = F.mse_loss(pred_delta, target_delta)
            cosine = F.cosine_similarity(pred_delta, target_delta, dim=-1).mean()

        return loss, {
            "loss": loss.item(),
            "loss_recon": loss_recon.item(),
            "loss_delta": loss_delta.item(),
            "mse": mse.item(),
            "delta_mse": delta_mse.item(),
            "delta_cosine": cosine.item(),
        }


if __name__ == "__main__":
    model = LightweightForesight()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 16
    qpos = torch.randn(B, 7)
    eef = torch.randn(B, 6)
    action = torch.randn(B, 20, 7)
    marker_cur = torch.randn(B, 162)

    pred = model(qpos, eef, action, marker_cur)
    print(f"Output shape: {pred.shape}")

    target = torch.randn(B, 162)
    criterion = ForesightLoss()
    loss, metrics = criterion(pred, target, marker_cur)
    print(f"Loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
