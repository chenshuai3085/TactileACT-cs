"""
ContactQualityScorer (CQF Core Scorer):
  3-branch MLP打分器，从(state, action, tactile_cur, tactile_pred)评估接触质量。

Branch 1 (触觉分析): [z_cur, z_pred, delta] → h_tac  (最重要)
Branch 2 (Action分析): action_chunk → h_act  (50% dropout防走捷径)
Branch 3 (状态上下文): qpos → h_state

融合: [h_tac, h_act, h_state] → score (标量)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContactQualityScorer(nn.Module):

    def __init__(self,
                 tac_dim=162,
                 action_dim=7,
                 chunk_size=20,
                 qpos_dim=7,
                 hidden=256,
                 action_dropout=0.5):
        super().__init__()
        self.tac_dim = tac_dim
        self.action_dropout = action_dropout

        # Branch 1: 触觉分析 — [cur, pred, delta]
        self.tac_encoder = nn.Sequential(
            nn.Linear(tac_dim * 3, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Branch 2: Action分析
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * chunk_size, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Branch 3: 状态上下文
        self.state_encoder = nn.Sequential(
            nn.Linear(qpos_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # 融合 → score
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, qpos, action_chunk, tac_cur, tac_pred):
        """
        Args:
            qpos:         (B, 7)
            action_chunk: (B, chunk_size, 7)
            tac_cur:      (B, tac_dim) — 展平的当前触觉
            tac_pred:     (B, tac_dim) — 展平的预测/GT未来触觉
        Returns:
            score: (B, 1) — raw logit, 越高越好
        """
        B = qpos.shape[0]

        # Branch 1: 触觉
        delta = tac_pred - tac_cur
        h_tac = self.tac_encoder(torch.cat([tac_cur, tac_pred, delta], dim=-1))

        # Branch 2: Action (训练时50% dropout)
        h_act = self.action_encoder(action_chunk.reshape(B, -1))
        if self.training and self.action_dropout > 0:
            mask = (torch.rand(B, 1, device=h_act.device) > self.action_dropout).float()
            h_act = h_act * mask

        # Branch 3: State
        h_state = self.state_encoder(qpos)

        # 融合打分
        h_all = torch.cat([h_tac, h_act, h_state], dim=-1)
        score = self.scorer(h_all)

        return score

    @torch.no_grad()
    def predict(self, qpos, action_chunk, tac_cur, tac_pred):
        """Inference: returns raw logit (higher = better)."""
        self.eval()
        return self.forward(qpos, action_chunk, tac_cur, tac_pred)


class CQFLoss(nn.Module):
    """All-pair adaptive-margin ranking loss for CQF training.

    Every pair (i, j) where label_i > label_j contributes:
      loss = relu(margin * (label_i - label_j) - (score_i - score_j))

    This forces the model to rank ALL quality levels, not just pos vs neg.
    """

    def __init__(self, base_margin=1.0, label_threshold=0.02):
        super().__init__()
        self.base_margin = base_margin
        self.label_threshold = label_threshold

    def forward(self, scores, labels):
        """
        Args:
            scores: (B, 1) raw logits
            labels: (B,) float, continuous quality labels
        Returns:
            loss, loss_dict
        """
        scores_flat = scores.squeeze(-1)
        B = scores_flat.shape[0]

        # All-pair pairwise ranking loss
        # (B, 1) - (1, B) → (B, B) pairwise differences
        score_diff = scores_flat.unsqueeze(1) - scores_flat.unsqueeze(0)
        label_diff = labels.unsqueeze(1) - labels.unsqueeze(0)

        # Only pairs where label_i > label_j (upper triangle of sorted order)
        valid_mask = label_diff > self.label_threshold
        n_pairs = valid_mask.sum().item()

        if n_pairs > 0:
            adaptive_margin = self.base_margin * label_diff
            pair_loss = F.relu(adaptive_margin - score_diff)
            loss_rank = pair_loss[valid_mask].mean()
        else:
            loss_rank = torch.tensor(0.0, device=scores.device)

        loss = loss_rank

        # Metrics
        with torch.no_grad():
            if n_pairs > 0:
                pairwise_correct = (score_diff[valid_mask] > 0).float().mean()
            else:
                pairwise_correct = torch.tensor(0.0)

            unique_labels = labels.unique(sorted=True)
            score_by_label = {}
            for lbl in unique_labels:
                mask = (labels - lbl).abs() < 0.01
                if mask.any():
                    score_by_label[lbl.item()] = scores_flat[mask].mean().item()

            top_label_mask = (labels - labels.max()).abs() < 0.01
            bot_label_mask = (labels - labels.min()).abs() < 0.01
            if top_label_mask.any() and bot_label_mask.any():
                spread = scores_flat[top_label_mask].mean() - scores_flat[bot_label_mask].mean()
            else:
                spread = torch.tensor(0.0)

        loss_dict = {
            "loss": loss.item(),
            "loss_rank": loss_rank.item(),
            "pairwise_acc": pairwise_correct.item(),
            "n_pairs": n_pairs,
            "score_spread": spread.item(),
            "score_mean": scores_flat.mean().item(),
            "score_std": scores_flat.std().item(),
        }

        return loss, loss_dict


if __name__ == "__main__":
    model = ContactQualityScorer(tac_dim=144)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 8
    qpos = torch.randn(B, 7)
    action = torch.randn(B, 20, 7)
    tac_cur = torch.randn(B, 144)
    tac_pred = torch.randn(B, 144)

    scores = model(qpos, action, tac_cur, tac_pred)
    print(f"Score shape: {scores.shape}")

    labels = torch.tensor([1.0, 0.85, 0.70, 0.40, 0.20, 0.10, 0.05, 0.0])
    criterion = CQFLoss()
    loss, metrics = criterion(scores, labels)
    print(f"Loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
