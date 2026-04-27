"""
ContactQualityScorer (CQF Core Scorer):
  3-branch MLP打分器，从(state, action, tactile_cur, tactile_pred)评估接触质量。

Branch 1 (触觉分析): [z_cur, z_pred, delta] → h_tac  (最重要)
Branch 2 (Action分析): action_chunk → h_act  (50% dropout防走捷径)
Branch 3 (状态上下文): [qpos, eef] → h_state

融合: [h_tac, h_act, h_state] → score (标量)

支持两种触觉输入:
  - raw marker: (9, 9, 2) = 162维
  - TactileVAE latent: 72维 (C=8, 3×3)
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
                 eef_dim=6,
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
            nn.Linear(qpos_dim + eef_dim, hidden),
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

    def forward(self, qpos, eef, action_chunk, tac_cur, tac_pred):
        """
        Args:
            qpos:         (B, 7)
            eef:          (B, 6)
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
        h_state = self.state_encoder(torch.cat([qpos, eef], dim=-1))

        # 融合打分
        h_all = torch.cat([h_tac, h_act, h_state], dim=-1)
        score = self.scorer(h_all)

        return score


class CQFLoss(nn.Module):
    """Ranking Loss + BCE Loss for CQF training."""

    def __init__(self, margin=0.5, bce_weight=0.5, hard_k=5):
        super().__init__()
        self.margin = margin
        self.bce_weight = bce_weight
        self.hard_k = hard_k

    def forward(self, scores, labels):
        """
        Args:
            scores: (B, 1) raw logits
            labels: (B,) float, 1.0=positive, 0.0=negative
        Returns:
            loss, loss_dict
        """
        scores_flat = scores.squeeze(-1)

        # BCE loss
        loss_bce = F.binary_cross_entropy_with_logits(scores_flat, labels)

        # Hard pair ranking loss
        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5

        pos_scores = scores_flat[pos_mask]
        neg_scores = scores_flat[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            loss_rank = torch.tensor(0.0, device=scores.device)
        else:
            # 每个正样本 vs top-K最高分负样本 (hard negatives)
            k = min(self.hard_k, len(neg_scores))
            hard_neg_scores, _ = neg_scores.topk(k)

            # 每个负样本 vs bottom-K最低分正样本 (hard positives)
            k_pos = min(self.hard_k, len(pos_scores))
            hard_pos_scores, _ = pos_scores.topk(k_pos, largest=False)

            # Pairwise margin ranking: score_pos - score_neg > margin
            # 用broadcast: (N_pos, 1) - (1, K_neg)
            diff1 = pos_scores.unsqueeze(-1) - hard_neg_scores.unsqueeze(0)
            loss_rank_1 = F.relu(self.margin - diff1).mean()

            diff2 = hard_pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(0)
            loss_rank_2 = F.relu(self.margin - diff2).mean()

            loss_rank = (loss_rank_1 + loss_rank_2) / 2

        loss = loss_rank + self.bce_weight * loss_bce

        # Metrics
        with torch.no_grad():
            pred_labels = (scores_flat > 0).float()
            accuracy = (pred_labels == labels).float().mean()

            if len(pos_scores) > 0 and len(neg_scores) > 0:
                ranking_acc = (pos_scores.mean() > neg_scores.mean()).float()
                score_spread = pos_scores.mean() - neg_scores.mean()
            else:
                ranking_acc = torch.tensor(0.0)
                score_spread = torch.tensor(0.0)

        loss_dict = {
            "loss": loss.item(),
            "loss_rank": loss_rank.item(),
            "loss_bce": loss_bce.item(),
            "accuracy": accuracy.item(),
            "ranking_acc": ranking_acc.item(),
            "score_spread": score_spread.item(),
            "pos_score_mean": pos_scores.mean().item() if len(pos_scores) > 0 else 0,
            "neg_score_mean": neg_scores.mean().item() if len(neg_scores) > 0 else 0,
        }

        return loss, loss_dict


if __name__ == "__main__":
    model = ContactQualityScorer(tac_dim=162)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    B = 8
    qpos = torch.randn(B, 7)
    eef = torch.randn(B, 6)
    action = torch.randn(B, 20, 7)
    tac_cur = torch.randn(B, 162)
    tac_pred = torch.randn(B, 162)

    scores = model(qpos, eef, action, tac_cur, tac_pred)
    print(f"Score shape: {scores.shape}")

    labels = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    criterion = CQFLoss()
    loss, metrics = criterion(scores, labels)
    print(f"Loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
