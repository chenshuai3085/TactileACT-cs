"""Loss functions for board chunk energy training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def expert_margin_loss(score_good: torch.Tensor, labels: torch.Tensor, margin: float = 0.25) -> torch.Tensor:
    pos = score_good[labels == 0]
    neg = score_good[labels != 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return score_good.new_tensor(0.0)
    return F.relu(margin - pos.mean() + neg).mean()


def supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
) -> torch.Tensor:
    """Supervised contrastive loss over fused chunk embeddings."""

    if embedding.shape[0] <= 1:
        return embedding.new_tensor(0.0)
    z = F.normalize(embedding, dim=-1)
    logits = z @ z.T / max(temperature, 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    same = labels.view(-1, 1).eq(labels.view(1, -1)) & ~eye
    exp_logits = torch.exp(logits) * (~eye).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    denom = same.float().sum(dim=1)
    valid = denom > 0
    if not torch.any(valid):
        return embedding.new_tensor(0.0)
    mean_log_prob_pos = (same.float() * log_prob).sum(dim=1)[valid] / denom[valid]
    return -mean_log_prob_pos.mean()
