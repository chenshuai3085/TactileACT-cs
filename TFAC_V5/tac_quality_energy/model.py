"""Task-conditioned multi-head TacQualityEnergy model.

The model is a differentiable scorer for tactile consequence quality.  It is
not a rollout policy: it maps predicted/observed tactile-action proxy features
to several heads, then fuses those heads into a scalar energy that can be used
for classifier/scorer guidance.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


TASK_TO_ID = {"insertion": 0, "board": 1}


class DistilledTacQualityEnergy(nn.Module):
    """Shared encoder with five parallel quality heads.

    Inputs are normalized proxy features plus a task id.  The five heads are:
    binary good/bad, reason class, continuous quality, RF-teacher distillation,
    and free residual energy.
    """

    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.reason_head = nn.Linear(hidden // 2, 5)
        self.quality_head = nn.Linear(hidden // 2, 1)
        self.teacher_head = nn.Linear(hidden // 2, 1)
        self.energy_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([x, task_oh], dim=-1))

        binary_logits = self.binary_head(h)
        reason_logits = self.reason_head(h)
        quality_logit = self.quality_head(h).squeeze(-1)
        teacher_logit = self.teacher_head(h).squeeze(-1)
        free_energy = self.energy_head(h).squeeze(-1)

        good_margin = binary_logits[:, 1] - binary_logits[:, 0]
        bad_reason_score = torch.logsumexp(
            torch.stack([reason_logits[:, 0], torch.logsumexp(reason_logits[:, 2:], dim=-1)], dim=-1),
            dim=-1,
        )
        reason_margin = reason_logits[:, 1] - bad_reason_score

        energy_logit = (
            0.45 * quality_logit
            + 0.30 * teacher_logit
            + 0.15 * good_margin
            + 0.10 * reason_margin
            + 0.10 * free_energy
        )
        energy_clipped = torch.tanh(energy_logit / 4.0) * 4.0

        return {
            "binary_logits": binary_logits,
            "reason_logits": reason_logits,
            "quality_logit": quality_logit,
            "teacher_logit": teacher_logit,
            "free_energy": free_energy,
            "good_margin": good_margin,
            "reason_margin": reason_margin,
            "energy_logit": energy_logit,
            "energy_clipped": energy_clipped,
        }
