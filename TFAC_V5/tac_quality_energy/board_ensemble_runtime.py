"""Ablation-only differentiable ensemble for board TacQuality scorers.

This runtime keeps the deployment contract of ``score(marker, action)`` while
combining two already-trained ForceBand scorers.  It is intended for offline
ablation and gradient-audit experiments; it is not the current default board
guidance scorer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from .force_band_runtime import ForceBandTacQualityEnergyRuntime


DEFAULT_OLD_CKPT = (
    "/home/chenshuai/Project/output/"
    "board_force_band_tac_quality_energy_with_260617_positive_20260618/"
    "force_band_tac_quality_energy_best.pt"
)
DEFAULT_S12_CKPT = (
    "/home/chenshuai/Project/output/"
    "board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "force_band_tac_quality_energy_best.pt"
)


class BoardForceBandEnsembleRuntime(nn.Module):
    """Weighted differentiable ensemble of two board ForceBand scorers.

    The default weights come from the 2026-06-19 offline ensemble sweep:
    raw old-weight 0.95 was the best smooth/deployable candidate, while the
    rank-normalized 0.85 candidate is kept offline only because rank is not a
    local differentiable serving transform.
    """

    def __init__(
        self,
        old_checkpoint: str | Path = DEFAULT_OLD_CKPT,
        s12_checkpoint: str | Path = DEFAULT_S12_CKPT,
        *,
        old_weight: float = 0.95,
        s12_weight: Optional[float] = None,
        old_mode: str = "energy_clipped",
        s12_mode: str = "energy_clipped",
        device: str = "cuda:0",
    ):
        super().__init__()
        self.old = ForceBandTacQualityEnergyRuntime(str(old_checkpoint), device=device)
        self.s12 = ForceBandTacQualityEnergyRuntime(str(s12_checkpoint), device=device)
        self.old_weight = float(old_weight)
        self.s12_weight = float(1.0 - old_weight if s12_weight is None else s12_weight)
        total = self.old_weight + self.s12_weight
        if abs(total) < 1e-8:
            raise ValueError("Ensemble weights must not sum to zero.")
        self.old_weight /= total
        self.s12_weight /= total
        self.old_mode = old_mode
        self.s12_mode = s12_mode
        self.device = self.old.device

    def _scores(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        old_score = self.old.score(
            left_marker_seq,
            right_marker_seq=right_marker_seq,
            eef_action_seq=eef_action_seq,
            joint_action_seq=joint_action_seq,
            mode=self.old_mode,
        )
        s12_score = self.s12.score(
            left_marker_seq,
            right_marker_seq=right_marker_seq,
            eef_action_seq=eef_action_seq,
            joint_action_seq=joint_action_seq,
            mode=self.s12_mode,
        )
        ensemble = self.old_weight * old_score + self.s12_weight * s12_score
        return {"old_score": old_score, "s12_score": s12_score, "ensemble_score": ensemble}

    def forward(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del task_id
        out = self._scores(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq)
        out.update(
            {
                "energy_logit": out["ensemble_score"],
                "energy_clipped": torch.tanh(out["ensemble_score"] / 4.0) * 4.0,
                "old_weight": torch.as_tensor(self.old_weight, device=out["ensemble_score"].device),
                "s12_weight": torch.as_tensor(self.s12_weight, device=out["ensemble_score"].device),
            }
        )
        return out

    def score(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        mode: str = "energy_clipped",
    ) -> torch.Tensor:
        out = self.forward(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq, task_id)
        if mode in {"ensemble", "energy", "energy_logit"}:
            return out["ensemble_score"]
        if mode == "energy_clipped":
            return out["energy_clipped"]
        if mode == "old":
            return out["old_score"]
        if mode == "s12":
            return out["s12_score"]
        raise ValueError(f"Unknown BoardForceBandEnsemble score mode: {mode}")

