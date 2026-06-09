"""Runtime wrapper for the distilled differentiable TacQualityEnergy scorer.

The checkpoint is trained by train_distilled_tac_quality_energy.py from the
same proxy feature schema as PTGProxyScorerV2:

  left marker + right marker + abs diff + eef action + joint action -> energy

This wrapper reconstructs those proxy features in torch, so gradients can flow
from the guidance energy back to predicted tactile fields and candidate actions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.action_aware_scorer_runtime import action_proxy_features_torch, marker_proxy_features_torch  # noqa: E402


TASK_TO_ID = {"insertion": 0, "board": 1}
DEFAULT_CKPT = "/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt"


class DistilledTacQualityEnergy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.0):
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
        reason_margin = reason_logits[:, 1] - torch.logsumexp(
            torch.stack([reason_logits[:, 0], torch.logsumexp(reason_logits[:, 2:], dim=-1)], dim=-1),
            dim=-1,
        )
        energy_logit = (
            0.45 * quality_logit
            + 0.30 * teacher_logit
            + 0.15 * good_margin
            + 0.10 * reason_margin
            + 0.10 * free_energy
        )
        return {
            "binary_logits": binary_logits,
            "reason_logits": reason_logits,
            "quality_logit": quality_logit,
            "teacher_logit": teacher_logit,
            "free_energy": free_energy,
            "good_margin": good_margin,
            "reason_margin": reason_margin,
            "energy_logit": energy_logit,
            "energy_clipped": torch.tanh(energy_logit / 4.0) * 4.0,
        }


def _as_action(action: Optional[torch.Tensor], batch: int, window: int, dim: int, device) -> torch.Tensor:
    if action is None:
        return torch.zeros(batch, window, dim, device=device)
    action = action.to(device).float()
    if action.shape[-1] > dim:
        return action[..., :dim]
    if action.shape[-1] < dim:
        pad = torch.zeros(*action.shape[:-1], dim - action.shape[-1], dtype=action.dtype, device=device)
        return torch.cat([action, pad], dim=-1)
    return action


class DistilledTacQualityEnergyRuntime(nn.Module):
    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, device: str = "cuda:0"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.feature_dim = int(ckpt["feature_dim"])
        self.model = DistilledTacQualityEnergy(
            self.feature_dim,
            hidden=int(ckpt.get("hidden", 192)),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.task_to_id = ckpt.get("task_to_id", TASK_TO_ID)
        self.register_buffer("scaler_mean", torch.as_tensor(ckpt["scaler_mean"], dtype=torch.float32, device=self.device))
        self.register_buffer("scaler_scale", torch.as_tensor(ckpt["scaler_scale"], dtype=torch.float32, device=self.device))

    def proxy_features(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        left = left_marker_seq.to(self.device).float()
        right = right_marker_seq.to(self.device).float() if right_marker_seq is not None else left
        batch = left.shape[0]
        window = left.shape[1]
        left_feat = marker_proxy_features_torch(left)
        right_feat = marker_proxy_features_torch(right)
        diff_feat = (left_feat - right_feat).abs()
        eef_action = _as_action(eef_action_seq, batch, window, 6, self.device)
        joint_action = _as_action(joint_action_seq, batch, window, 7, self.device)
        eef_feat = action_proxy_features_torch(eef_action, action_dim=6)
        joint_feat = action_proxy_features_torch(joint_action, action_dim=7)
        return torch.cat([left_feat, right_feat, diff_feat, eef_feat, joint_feat], dim=-1)

    def forward(
        self,
        left_marker_seq: torch.Tensor,
        right_marker_seq: Optional[torch.Tensor] = None,
        eef_action_seq: Optional[torch.Tensor] = None,
        joint_action_seq: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        feat = self.proxy_features(left_marker_seq, right_marker_seq, eef_action_seq, joint_action_seq)
        feat_norm = (feat - self.scaler_mean.view(1, -1)) / (self.scaler_scale.view(1, -1) + 1e-8)
        if task_id is None:
            task_id = torch.zeros(feat.shape[0], dtype=torch.long, device=self.device)
        else:
            task_id = task_id.to(self.device).long()
        raw = self.model(feat_norm, task_id)
        p_good = torch.softmax(raw["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(raw["reason_logits"], dim=-1)
        raw.update(
            {
                "p_good": p_good,
                "log_p_good": torch.log(p_good.clamp_min(1e-8)),
                "reason_prob": reason_prob,
                "quality_score": torch.sigmoid(raw["quality_logit"]),
                "teacher_score": torch.sigmoid(raw["teacher_logit"]),
                "proxy_features": feat,
                "proxy_features_norm": feat_norm,
            }
        )
        return raw

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
        if mode == "energy_clipped":
            return out["energy_clipped"]
        if mode == "energy":
            return out["energy_logit"]
        if mode == "quality":
            return out["quality_score"]
        if mode == "teacher":
            return out["teacher_score"]
        if mode == "p_good":
            return out["p_good"]
        if mode == "log_p_good":
            return out["log_p_good"]
        if mode == "reason_good":
            return out["reason_prob"][:, 1]
        raise ValueError(mode)


def sanity(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    runtime = DistilledTacQualityEnergyRuntime(args.checkpoint, device=args.device)
    bsz = args.batch_size
    window = args.window
    device = runtime.device
    left = torch.randn(bsz, window, 9, 9, 2, device=device, requires_grad=True)
    right = torch.randn(bsz, window, 9, 9, 2, device=device, requires_grad=True)
    eef = torch.randn(bsz, window, 6, device=device, requires_grad=True)
    joint = torch.randn(bsz, window, 7, device=device, requires_grad=True)
    task_id = torch.tensor([0, 1] * (bsz // 2), dtype=torch.long, device=device)
    score = runtime.score(left, right_marker_seq=right, eef_action_seq=eef, joint_action_seq=joint, task_id=task_id)
    grads = torch.autograd.grad(score.sum(), [left, right, eef, joint], retain_graph=False)
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "score_mean": float(score.detach().mean().cpu()),
        "score_std": float(score.detach().std().cpu()),
        "all_finite": bool(torch.isfinite(score).all().item() and all(torch.isfinite(g).all().item() for g in grads)),
        "left_grad_norm": float(grads[0].flatten(1).norm(dim=1).mean().detach().cpu()),
        "right_grad_norm": float(grads[1].flatten(1).norm(dim=1).mean().detach().cpu()),
        "eef_grad_norm": float(grads[2].flatten(1).norm(dim=1).mean().detach().cpu()),
        "joint_grad_norm": float(grads[3].flatten(1).norm(dim=1).mean().detach().cpu()),
    }
    result["usable_for_guidance"] = bool(
        result["all_finite"]
        and result["left_grad_norm"] > 1e-8
        and result["right_grad_norm"] > 1e-8
        and result["eef_grad_norm"] > 1e-8
        and result["joint_grad_norm"] > 1e-8
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="/home/chenshuai/Project/output/distilled_tac_quality_energy/runtime_sanity.json")
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
