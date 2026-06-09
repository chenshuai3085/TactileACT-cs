"""Runtime wrapper for the insertion-specific risk scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.action_aware_scorer_runtime import (  # noqa: E402
    action_proxy_features_torch,
    marker_proxy_features_torch,
)


DEFAULT_CKPT = "/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"
WINDOW = 8
ACTION_DIM = 7


class InsertionRiskScorer(nn.Module):
    def __init__(self, marker_proxy_dim: int, action_proxy_dim: int, action_dim: int = ACTION_DIM, hidden: int = 160, dropout: float = 0.0):
        super().__init__()
        self.action_dim = action_dim
        self.marker_encoder = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        in_dim = 48 * 2 * 3 * 3 + 48 * 4 + marker_proxy_dim + action_proxy_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.reason_head = nn.Linear(hidden // 2, 4)
        self.quality_head = nn.Linear(hidden // 2, 1)

    def forward(self, marker, marker_proxy, action, action_proxy):
        marker_feat = self.marker_encoder(marker.permute(0, 4, 1, 2, 3))
        action_feat = self.action_encoder(action.permute(0, 2, 1))
        h = self.encoder(torch.cat([marker_feat, action_feat, marker_proxy, action_proxy], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "reason_logits": self.reason_head(h),
            "quality": self.quality_head(h).squeeze(-1),
        }


def _ensure_action(action_seq: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action_seq.shape[-1] > action_dim:
        return action_seq[..., :action_dim]
    if action_seq.shape[-1] < action_dim:
        pad = torch.zeros(*action_seq.shape[:-1], action_dim - action_seq.shape[-1], device=action_seq.device)
        return torch.cat([action_seq, pad], dim=-1)
    return action_seq


class InsertionRiskScorerRuntime(nn.Module):
    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, device: str = "cuda"):
        super().__init__()
        self.device_name = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.device = torch.device(self.device_name)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.action_dim = int(ckpt.get("action_dim", ACTION_DIM))
        self.model = InsertionRiskScorer(
            marker_proxy_dim=int(ckpt["marker_proxy_dim"]),
            action_proxy_dim=int(ckpt["action_proxy_dim"]),
            action_dim=self.action_dim,
            hidden=int(ckpt.get("hidden", 160)),
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.reason_names = ckpt.get("reason_names", {})
        self.register_buffer("marker_mean", torch.as_tensor(ckpt["marker_mean"], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("marker_std", torch.as_tensor(ckpt["marker_std"], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 2))
        self.register_buffer("action_mean", torch.as_tensor(ckpt["action_mean"], dtype=torch.float32, device=self.device).view(1, 1, self.action_dim))
        self.register_buffer("action_std", torch.as_tensor(ckpt["action_std"], dtype=torch.float32, device=self.device).view(1, 1, self.action_dim))
        self.register_buffer("marker_proxy_mean", torch.as_tensor(ckpt["marker_proxy_mean"], dtype=torch.float32, device=self.device).view(1, -1))
        self.register_buffer("marker_proxy_scale", torch.as_tensor(ckpt["marker_proxy_scale"], dtype=torch.float32, device=self.device).view(1, -1))
        self.register_buffer("action_proxy_mean", torch.as_tensor(ckpt["action_proxy_mean"], dtype=torch.float32, device=self.device).view(1, -1))
        self.register_buffer("action_proxy_scale", torch.as_tensor(ckpt["action_proxy_scale"], dtype=torch.float32, device=self.device).view(1, -1))

    def forward(self, marker_seq: torch.Tensor, action_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        marker_seq = marker_seq.to(self.device).float()
        action_seq = _ensure_action(action_seq.to(self.device).float(), self.action_dim)
        marker_proxy = marker_proxy_features_torch(marker_seq)
        action_proxy = action_proxy_features_torch(action_seq, self.action_dim)
        marker_norm = (marker_seq - self.marker_mean) / self.marker_std
        action_norm = (action_seq - self.action_mean) / self.action_std
        mp_norm = (marker_proxy - self.marker_proxy_mean) / (self.marker_proxy_scale + 1e-8)
        ap_norm = (action_proxy - self.action_proxy_mean) / (self.action_proxy_scale + 1e-8)
        out = self.model(marker_norm, mp_norm, action_norm, ap_norm)
        p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(out["reason_logits"], dim=-1)
        risk_prob = reason_prob[:, 2] + reason_prob[:, 3]
        quality = torch.sigmoid(out["quality"])
        out.update(
            {
                "p_good": p_good,
                "log_p_good": torch.log(p_good.clamp_min(1e-8)),
                "reason_prob": reason_prob,
                "risk_prob": risk_prob,
                "quality_score": quality,
                "marker_proxy": marker_proxy,
                "action_proxy": action_proxy,
            }
        )
        return out

    def score(self, marker_seq: torch.Tensor, action_seq: torch.Tensor, mode: str = "risk_guidance") -> torch.Tensor:
        out = self.forward(marker_seq, action_seq)
        if mode == "quality":
            return out["quality_score"]
        if mode == "p_good":
            return out["p_good"]
        if mode == "log_p_good":
            return out["log_p_good"]
        if mode == "neg_risk":
            return -out["risk_prob"]
        if mode == "risk_guidance":
            return out["quality_score"] + 0.35 * out["log_p_good"] - 0.5 * out["risk_prob"]
        raise ValueError(mode)


def sanity(args):
    torch.manual_seed(0)
    runtime = InsertionRiskScorerRuntime(args.checkpoint, args.device)
    device = runtime.device
    marker = torch.randn(8, WINDOW, 9, 9, 2, device=device, requires_grad=True)
    action = torch.randn(8, WINDOW, runtime.action_dim, device=device, requires_grad=True)
    score = runtime.score(marker, action, mode=args.mode).mean()
    grad_marker, grad_action = torch.autograd.grad(score, [marker, action], retain_graph=False)
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "mode": args.mode,
        "score": float(score.detach().cpu()),
        "grad_marker_norm": float(grad_marker.norm().detach().cpu()),
        "grad_action_norm": float(grad_action.norm().detach().cpu()),
        "usable_for_guidance": bool(
            torch.isfinite(grad_marker).all().item()
            and torch.isfinite(grad_action).all().item()
            and grad_marker.norm().item() > 1e-8
            and grad_action.norm().item() > 1e-8
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", default="risk_guidance")
    parser.add_argument("--output", default="/home/chenshuai/Project/output/insertion_risk_scorer/runtime_gradient_sanity.json")
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
