"""Foresight-to-TacQuality bridge for DP classifier guidance.

This module converts a trained LatentForesight-style model into the
``foresight_predict_fn(action_raw) -> tactile_dict`` contract required by
``TacQualityDPIntegrationAdapter``.

The bridge is intentionally narrow:

  raw action -> Foresight action/qpos normalization
  Foresight latent prediction -> TactileVAE decoder
  decoded marker normalization -> raw marker sequence
  TacQuality input dict with differentiable tensors

It is not a reranker.  It is the differentiable path used by final-clean-action
TacQuality guidance.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_dp_integration_adapter import (  # noqa: E402
    TacQualityDPIntegrationAdapter,
    summarize_tensor,
)
from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


DEFAULT_OUT = "/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.json"
DEFAULT_MD = "/home/chenshuai/Project/output/tac_quality_foresight_bridge/foresight_bridge_sanity.md"


@dataclass(frozen=True)
class ForesightBridgeConfig:
    task: str = "insertion"
    window: int = 8
    action_chunk: int = 10
    latent_dim: int = 16
    marker_mean: Tuple[float, float] = (0.2102, -0.6422)
    marker_std: Tuple[float, float] = (1.6805, 3.6717)
    residual_prediction: bool = False
    board_right_source: str = "mirror_left"


def _normalize_action(action_raw: torch.Tensor, fs_norm: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(fs_norm["action_mean"], dtype=action_raw.dtype, device=action_raw.device).view(1, 1, -1)
    std = torch.as_tensor(fs_norm["action_std"], dtype=action_raw.dtype, device=action_raw.device).view(1, 1, -1)
    return (action_raw - mean) / std.clamp_min(1e-8)


def _normalize_qpos(qpos_raw: torch.Tensor, fs_norm: Mapping[str, Any]) -> torch.Tensor:
    mean = torch.as_tensor(fs_norm["qpos_mean"], dtype=qpos_raw.dtype, device=qpos_raw.device).view(1, -1)
    std = torch.as_tensor(fs_norm["qpos_std"], dtype=qpos_raw.dtype, device=qpos_raw.device).view(1, -1)
    return (qpos_raw - mean) / std.clamp_min(1e-8)


def _expand_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"Cannot expand tensor with batch {tensor.shape[0]} to {batch}")
    return tensor.expand(batch, *tensor.shape[1:])


def _last_or_sequence(z_pred: torch.Tensor) -> torch.Tensor:
    if z_pred.dim() == 2:
        return z_pred.unsqueeze(1)
    if z_pred.dim() == 3:
        return z_pred
    raise ValueError(f"Expected z_pred shape (B,D) or (B,L,D), got {tuple(z_pred.shape)}")


class ForesightTacQualityBridge(nn.Module):
    """Wrap Foresight as the TacQuality adapter's differentiable predict fn."""

    def __init__(
        self,
        foresight: nn.Module,
        fs_norm: Mapping[str, Any],
        *,
        qpos_raw: torch.Tensor,
        foresight_images: Sequence[torch.Tensor],
        marker_window_norm: Optional[torch.Tensor] = None,
        config: Optional[ForesightBridgeConfig] = None,
    ):
        super().__init__()
        self.foresight = foresight
        self.fs_norm = fs_norm
        self.config = config or ForesightBridgeConfig()
        device = next(foresight.parameters(), torch.empty(0)).device
        self.register_buffer("qpos_raw", qpos_raw.detach().float().to(device).view(1, -1), persistent=False)
        self.foresight_images = [img.detach().float().to(device) for img in foresight_images]
        if marker_window_norm is not None:
            self.register_buffer("marker_window_norm", marker_window_norm.detach().float().to(device), persistent=False)
        else:
            self.marker_window_norm = None
        self.register_buffer(
            "marker_mean",
            torch.tensor(self.config.marker_mean, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2),
            persistent=False,
        )
        self.register_buffer(
            "marker_std",
            torch.tensor(self.config.marker_std, dtype=torch.float32, device=device).view(1, 1, 1, 1, 2),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self.qpos_raw.device

    def _build_images(self, batch: int) -> List[torch.Tensor]:
        images = [_expand_batch(img.to(self.device), batch) for img in self.foresight_images]
        if self.marker_window_norm is not None:
            images.append(_expand_batch(self.marker_window_norm.to(self.device), batch))
        return images

    def _call_foresight(self, images: Sequence[torch.Tensor], action_fs_norm: torch.Tensor, qpos_norm: torch.Tensor):
        try:
            return self.foresight(images, action_fs_norm, future_images=None, qpos=qpos_norm)
        except TypeError:
            return self.foresight(images, action_fs_norm, qpos=qpos_norm)

    def _apply_residual_if_needed(self, z_seq: torch.Tensor, batch: int) -> torch.Tensor:
        if not self.config.residual_prediction:
            return z_seq
        if self.marker_window_norm is None or not hasattr(self.foresight, "tactile_vae"):
            raise ValueError("residual_prediction requires marker_window_norm and foresight.tactile_vae")
        marker_win = _expand_batch(self.marker_window_norm.to(self.device), batch)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(batch, 1, -1)
        return z_seq + z_cur.expand_as(z_seq)

    def decode_latent_to_marker_seq(self, z_pred: torch.Tensor) -> torch.Tensor:
        z_seq = _last_or_sequence(z_pred)
        batch, length, dim = z_seq.shape
        expected = self.config.latent_dim * 3 * 3
        if dim != expected:
            raise ValueError(f"Expected latent dim {expected}, got {dim}")
        z_seq = self._apply_residual_if_needed(z_seq, batch)
        z_spatial = z_seq.reshape(batch * length, self.config.latent_dim, 3, 3)
        marker_norm = self.foresight.tactile_vae.decoder(z_spatial)
        marker_norm = marker_norm.view(batch, length, 9, 9, 2)
        marker_raw = marker_norm * self.marker_std + self.marker_mean
        if length >= self.config.window:
            return marker_raw[:, -self.config.window :]
        return marker_raw[:, -1:].expand(batch, self.config.window, 9, 9, 2)

    def forward(self, action_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_raw = action_raw.to(self.device).float()
        batch = action_raw.shape[0]
        action_fs = action_raw[:, : self.config.action_chunk, :]
        action_fs_norm = _normalize_action(action_fs, self.fs_norm)
        qpos = _expand_batch(self.qpos_raw, batch)
        qpos_norm = _normalize_qpos(qpos, self.fs_norm)
        outputs = self._call_foresight(self._build_images(batch), action_fs_norm, qpos_norm)
        z_pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        left_marker = self.decode_latent_to_marker_seq(z_pred)
        action_seq = action_raw[:, : self.config.window, :]
        result: Dict[str, torch.Tensor] = {
            "left_marker_seq": left_marker,
            "eef_action_seq": action_seq[..., :6] if action_seq.shape[-1] >= 6 else torch.nn.functional.pad(action_seq, (0, 6 - action_seq.shape[-1])),
        }
        if self.config.task.lower() == "board":
            if self.config.board_right_source != "mirror_left":
                raise ValueError(f"Unsupported board_right_source={self.config.board_right_source!r}")
            result["right_marker_seq"] = left_marker
        return result


class SyntheticTactileVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.decoder_head = nn.Linear(latent_dim * 3 * 3, 9 * 9 * 2)

    def decoder(self, z_spatial: torch.Tensor) -> torch.Tensor:
        flat = z_spatial.flatten(1)
        return self.decoder_head(flat).view(z_spatial.shape[0], 9, 9, 2)


class SyntheticLatentForesight(nn.Module):
    def __init__(self, action_dim: int = 7, latent_dim: int = 16, pred_steps: int = 4):
        super().__init__()
        self.pred_steps = pred_steps
        self.latent_dim = latent_dim
        self.tactile_vae = SyntheticTactileVAE(latent_dim)
        self.action_proj = nn.Linear(action_dim, pred_steps * latent_dim * 3 * 3)
        self.qpos_proj = nn.Linear(action_dim, pred_steps * latent_dim * 3 * 3)

    def forward(self, images: Sequence[torch.Tensor], action: torch.Tensor, future_images=None, qpos=None):
        del images, future_images
        pooled = action.mean(dim=1)
        qpos_term = 0.0 if qpos is None else self.qpos_proj(qpos)
        z = self.action_proj(pooled) + qpos_term
        z = z.view(action.shape[0], self.pred_steps, self.latent_dim * 3 * 3)
        z_current = z[:, 0]
        return z, None, None, z_current, None, None


def _grad_check(bridge: ForesightTacQualityBridge, action: torch.Tensor) -> Dict[str, Any]:
    x = action.detach().clone().requires_grad_(True)
    tactile = bridge(x)
    scalar = tactile["left_marker_seq"].square().mean() + tactile["eef_action_seq"].square().mean()
    grad = torch.autograd.grad(scalar, x, retain_graph=False)[0]
    return {
        "scalar": float(scalar.detach().cpu()),
        "grad": summarize_tensor(grad.flatten(1).norm(dim=1)),
        "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
        "positive_grad_rate": float((grad.flatten(1).norm(dim=1) > 1e-8).float().mean().cpu()),
    }


def _shape_summary(tactile: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    return {key: list(value.shape) for key, value in tactile.items()}


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Foresight Bridge Sanity",
        "",
        f"- passes_foresight_bridge_sanity: `{result['passes_foresight_bridge_sanity']}`",
        f"- purpose: {result['purpose']}",
        f"- guidance_mode: `{result['guidance_mode']}`",
        "",
        "## Checks",
        "",
        f"- insertion bridge grad positive rate: `{result['insertion']['bridge_grad']['positive_grad_rate']}`",
        f"- board bridge grad positive rate: `{result['board']['bridge_grad']['positive_grad_rate']}`",
        f"- insertion adapter improved rate: `{result['insertion']['adapter_report']['improved_rate']}`",
        f"- board adapter improved rate: `{result['board']['adapter_report']['improved_rate']}`",
        "",
        "## Contract",
        "",
        "```text",
        "action_raw -> Foresight(action_mean/action_std, qpos_mean/qpos_std)",
        "z_pred -> tactile_vae.decoder -> marker_raw",
        "marker_raw/action_raw -> TacQuality score -> d score / d action_raw",
        "```",
        "",
        "This is a differentiable final-action guidance bridge, not candidate reranking.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def sanity(args) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    runtime = TacQualityGuidanceRuntime(device=str(device))
    action_dim = args.action_dim
    fs_norm = {
        "action_mean": torch.zeros(action_dim, device=device),
        "action_std": torch.ones(action_dim, device=device),
        "qpos_mean": torch.zeros(action_dim, device=device),
        "qpos_std": torch.ones(action_dim, device=device),
    }
    qpos = torch.zeros(1, action_dim, device=device)
    images = [
        torch.zeros(1, 3, 64, 64, device=device),
        torch.zeros(1, 3, 64, 64, device=device),
    ]
    marker_window = torch.zeros(1, args.window, 9, 9, 2, device=device)
    action = torch.randn(args.batch_size, args.horizon, action_dim, device=device) * 0.1

    insertion_foresight = SyntheticLatentForesight(action_dim=action_dim, pred_steps=args.pred_steps).to(device)
    insertion_bridge = ForesightTacQualityBridge(
        insertion_foresight,
        fs_norm,
        qpos_raw=qpos,
        foresight_images=images,
        marker_window_norm=marker_window,
        config=ForesightBridgeConfig(task="insertion", window=args.window, action_chunk=args.horizon),
    )
    insertion_tactile = insertion_bridge(action.detach().clone().requires_grad_(True))
    insertion_adapter = TacQualityDPIntegrationAdapter("insertion", runtime=runtime)
    _, insertion_report = insertion_adapter.guide_final_action(action, insertion_bridge)

    board_foresight = SyntheticLatentForesight(action_dim=action_dim, pred_steps=args.pred_steps).to(device)
    board_bridge = ForesightTacQualityBridge(
        board_foresight,
        fs_norm,
        qpos_raw=qpos,
        foresight_images=images,
        marker_window_norm=marker_window,
        config=ForesightBridgeConfig(task="board", window=args.window, action_chunk=args.horizon),
    )
    board_tactile = board_bridge(action.detach().clone().requires_grad_(True))
    board_adapter = TacQualityDPIntegrationAdapter("board", runtime=runtime)
    _, board_report = board_adapter.guide_final_action(action, board_bridge)

    result = {
        "purpose": "Verify the differentiable Foresight -> decoded marker -> TacQuality adapter contract.",
        "device": str(device),
        "guidance_mode": "final_clean_action_trust_region_refinement",
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "bridge_config": {
            "insertion": asdict(insertion_bridge.config),
            "board": asdict(board_bridge.config),
        },
        "insertion": {
            "tactile_shapes": _shape_summary(insertion_tactile),
            "bridge_grad": _grad_check(insertion_bridge, action),
            "adapter_report": insertion_report,
        },
        "board": {
            "tactile_shapes": _shape_summary(board_tactile),
            "bridge_grad": _grad_check(board_bridge, action),
            "adapter_report": board_report,
        },
    }
    result["passes_foresight_bridge_sanity"] = bool(
        result["not_reranking"]
        and result["not_every_step_ddpm_guidance"]
        and result["insertion"]["tactile_shapes"]["left_marker_seq"] == [args.batch_size, args.window, 9, 9, 2]
        and result["board"]["tactile_shapes"]["left_marker_seq"] == [args.batch_size, args.window, 9, 9, 2]
        and result["board"]["tactile_shapes"]["right_marker_seq"] == [args.batch_size, args.window, 9, 9, 2]
        and result["insertion"]["bridge_grad"]["finite_grad_rate"] >= 0.999
        and result["board"]["bridge_grad"]["finite_grad_rate"] >= 0.999
        and result["insertion"]["bridge_grad"]["positive_grad_rate"] >= 0.999
        and result["board"]["bridge_grad"]["positive_grad_rate"] >= 0.999
        and insertion_report["finite_grad_rate"] >= 0.999
        and board_report["finite_grad_rate"] >= 0.999
        and insertion_report["improved_rate"] >= args.min_improved_rate
        and board_report["improved_rate"] >= args.min_improved_rate
        and insertion_report["max_delta_within_trust_region"]
        and board_report["max_delta_within_trust_region"]
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, Path(args.markdown))
    print(
        json.dumps(
            {
                "passes_foresight_bridge_sanity": result["passes_foresight_bridge_sanity"],
                "insertion_improved_rate": insertion_report["improved_rate"],
                "board_improved_rate": board_report["improved_rate"],
                "json": str(out),
                "markdown": args.markdown,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--pred_steps", type=int, default=4)
    parser.add_argument("--min_improved_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--markdown", default=DEFAULT_MD)
    return parser.parse_args()


if __name__ == "__main__":
    sanity(parse_args())
