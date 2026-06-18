#!/usr/bin/env python3
"""Audit TacQuality guidance inside DP denoising steps.

This is stronger than the clean-action and noisy-action audits because it
places the current deployed TacQuality scorer/Foresight chain inside the DDPM
sampling loop.  It is still an offline audit: it does not prove real robot
improvement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from for_show_xiaomi.serve_dp_tac_quality_guided import (  # noqa: E402
    GuidedDPStack,
    freeze,
    make_synthetic_obs,
)
from TFAC_V5.tac_quality_energy.foresight_bridge import SyntheticLatentForesight  # noqa: E402


DEFAULT_BOARD_DP_DIR = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext"
)
DEFAULT_BOARD_FORESIGHT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload"
)
DEFAULT_BOARD_FORESIGHT_CKPT = DEFAULT_BOARD_FORESIGHT_DIR / "foresight_best.ckpt"
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_marker_joint_20260618.json"
)
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit")


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def summarize(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def predict_x0_from_eps(
    sample: torch.Tensor,
    eps: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    *,
    clip: bool = True,
) -> torch.Tensor:
    alpha = alphas_cumprod[timestep.to(alphas_cumprod.device).long()].to(sample.device, sample.dtype)
    while alpha.ndim < sample.ndim:
        alpha = alpha.view(*alpha.shape, 1)
    x0 = (sample - (1.0 - alpha).sqrt() * eps) / alpha.sqrt().clamp_min(1e-8)
    if clip:
        x0 = x0.clamp(-1.0, 1.0)
    return x0


def unit_update(
    grad: torch.Tensor,
    *,
    scale: float,
    min_grad_norm: float,
    max_delta_norm: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    flat = grad.flatten(1)
    grad_norm = flat.norm(dim=1).clamp_min(min_grad_norm)
    update = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1)))
    update = update * float(scale)
    update_norm = update.flatten(1).norm(dim=1)
    if max_delta_norm > 0:
        coef = (float(max_delta_norm) / update_norm.clamp_min(1e-8)).clamp(max=1.0)
        update = update * coef.view(-1, *([1] * (update.ndim - 1)))
        update_norm = update.flatten(1).norm(dim=1)
    finite_rate = torch.isfinite(grad).flatten(1).all(dim=1).float().mean()
    return update, {
        "grad_norm": float(grad_norm.mean().detach().cpu()),
        "grad_norm_max": float(grad_norm.max().detach().cpu()),
        "finite_grad_rate": float(finite_rate.detach().cpu()),
        "update_norm": float(update_norm.mean().detach().cpu()),
    }


def set_synthetic_foresight(stack: GuidedDPStack, marker_mean: float = 0.0, marker_std: float = 1.0) -> None:
    stack.foresight = SyntheticLatentForesight(action_dim=stack.action_dim).to(stack.device)
    freeze(stack.foresight)
    stack.fs_norm = {
        "action_mean": torch.zeros(stack.action_dim, device=stack.device),
        "action_std": torch.ones(stack.action_dim, device=stack.device),
        "qpos_mean": torch.zeros(stack.action_dim, device=stack.device),
        "qpos_std": torch.ones(stack.action_dim, device=stack.device),
    }
    stack.fs_config = {
        "camera_names": ["global", "wrist", "gelsight"],
        "chunk_size": min(10, stack.pred_horizon),
        "tactile_vae_latent_dim": 16,
        "norm_stats": {
            "marker_offset_mean": [marker_mean, marker_mean],
            "marker_offset_std": [marker_std, marker_std],
        },
    }
    stack.fs_marker_mean = torch.full((1, 1, 1, 1, 2), marker_mean, device=stack.device)
    stack.fs_marker_std = torch.full((1, 1, 1, 1, 2), marker_std, device=stack.device)


def load_episode_obs(
    stack: GuidedDPStack,
    dataset_dir: Path,
    episode_id: int,
    start: int,
) -> Tuple[deque[Dict[str, Any]], List[np.ndarray]]:
    path = dataset_dir / f"episode_{episode_id}.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    proprio_key = stack.config.get("proprio_key", "proprio_joint")
    tac_side = stack.config.get("tac_side", "left")
    obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
    marker_buffer: List[np.ndarray] = []
    with h5py.File(path, "r") as f:
        length = int(f[f"observations/{proprio_key}"].shape[0])
        if start < 0:
            start = max(0, length + start)
        start = max(0, min(start, length - 1))
        first = max(0, start - stack.obs_horizon + 1)
        for idx in range(first, start + 1):
            obs = {
                "images": {
                    cam: f[f"observations/images/{cam}"][idx]
                    for cam in stack.camera_names
                    if cam != "gelsight"
                },
                "qpos": f[f"observations/{proprio_key}"][idx].astype(np.float32),
                "tac": {
                    tac_side: {
                        "marker_offset": f[f"observations/tac/{tac_side}/marker_offset"][idx].astype(np.float32)
                    }
                },
            }
            processed = stack.preprocess_obs(obs)
            marker_buffer.append(processed["marker_offset"])
            processed["_marker_idx"] = len(marker_buffer) - 1
            obs_buffer.append(processed)
    while len(obs_buffer) < stack.obs_horizon:
        pad = dict(obs_buffer[0])
        pad["_marker_idx"] = 0
        obs_buffer.appendleft(pad)
    return obs_buffer, marker_buffer


def synthetic_obs_context(stack: GuidedDPStack, marker_value: float) -> Tuple[deque[Dict[str, Any]], List[np.ndarray]]:
    obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
    marker_buffer: List[np.ndarray] = []
    obs = make_synthetic_obs(stack, marker_value=marker_value)
    for idx in range(stack.obs_horizon):
        processed = stack.preprocess_obs(obs)
        marker_buffer.append(processed["marker_offset"])
        processed["_marker_idx"] = idx
        obs_buffer.append(processed)
    return obs_buffer, marker_buffer


def score_x0(stack: GuidedDPStack, x0_norm: torch.Tensor, bridge) -> torch.Tensor:
    if stack.guidance is None:
        raise RuntimeError("TacQuality guidance must be enabled for scoring")
    action_raw = stack.guidance.adapter.action_normalizer.denormalize(x0_norm)
    tactile = bridge(action_raw)
    return stack.guidance.adapter.score_from_prediction(tactile, action_raw)


def run_sample(
    stack: GuidedDPStack,
    obs_buffer: deque[Dict[str, Any]],
    marker_buffer: List[np.ndarray],
    *,
    seed: int,
    guidance_steps: int,
    guidance_scale: float,
    max_delta_norm: float,
    sample_clip: float,
) -> Dict[str, Any]:
    obs_cond = stack.build_obs_cond(list(obs_buffer), marker_buffer)
    bridge = stack.make_bridge(obs_buffer[-1], marker_buffer)
    contact_gate = stack.contact_gate_report(marker_buffer)
    gate_value = float(contact_gate.get("contact_gate_value", 1.0))
    stack.noise_scheduler.set_timesteps(stack.num_inference_steps)
    timesteps = list(stack.noise_scheduler.timesteps)
    alphas = stack.noise_scheduler.alphas_cumprod.to(stack.device)
    generator = torch.Generator(device=stack.device)
    generator.manual_seed(int(seed))
    initial = torch.randn(
        (1, stack.pred_horizon, stack.action_dim),
        generator=generator,
        device=stack.device,
    )

    def denoise(enable_guidance: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        action = initial.detach().clone()
        guide_start = max(0, len(timesteps) - int(guidance_steps))
        logs: List[Dict[str, Any]] = []
        for step_idx, t in enumerate(timesteps):
            t_batch = t.reshape(1).to(stack.device)
            with torch.no_grad():
                eps = stack.noise_pred_net(action, t_batch, global_cond=obs_cond)
            do_guide = (
                enable_guidance
                and guidance_steps > 0
                and guidance_scale > 0.0
                and gate_value > 0.0
                and step_idx >= guide_start
            )
            if do_guide:
                action_for_grad = action.detach().clone().requires_grad_(True)
                x0_before = predict_x0_from_eps(action_for_grad, eps.detach(), t, alphas, clip=True)
                score_before = score_x0(stack, x0_before, bridge)
                objective = score_before.mean()
                grad = torch.autograd.grad(objective, action_for_grad, retain_graph=False)[0]
                update, grad_report = unit_update(
                    grad,
                    scale=guidance_scale * gate_value,
                    min_grad_norm=1e-8,
                    max_delta_norm=max_delta_norm,
                )
                guided_action = (action_for_grad.detach() + update).clamp(-sample_clip, sample_clip)
                with torch.no_grad():
                    eps_after = stack.noise_pred_net(guided_action, t_batch, global_cond=obs_cond)
                    x0_after = predict_x0_from_eps(guided_action, eps_after, t, alphas, clip=True)
                    score_after = score_x0(stack, x0_after, bridge)
                logs.append(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(t.item()),
                        "score_before": float(score_before.mean().detach().cpu()),
                        "score_after": float(score_after.mean().detach().cpu()),
                        "score_delta": float((score_after - score_before.detach()).mean().detach().cpu()),
                        "contact_gate_value": gate_value,
                        **grad_report,
                    }
                )
                action = guided_action.detach()
                eps = eps_after.detach()
            with torch.no_grad():
                action = stack.noise_scheduler.step(eps, t, action).prev_sample.detach()
        final_score = score_x0(stack, action.detach().clone().requires_grad_(True), bridge).detach()
        return action.detach(), {
            "final_score": float(final_score.mean().cpu()),
            "guided_steps": len(logs),
            "logs": logs,
        }

    base_action, base = denoise(enable_guidance=False)
    guided_action, guided = denoise(enable_guidance=True)
    delta = (guided_action - base_action).flatten(1).norm(dim=1)
    per_step_delta = [row["score_delta"] for row in guided["logs"]]
    return {
        "seed": int(seed),
        "contact_gate": contact_gate,
        "base_final_score": base["final_score"],
        "guided_final_score": guided["final_score"],
        "guided_minus_base_final_score": guided["final_score"] - base["final_score"],
        "guided_steps": guided["guided_steps"],
        "guided_action_delta_norm": float(delta.mean().cpu()),
        "per_step_score_delta": summarize(per_step_delta),
        "finite_grad_rate": float(np.mean([row["finite_grad_rate"] for row in guided["logs"]])) if guided["logs"] else 1.0,
        "positive_grad_rate": float(np.mean([row["grad_norm"] > 1e-8 for row in guided["logs"]])) if guided["logs"] else 1.0,
        "logs": guided["logs"],
    }


def render_markdown(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# DDPM-Step TacQuality Guidance Audit",
        "",
        f"- task: `{result['task']}`",
        f"- arm: `{result['arm']}`",
        f"- evidence boundary: {result['evidence_boundary']}",
        f"- DP ckpt: `{result['inputs']['ckpt_dir']}/{result['inputs']['ckpt_name']}`",
        f"- Foresight: `{result['inputs']['foresight_ckpt']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| samples | {summary['n_samples']} |",
        f"| guidance steps | {result['guidance']['guidance_steps']} |",
        f"| inference steps | {result['guidance']['num_inference_steps']} |",
        f"| final score improve rate | {summary['final_score_improve_rate']:.4f} |",
        f"| final score delta mean | {summary['final_score_delta']['mean']:.6f} |",
        f"| per-step score delta mean | {summary['per_step_score_delta_mean']['mean']:.6f} |",
        f"| finite grad mean | {summary['finite_grad_rate']['mean']:.4f} |",
        f"| action delta norm mean | {summary['guided_action_delta_norm']['mean']:.6f} |",
        "",
        "## Interpretation",
        "",
        "- This audit inserts TacQuality guidance into the DDPM denoising loop and scores the predicted clean action estimate.",
        "- Passing this audit means the current scorer can provide finite local gradients inside the sampler under this configuration.",
        "- It still does not prove real robot improvement; that requires paired baseline/guided rollouts with force/outcome logs.",
    ]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["board", "insertion"], default="board")
    parser.add_argument("--arm", default="marker_joint_guided")
    parser.add_argument("--ckpt_dir", default=str(DEFAULT_BOARD_DP_DIR))
    parser.add_argument("--ckpt_name", default="dp_best.pth")
    parser.add_argument("--vae_checkpoint_override", default=None)
    parser.add_argument("--foresight_dir", default=str(DEFAULT_BOARD_FORESIGHT_DIR))
    parser.add_argument("--foresight_ckpt", default=str(DEFAULT_BOARD_FORESIGHT_CKPT))
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_CONFIG))
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--start", type=int, default=64)
    parser.add_argument("--use_synthetic_obs", action="store_true")
    parser.add_argument("--synthetic_foresight_for_smoke", action="store_true")
    parser.add_argument("--smoke_marker_value", type=float, default=3.0)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num_inference_steps", type=int, default=8)
    parser.add_argument("--guidance_steps", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=0.002)
    parser.add_argument("--max_delta_norm", type=float, default=0.02)
    parser.add_argument("--sample_clip", type=float, default=1.0)
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--dp_norm_mode", choices=["minmax", "standard", "identity"], default="minmax")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--disable_guidance", action="store_true")
    parser.add_argument("--disable_contact_gate", action="store_true")
    parser.add_argument("--contact_gate_low", type=float, default=1.8)
    parser.add_argument("--contact_gate_high", type=float, default=2.3)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR / "board_marker_joint_current_smoke"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.disable_guidance:
        raise ValueError("This audit requires guidance; do not pass --disable_guidance")

    stack = GuidedDPStack(args)
    stack.num_inference_steps = int(args.num_inference_steps)
    if args.synthetic_foresight_for_smoke:
        set_synthetic_foresight(stack)

    if args.use_synthetic_obs or not args.dataset_dir:
        obs_buffer, marker_buffer = synthetic_obs_context(stack, float(args.smoke_marker_value))
        obs_source = {"kind": "synthetic", "marker_value": float(args.smoke_marker_value)}
    else:
        obs_buffer, marker_buffer = load_episode_obs(
            stack,
            Path(args.dataset_dir),
            int(args.episode_id),
            int(args.start),
        )
        obs_source = {
            "kind": "hdf5",
            "dataset_dir": str(args.dataset_dir),
            "episode_id": int(args.episode_id),
            "start": int(args.start),
        }

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    rows = [
        run_sample(
            stack,
            obs_buffer,
            marker_buffer,
            seed=seed,
            guidance_steps=int(args.guidance_steps),
            guidance_scale=float(args.guidance_scale),
            max_delta_norm=float(args.max_delta_norm),
            sample_clip=float(args.sample_clip),
        )
        for seed in seeds
    ]
    final_deltas = [row["guided_minus_base_final_score"] for row in rows]
    result = {
        "purpose": "Audit current TacQuality scorer as DDPM-step classifier guidance.",
        "evidence_boundary": (
            "Offline sampler audit only. It verifies local DDPM-step gradient behavior; "
            "it does not prove real robot improvement."
        ),
        "task": args.task,
        "arm": args.arm,
        "obs_source": obs_source,
        "inputs": {
            "ckpt_dir": str(args.ckpt_dir),
            "ckpt_name": str(args.ckpt_name),
            "foresight_dir": str(args.foresight_dir),
            "foresight_ckpt": str(args.foresight_ckpt),
            "rollout_arm_config": str(args.rollout_arm_config),
            "synthetic_foresight_for_smoke": bool(args.synthetic_foresight_for_smoke),
        },
        "guidance": {
            "scheduler": args.scheduler,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_steps": int(args.guidance_steps),
            "guidance_scale": float(args.guidance_scale),
            "max_delta_norm": float(args.max_delta_norm),
            "sample_clip": float(args.sample_clip),
        },
        "summary": {
            "n_samples": len(rows),
            "final_score_delta": summarize(final_deltas),
            "final_score_improve_rate": float(np.mean([x > 0.0 for x in final_deltas])) if rows else 0.0,
            "per_step_score_delta_mean": summarize([row["per_step_score_delta"].get("mean", 0.0) for row in rows]),
            "finite_grad_rate": summarize([row["finite_grad_rate"] for row in rows]),
            "positive_grad_rate": summarize([row["positive_grad_rate"] for row in rows]),
            "guided_action_delta_norm": summarize([row["guided_action_delta_norm"] for row in rows]),
        },
        "rows": rows,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "ddpm_step_guidance_audit.json"
    md_path = out_dir / "ddpm_step_guidance_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "final_score_improve_rate": result["summary"]["final_score_improve_rate"],
        "final_score_delta_mean": result["summary"]["final_score_delta"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
