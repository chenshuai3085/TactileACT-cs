"""Prototype TacQualityEnergy guidance inside the DP denoising loop.

This is a dry-run integration script, not a deployment path.  It keeps the
trained DP frozen, samples from the same initial noise twice, and compares:

  baseline DDPM denoising
  guided DDPM denoising with late-step TacQualityEnergy gradients

The goal is to test whether the scorer can act as classifier guidance during
denoising without editing the main policy code.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TFAC_V5"))

from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.eval_action_aware_reranking import ForesightOnlyReranker, load_tactile_vae_norm  # noqa: E402
from TFAC_V5.eval_full_chain_guidance_gradient import freeze, score_actions  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_energy_guided_denoising")


def summarize(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def action_smoothness_np(actions):
    if actions.shape[1] < 3:
        return np.zeros(actions.shape[0], dtype=np.float32)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return np.linalg.norm(accel, axis=-1).mean(axis=1)


@torch.no_grad()
def dp_step(reranker, noisy_action, timestep, obs_cond_k):
    noise_pred = reranker.noise_pred_net(sample=noisy_action, timestep=timestep, global_cond=obs_cond_k)
    return reranker.noise_scheduler.step(model_output=noise_pred, timestep=timestep, sample=noisy_action).prev_sample


def guide_step(args, reranker, scorer, noisy_action, data, vae_mean, vae_std):
    x = noisy_action.detach().clone().requires_grad_(True)
    raw_action = reranker._dp_unnorm_action(x)
    score, _, _ = score_actions(
        reranker,
        scorer,
        raw_action,
        data,
        vae_mean,
        vae_std,
        args.score_mode,
        args.smooth_weight,
    )
    grad = torch.autograd.grad(score.sum(), x, retain_graph=False)[0]
    grad_norm = grad.flatten(1).norm(dim=1)
    grad_unit = grad / grad_norm.view(-1, 1, 1).clamp_min(1e-8)
    with torch.no_grad():
        step = args.guidance_scale * grad_unit
        if args.max_norm_delta_per_step > 0:
            step_norm = step.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
            step = step * torch.clamp(args.max_norm_delta_per_step / step_norm, max=1.0)
        guided = x + step
        if args.clamp_norm_action:
            guided = guided.clamp(-1.0, 1.0)
    if args.accept_only_improved:
        with torch.no_grad():
            raw_guided = reranker._dp_unnorm_action(guided)
        score_new, _, _ = score_actions(
            reranker,
            scorer,
            raw_guided,
            data,
            vae_mean,
            vae_std,
            args.score_mode,
            args.smooth_weight,
        )
        accept = (score_new > score).view(-1, 1, 1)
        guided = torch.where(accept, guided, x.detach())
        accept_rate = float(accept.float().mean().detach().cpu())
    else:
        accept_rate = 1.0
    return guided.detach(), score.detach(), grad_norm.detach(), accept_rate


def sample_actions(args, reranker, scorer, obs_cond, data, vae_mean, vae_std, initial_noise, guided):
    k = initial_noise.shape[0]
    obs_cond_k = obs_cond.expand(k, -1)
    noisy_action = initial_noise.clone()
    reranker.noise_scheduler.set_timesteps(reranker.dp_config.get("num_inference_steps", 100))
    timesteps = list(reranker.noise_scheduler.timesteps)
    guide_start = int(len(timesteps) * args.guide_start_frac)
    guide_logs = []

    for step_idx, timestep in enumerate(timesteps):
        noisy_action = dp_step(reranker, noisy_action, timestep, obs_cond_k)
        if guided and step_idx >= guide_start and ((step_idx - guide_start) % args.guide_every == 0):
            noisy_action, score, grad_norm, accept_rate = guide_step(args, reranker, scorer, noisy_action, data, vae_mean, vae_std)
            guide_logs.append(
                {
                    "step_idx": int(step_idx),
                    "timestep": int(timestep),
                    "score_mean": float(score.mean().detach().cpu()),
                    "grad_norm_mean": float(grad_norm.mean().detach().cpu()),
                    "grad_norm_max": float(grad_norm.max().detach().cpu()),
                    "accept_rate": accept_rate,
                }
            )
    return reranker._dp_unnorm_action(noisy_action), noisy_action, guide_logs


def eval_final_scores(args, reranker, scorer, actions_raw, data, vae_mean, vae_std):
    with torch.enable_grad():
        actions = actions_raw.detach().clone().requires_grad_(True)
        score, marker, z = score_actions(
            reranker,
            scorer,
            actions,
            data,
            vae_mean,
            vae_std,
            args.score_mode,
            args.smooth_weight,
        )
    return score.detach().cpu().numpy(), marker.detach().cpu().numpy(), z.detach().cpu().numpy()


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    reranker = ForesightOnlyReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )
    freeze(reranker.noise_pred_net)
    freeze(reranker.dp_vision)
    freeze(reranker.dp_tac_encoder)
    freeze(reranker.foresight)
    scorer = InsertionRiskScorerRuntime(args.scorer_ckpt, device=args.device)
    freeze(scorer)

    vae_ckpt = reranker.foresight_config.get(
        "tactile_vae_ckpt", "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt"
    )
    vae_mean, vae_std = load_tactile_vae_norm(vae_ckpt)

    with open(os.path.join(args.data_dir, "annotations.pkl"), "rb") as f:
        ann = pickle.load(f)
    frames = _collect_insertion_frames(
        _find_hdf5_files(args.data_dir),
        ann,
        cs=reranker.pred_horizon,
        obs_horizon=reranker.obs_horizon,
    )
    if len(frames) > args.n_eval:
        frames = [frames[i] for i in rng.choice(len(frames), args.n_eval, replace=False)]

    rows = []
    all_base_scores, all_guided_scores = [], []
    all_base_smooth, all_guided_smooth = [], []
    all_norm_delta, all_range_violation = [], []
    guide_grad_norms = []
    guide_accept_rates = []

    for hdf5_path, ep_name, t in tqdm(frames, desc="TacEnergy guided denoising"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
            initial_noise = torch.randn(args.K, reranker.pred_horizon, reranker.action_dim, device=reranker.device)
            base_raw, base_norm, _ = sample_actions(args, reranker, scorer, obs_cond, data, vae_mean, vae_std, initial_noise, guided=False)
            guided_raw, guided_norm, guide_logs = sample_actions(args, reranker, scorer, obs_cond, data, vae_mean, vae_std, initial_noise, guided=True)

            base_score, _, _ = eval_final_scores(args, reranker, scorer, base_raw, data, vae_mean, vae_std)
            guided_score, _, _ = eval_final_scores(args, reranker, scorer, guided_raw, data, vae_mean, vae_std)
            base_np = base_raw.detach().cpu().numpy()
            guided_np = guided_raw.detach().cpu().numpy()
            base_norm_np = base_norm.detach().cpu().numpy()
            guided_norm_np = guided_norm.detach().cpu().numpy()
            norm_delta = np.linalg.norm((guided_norm_np - base_norm_np).reshape(args.K, -1), axis=1)
            range_violation = np.maximum(np.abs(guided_norm_np) - 1.0, 0.0).max(axis=(1, 2))
            base_smooth = action_smoothness_np(base_np)
            guided_smooth = action_smoothness_np(guided_np)
            all_base_scores.append(base_score)
            all_guided_scores.append(guided_score)
            all_base_smooth.append(base_smooth)
            all_guided_smooth.append(guided_smooth)
            all_norm_delta.append(norm_delta)
            all_range_violation.append(range_violation)
            guide_grad_norms.extend([g["grad_norm_mean"] for g in guide_logs])
            guide_accept_rates.extend([g["accept_rate"] for g in guide_logs])
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "base_score_mean": float(base_score.mean()),
                    "guided_score_mean": float(guided_score.mean()),
                    "score_delta_mean": float((guided_score - base_score).mean()),
                    "guided_beats_base_rate": float(np.mean(guided_score > base_score)),
                    "base_smooth_mean": float(base_smooth.mean()),
                    "guided_smooth_mean": float(guided_smooth.mean()),
                    "norm_action_delta_mean": float(norm_delta.mean()),
                    "range_violation_max": float(range_violation.max()),
                    "n_guidance_steps": len(guide_logs),
                }
            )
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    if not rows:
        raise RuntimeError("No valid rows")

    base_scores = np.concatenate(all_base_scores)
    guided_scores = np.concatenate(all_guided_scores)
    base_smooth = np.concatenate(all_base_smooth)
    guided_smooth = np.concatenate(all_guided_smooth)
    norm_delta = np.concatenate(all_norm_delta)
    range_violation = np.concatenate(all_range_violation)
    result = {
        "config": vars(args),
        "n_frames": len(rows),
        "n_action_samples": int(len(base_scores)),
        "summary": {
            "base_score": summarize(base_scores),
            "guided_score": summarize(guided_scores),
            "score_delta": summarize(guided_scores - base_scores),
            "guided_beats_base_rate": float(np.mean(guided_scores > base_scores)),
            "base_smoothness": summarize(base_smooth),
            "guided_smoothness": summarize(guided_smooth),
            "smoothness_delta": summarize(guided_smooth - base_smooth),
            "norm_action_delta": summarize(norm_delta),
            "range_violation": summarize(range_violation),
            "guide_grad_norm_mean_per_step": summarize(guide_grad_norms),
            "guide_accept_rate_per_step": summarize(guide_accept_rates),
        },
        "interpretation": {
            "passes_guided_denoising_sanity": bool(
                np.mean(guided_scores > base_scores) > 0.85
                and np.mean(guided_scores - base_scores) > 0
                and np.max(range_violation) <= 1e-6
            ),
            "meaning": "Late-step DP denoising guidance increases TacQualityEnergy while keeping normalized action inside [-1,1].",
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0414")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n_eval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_mode", default="energy_clipped", choices=["quality", "log_p_good", "risk_guidance", "energy", "energy_clipped"])
    parser.add_argument("--guide_start_frac", type=float, default=0.75)
    parser.add_argument("--guide_every", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=0.015)
    parser.add_argument("--max_norm_delta_per_step", type=float, default=0.0)
    parser.add_argument("--accept_only_improved", action="store_true")
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--clamp_norm_action", action="store_true", default=True)
    parser.add_argument("--output", default=str(OUT_DIR / "insertion_guided_denoising_energy_clipped_K4_N8.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
