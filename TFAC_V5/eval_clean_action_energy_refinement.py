"""One-shot clean-action refinement with TacQualityEnergy.

This evaluates a safer alternative to injecting gradients into noisy DDPM
states.  DP first produces a clean action trajectory normally.  Then a small
trust-region optimizer refines the final raw action directly:

  clean action -> Foresight -> predicted tactile -> TacQualityEnergy -> dS/da

The refinement is accepted per sample only when the scorer improves, and the
action update is bounded relative to the original DP output.
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
from TFAC_V5.eval_tac_energy_guided_denoising import action_smoothness_np, summarize  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/clean_action_energy_refinement")


def get_action_limits(reranker):
    stats = reranker.dp_config.get("norm_stats", {})
    if "action_min" not in stats or "action_max" not in stats:
        return None, None
    action_min = torch.tensor(stats["action_min"], dtype=torch.float32, device=reranker.device)
    action_max = torch.tensor(stats["action_max"], dtype=torch.float32, device=reranker.device)
    return action_min, action_max


def action_limit_penalty(actions, action_min, action_max, margin_frac):
    """Soft barrier for raw actions approaching/leaving the training action range."""
    if action_min is None or action_max is None:
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
    span = (action_max - action_min).clamp_min(1e-6).view(1, 1, -1)
    inner_min = action_min.view(1, 1, -1) + margin_frac * span
    inner_max = action_max.view(1, 1, -1) - margin_frac * span
    low_violation = torch.relu((inner_min - actions) / span)
    high_violation = torch.relu((actions - inner_max) / span)
    return (low_violation.square() + high_violation.square()).mean(dim=(1, 2))


def action_limit_penalty_np(actions, action_min, action_max, margin_frac):
    if action_min is None or action_max is None:
        return np.zeros(actions.shape[0], dtype=np.float32)
    action_min = np.asarray(action_min, dtype=np.float32)
    action_max = np.asarray(action_max, dtype=np.float32)
    span = np.maximum(action_max - action_min, 1e-6).reshape(1, 1, -1)
    inner_min = action_min.reshape(1, 1, -1) + margin_frac * span
    inner_max = action_max.reshape(1, 1, -1) - margin_frac * span
    low_violation = np.maximum((inner_min - actions) / span, 0.0)
    high_violation = np.maximum((actions - inner_max) / span, 0.0)
    return np.mean(low_violation**2 + high_violation**2, axis=(1, 2))


def hard_range_violation_np(actions, action_min, action_max):
    if action_min is None or action_max is None:
        return np.zeros(actions.shape[0], dtype=np.float32)
    action_min = np.asarray(action_min, dtype=np.float32).reshape(1, 1, -1)
    action_max = np.asarray(action_max, dtype=np.float32).reshape(1, 1, -1)
    below = np.maximum(action_min - actions, 0.0)
    above = np.maximum(actions - action_max, 0.0)
    return np.linalg.norm((below + above).reshape(actions.shape[0], -1), axis=1)


def objective_score(args, reranker, scorer, actions, data, vae_mean, vae_std, action_min, action_max):
    score, _, _ = score_actions(
        reranker,
        scorer,
        actions,
        data,
        vae_mean,
        vae_std,
        args.score_mode,
        args.smooth_weight,
    )
    if args.joint_limit_weight:
        score = score - args.joint_limit_weight * action_limit_penalty(actions, action_min, action_max, args.joint_margin_frac)
    return score


def refine_actions(args, reranker, scorer, actions_raw, data, vae_mean, vae_std, action_min, action_max):
    original = actions_raw.detach()
    current = original.clone()
    logs = []
    for step_idx in range(args.refine_steps):
        x = current.detach().clone().requires_grad_(True)
        score = objective_score(args, reranker, scorer, x, data, vae_mean, vae_std, action_min, action_max)
        grad = torch.autograd.grad(score.sum(), x, retain_graph=False)[0]
        grad_norm = grad.flatten(1).norm(dim=1)
        grad_unit = grad / grad_norm.view(-1, 1, 1).clamp_min(1e-8)
        with torch.no_grad():
            proposal = x + args.refine_step_size * grad_unit
            delta = proposal - original
            delta_norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
            delta = delta * torch.clamp(args.max_total_delta / delta_norm, max=1.0)
            proposal = original + delta
        score_new = objective_score(
            args,
            reranker,
            scorer,
            proposal.detach().clone().requires_grad_(True),
            data,
            vae_mean,
            vae_std,
            action_min,
            action_max,
        )
        accept = score_new.detach() > score.detach()
        with torch.no_grad():
            current = torch.where(accept.view(-1, 1, 1), proposal, current)
        logs.append(
            {
                "step": int(step_idx),
                "score_mean": float(score.detach().mean().cpu()),
                "score_new_mean": float(score_new.detach().mean().cpu()),
                "accept_rate": float(accept.float().mean().cpu()),
                "grad_norm_mean": float(grad_norm.detach().mean().cpu()),
            }
        )
    return current.detach(), logs


def score_np(args, reranker, scorer, actions, data, vae_mean, vae_std, action_min, action_max):
    score = objective_score(
        args,
        reranker,
        scorer,
        actions.detach().clone().requires_grad_(True),
        data,
        vae_mean,
        vae_std,
        action_min,
        action_max,
    )
    return score.detach().cpu().numpy()


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
    action_min, action_max = get_action_limits(reranker)
    action_min_np = None if action_min is None else action_min.detach().cpu().numpy()
    action_max_np = None if action_max is None else action_max.detach().cpu().numpy()

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
    all_base_scores, all_refined_scores = [], []
    all_base_smooth, all_refined_smooth = [], []
    all_delta_norm = []
    all_base_barrier, all_refined_barrier = [], []
    all_base_hard_violation, all_refined_hard_violation = [], []
    all_accept_rates = []
    for hdf5_path, ep_name, t in tqdm(frames, desc="Clean-action energy refinement"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
            base_raw = reranker.generate_candidates(obs_cond, K=args.K)
            refined_raw, refine_logs = refine_actions(args, reranker, scorer, base_raw, data, vae_mean, vae_std, action_min, action_max)

            base_score = score_np(args, reranker, scorer, base_raw, data, vae_mean, vae_std, action_min, action_max)
            refined_score = score_np(args, reranker, scorer, refined_raw, data, vae_mean, vae_std, action_min, action_max)
            base_np = base_raw.detach().cpu().numpy()
            refined_np = refined_raw.detach().cpu().numpy()
            delta_norm = np.linalg.norm((refined_np - base_np).reshape(args.K, -1), axis=1)
            base_smooth = action_smoothness_np(base_np)
            refined_smooth = action_smoothness_np(refined_np)
            base_barrier = action_limit_penalty_np(base_np, action_min_np, action_max_np, args.joint_margin_frac)
            refined_barrier = action_limit_penalty_np(refined_np, action_min_np, action_max_np, args.joint_margin_frac)
            base_hard_violation = hard_range_violation_np(base_np, action_min_np, action_max_np)
            refined_hard_violation = hard_range_violation_np(refined_np, action_min_np, action_max_np)

            all_base_scores.append(base_score)
            all_refined_scores.append(refined_score)
            all_base_smooth.append(base_smooth)
            all_refined_smooth.append(refined_smooth)
            all_delta_norm.append(delta_norm)
            all_base_barrier.append(base_barrier)
            all_refined_barrier.append(refined_barrier)
            all_base_hard_violation.append(base_hard_violation)
            all_refined_hard_violation.append(refined_hard_violation)
            all_accept_rates.extend([x["accept_rate"] for x in refine_logs])
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "base_score_mean": float(base_score.mean()),
                    "refined_score_mean": float(refined_score.mean()),
                    "score_delta_mean": float((refined_score - base_score).mean()),
                    "refined_beats_base_rate": float(np.mean(refined_score > base_score)),
                    "base_smooth_mean": float(base_smooth.mean()),
                    "refined_smooth_mean": float(refined_smooth.mean()),
                    "base_limit_barrier_mean": float(base_barrier.mean()),
                    "refined_limit_barrier_mean": float(refined_barrier.mean()),
                    "base_hard_range_violation_mean": float(base_hard_violation.mean()),
                    "refined_hard_range_violation_mean": float(refined_hard_violation.mean()),
                    "delta_norm_mean": float(delta_norm.mean()),
                    "refine_logs": refine_logs,
                }
            )
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    if not rows:
        raise RuntimeError("No valid rows")

    base_scores = np.concatenate(all_base_scores)
    refined_scores = np.concatenate(all_refined_scores)
    base_smooth = np.concatenate(all_base_smooth)
    refined_smooth = np.concatenate(all_refined_smooth)
    delta_norm = np.concatenate(all_delta_norm)
    base_barrier = np.concatenate(all_base_barrier)
    refined_barrier = np.concatenate(all_refined_barrier)
    base_hard_violation = np.concatenate(all_base_hard_violation)
    refined_hard_violation = np.concatenate(all_refined_hard_violation)
    result = {
        "config": vars(args),
        "n_frames": len(rows),
        "n_action_samples": int(len(base_scores)),
        "summary": {
            "base_score": summarize(base_scores),
            "refined_score": summarize(refined_scores),
            "score_delta": summarize(refined_scores - base_scores),
            "refined_beats_base_rate": float(np.mean(refined_scores > base_scores)),
            "base_smoothness": summarize(base_smooth),
            "refined_smoothness": summarize(refined_smooth),
            "smoothness_delta": summarize(refined_smooth - base_smooth),
            "action_delta_norm": summarize(delta_norm),
            "base_limit_barrier": summarize(base_barrier),
            "refined_limit_barrier": summarize(refined_barrier),
            "limit_barrier_delta": summarize(refined_barrier - base_barrier),
            "base_hard_range_violation": summarize(base_hard_violation),
            "refined_hard_range_violation": summarize(refined_hard_violation),
            "accept_rate_per_step": summarize(all_accept_rates),
        },
        "interpretation": {
            "passes_clean_refinement_sanity": bool(
                np.mean(refined_scores > base_scores) > 0.95
                and np.mean(refined_scores - base_scores) > 0
                and np.percentile(delta_norm, 95) <= args.max_total_delta + 1e-5
                and np.percentile(refined_hard_violation, 95) <= 1e-5
            ),
            "meaning": "Post-DP clean-action trust-region refinement increases constrained TacQualityEnergy with bounded action change.",
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
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--joint_limit_weight", type=float, default=0.0)
    parser.add_argument("--joint_margin_frac", type=float, default=0.03)
    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--refine_step_size", type=float, default=0.02)
    parser.add_argument("--max_total_delta", type=float, default=0.08)
    parser.add_argument("--output", default=str(OUT_DIR / "insertion_clean_refine_energy_clipped_K4_N8.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
