"""Insertion DP denoising smoke using TacQualityDPGuidanceController.

This is a controller-in-the-loop version of the earlier guided denoising dry
run.  It keeps the frozen DP and Foresight stack unchanged and replaces the
hand-written guidance step with:

  guided_action, report = controller.guide(noisy_action, current_score_fn)

The score_fn recomputes the current normalized action -> raw action ->
Foresight predicted tactile -> TacQuality score path on every call.
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
from TFAC_V5.eval_tac_energy_guided_denoising import action_smoothness_np, dp_step, eval_final_scores, summarize  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402
from TFAC_V5.tac_quality_dp_guidance_controller import from_guidance_profile  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_controller_denoising_smoke")


def controller_guide_step(args, reranker, scorer, controller, noisy_action, data, vae_mean, vae_std):
    def current_score_fn(action_norm):
        raw_action = reranker._dp_unnorm_action(action_norm)
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
        return score

    guided, report = controller.guide(noisy_action, current_score_fn)
    if args.clamp_norm_action:
        guided = guided.clamp(-1.0, 1.0)
    return guided.detach(), report


def sample_actions(args, reranker, scorer, controller, obs_cond, data, vae_mean, vae_std, initial_noise, guided):
    k = initial_noise.shape[0]
    obs_cond_k = obs_cond.expand(k, -1)
    noisy_action = initial_noise.clone()
    reranker.noise_scheduler.set_timesteps(reranker.dp_config.get("num_inference_steps", 100))
    timesteps = list(reranker.noise_scheduler.timesteps)
    guide_start = int(len(timesteps) * args.guide_start_frac)
    guide_logs = []

    for step_idx, timestep in enumerate(timesteps):
        noisy_action = dp_step(reranker, noisy_action, timestep, obs_cond_k)
        if args.clamp_norm_action:
            noisy_action = noisy_action.clamp(-1.0, 1.0)
        if guided and step_idx >= guide_start and ((step_idx - guide_start) % args.guide_every == 0):
            noisy_action, report = controller_guide_step(args, reranker, scorer, controller, noisy_action, data, vae_mean, vae_std)
            guide_logs.append(
                {
                    "step_idx": int(step_idx),
                    "timestep": int(timestep),
                    "score_mean": report["base_score"]["mean"],
                    "score_delta_mean": report["score_delta"]["mean"],
                    "accept_rate": report["accept_rate"],
                    "grad_norm_mean": report["grad_norm"]["mean"],
                    "finite_grad_rate": report["finite_grad_rate"],
                    "stale_gradient_reuse_allowed": report["guardrails"]["stale_gradient_reuse_allowed"],
                    "max_delta_within_trust_region": report["max_delta_within_trust_region"],
                }
            )
    return reranker._dp_unnorm_action(noisy_action), noisy_action, guide_logs


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
    controller = from_guidance_profile("insertion", scale=args.guidance_scale, clamp_norm_action=args.clamp_norm_action)

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
    guide_accept_rates, guide_delta_means, guide_grad_norms = [], [], []
    guide_finite_rates, guide_trust_region_rates, guide_stale_allowed = [], [], []

    for hdf5_path, ep_name, t in tqdm(frames, desc="Controller guided denoising"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
            initial_noise = torch.randn(args.K, reranker.pred_horizon, reranker.action_dim, device=reranker.device)
            base_raw, base_norm, _ = sample_actions(
                args, reranker, scorer, controller, obs_cond, data, vae_mean, vae_std, initial_noise, guided=False
            )
            guided_raw, guided_norm, guide_logs = sample_actions(
                args, reranker, scorer, controller, obs_cond, data, vae_mean, vae_std, initial_noise, guided=True
            )
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
            guide_accept_rates.extend([g["accept_rate"] for g in guide_logs])
            guide_delta_means.extend([g["score_delta_mean"] for g in guide_logs])
            guide_grad_norms.extend([g["grad_norm_mean"] for g in guide_logs])
            guide_finite_rates.extend([g["finite_grad_rate"] for g in guide_logs])
            guide_trust_region_rates.extend([float(g["max_delta_within_trust_region"]) for g in guide_logs])
            guide_stale_allowed.extend([float(g["stale_gradient_reuse_allowed"]) for g in guide_logs])
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "base_score_mean": float(base_score.mean()),
                    "guided_score_mean": float(guided_score.mean()),
                    "score_delta_mean": float((guided_score - base_score).mean()),
                    "guided_beats_base_rate": float(np.mean(guided_score > base_score)),
                    "range_violation_max": float(range_violation.max()),
                    "n_guidance_steps": len(guide_logs),
                    "controller_accept_rate_mean": float(np.mean([g["accept_rate"] for g in guide_logs])) if guide_logs else 0.0,
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
        "purpose": "Controller-in-the-loop insertion DP denoising smoke.",
        "scope": "Frozen DP/Foresight dry-run; not robot validation.",
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
            "controller_accept_rate_per_step": summarize(guide_accept_rates),
            "controller_score_delta_per_step": summarize(guide_delta_means),
            "controller_grad_norm_per_step": summarize(guide_grad_norms),
            "controller_finite_grad_rate_per_step": summarize(guide_finite_rates),
            "controller_trust_region_pass_rate_per_step": summarize(guide_trust_region_rates),
            "controller_stale_gradient_allowed_rate": summarize(guide_stale_allowed),
        },
        "interpretation": {
            "passes_controller_denoising_smoke": bool(
                np.mean(guided_scores > base_scores) > args.pass_beats
                and np.mean(guided_scores - base_scores) > 0
                and np.max(range_violation) <= 1e-6
                and np.mean(guide_finite_rates) >= 0.999
                and np.mean(guide_trust_region_rates) >= 0.999
                and np.max(guide_stale_allowed) == 0.0
            ),
            "meaning": "TacQualityDPGuidanceController can be inserted into the DP denoising loop and improve final TacQuality scores.",
        },
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(
        "\n".join(
            [
                "# TacQuality Controller Denoising Smoke",
                "",
                f"- pass: `{result['interpretation']['passes_controller_denoising_smoke']}`",
                f"- n_frames: `{result['n_frames']}`",
                f"- n_action_samples: `{result['n_action_samples']}`",
                f"- guided_beats_base_rate: `{result['summary']['guided_beats_base_rate']}`",
                f"- score_delta_mean: `{result['summary']['score_delta']['mean']}`",
                f"- range_violation_max: `{result['summary']['range_violation']['max']}`",
                f"- stale_gradient_allowed_rate: `{result['summary']['controller_stale_gradient_allowed_rate']['mean']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
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
    parser.add_argument("--n_eval", type=int, default=4)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--score_mode", default="energy_clipped", choices=["quality", "log_p_good", "risk_guidance", "energy", "energy_clipped"])
    parser.add_argument("--guide_start_frac", type=float, default=0.80)
    parser.add_argument("--guide_every", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=0.01)
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--clamp_norm_action", action="store_true", default=True)
    parser.add_argument("--pass_beats", type=float, default=0.80)
    parser.add_argument("--output", default=str(OUT_DIR / "insertion_controller_denoising_K4_N4.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
