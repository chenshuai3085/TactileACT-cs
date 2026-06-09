"""Compare insertion default scorer and distilled TacQualityEnergy.

This is an insertion counterpart of the board distilled clean-refine
comparison.  It runs the same DP -> Foresight -> scorer -> action-gradient
trust-region refinement protocol for:

  - insertion_risk: insertion-specific default scorer;
  - distilled_energy: cross-task RF-teacher-distilled differentiable energy.

Scope: offline clean-action refinement with the insertion DP/Foresight stack.
This is not a robot rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm


ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TFAC_V5"))

from TFAC_V5.distilled_tac_quality_energy_runtime import DistilledTacQualityEnergyRuntime  # noqa: E402
from TFAC_V5.dp_reranking import _collect_insertion_frames, _find_hdf5_files, _load_frame_data  # noqa: E402
from TFAC_V5.eval_action_aware_reranking import ForesightOnlyReranker, load_tactile_vae_norm  # noqa: E402
from TFAC_V5.eval_clean_action_energy_refinement import (  # noqa: E402
    action_limit_penalty,
    action_limit_penalty_np,
    get_action_limits,
    hard_range_violation_np,
)
from TFAC_V5.eval_full_chain_guidance_gradient import freeze, predict_marker_differentiable  # noqa: E402
from TFAC_V5.eval_tac_energy_guided_denoising import action_smoothness_np, summarize  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime, WINDOW  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison")


def scorer_energy(args, scorer_name, scorer, marker_seq, action_seq):
    if scorer_name == "insertion_risk":
        return scorer.score(marker_seq, action_seq, mode=args.insertion_score_mode)
    if scorer_name == "distilled_energy":
        task_id = torch.zeros(marker_seq.shape[0], dtype=torch.long, device=marker_seq.device)
        return scorer.score(
            marker_seq,
            right_marker_seq=marker_seq,
            joint_action_seq=action_seq,
            task_id=task_id,
            mode="energy_clipped" if not args.no_clip else "energy",
        )
    raise ValueError(scorer_name)


def objective_score(args, reranker, scorer_name, scorer, actions, data, vae_mean, vae_std, action_min, action_max):
    marker_raw, _ = predict_marker_differentiable(
        reranker,
        actions,
        data["qpos_raw"],
        data["marker_window"],
        data["foresight_images"],
        vae_mean,
        vae_std,
    )
    marker_seq = marker_raw.unsqueeze(1).expand(-1, WINDOW, -1, -1, -1)
    action_seq = actions[:, :WINDOW, :]
    score = scorer_energy(args, scorer_name, scorer, marker_seq, action_seq)
    if args.smooth_weight:
        accel = action_seq[:, 2:] - 2 * action_seq[:, 1:-1] + action_seq[:, :-2]
        score = score - args.smooth_weight * torch.linalg.norm(accel, dim=-1).mean(dim=1)
    if args.joint_limit_weight:
        score = score - args.joint_limit_weight * action_limit_penalty(
            actions,
            action_min,
            action_max,
            args.joint_margin_frac,
        )
    return score


def refine_actions(args, reranker, scorer_name, scorer, actions_raw, data, vae_mean, vae_std, action_min, action_max):
    original = actions_raw.detach()
    current = original.clone()
    logs = []
    for step_idx in range(args.refine_steps):
        x = current.detach().clone().requires_grad_(True)
        score = objective_score(args, reranker, scorer_name, scorer, x, data, vae_mean, vae_std, action_min, action_max)
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
            scorer_name,
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


def score_np(args, reranker, scorer_name, scorer, actions, data, vae_mean, vae_std, action_min, action_max):
    score = objective_score(
        args,
        reranker,
        scorer_name,
        scorer,
        actions.detach().clone().requires_grad_(True),
        data,
        vae_mean,
        vae_std,
        action_min,
        action_max,
    )
    return score.detach().cpu().numpy()


def run_one(args, reranker, scorer_name, scorer, frames, vae_mean, vae_std, action_min, action_max):
    action_min_np = None if action_min is None else action_min.detach().cpu().numpy()
    action_max_np = None if action_max is None else action_max.detach().cpu().numpy()
    accum = {
        k: []
        for k in [
            "base_score",
            "refined_score",
            "score_delta",
            "base_smooth",
            "refined_smooth",
            "delta_norm",
            "base_barrier",
            "refined_barrier",
            "base_hard_violation",
            "refined_hard_violation",
        ]
    }
    accept_rates = []
    grad_norms = []
    rows = []
    for hdf5_path, ep_name, t in tqdm(frames, desc=f"{scorer_name} insertion clean-refine"):
        try:
            torch.manual_seed(args.seed + len(rows))
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            obs_cond = reranker.build_obs_cond(data["images_obs"], data["qpos_obs"], data["marker_hists"])
            base_raw = reranker.generate_candidates(obs_cond, K=args.K)
            refined_raw, logs = refine_actions(
                args,
                reranker,
                scorer_name,
                scorer,
                base_raw,
                data,
                vae_mean,
                vae_std,
                action_min,
                action_max,
            )
            base_score = score_np(args, reranker, scorer_name, scorer, base_raw, data, vae_mean, vae_std, action_min, action_max)
            refined_score = score_np(
                args,
                reranker,
                scorer_name,
                scorer,
                refined_raw,
                data,
                vae_mean,
                vae_std,
                action_min,
                action_max,
            )
            base_np = base_raw.detach().cpu().numpy()
            refined_np = refined_raw.detach().cpu().numpy()
            delta_norm = np.linalg.norm((refined_np - base_np).reshape(args.K, -1), axis=1)
            base_smooth = action_smoothness_np(base_np)
            refined_smooth = action_smoothness_np(refined_np)
            base_barrier = action_limit_penalty_np(base_np, action_min_np, action_max_np, args.joint_margin_frac)
            refined_barrier = action_limit_penalty_np(refined_np, action_min_np, action_max_np, args.joint_margin_frac)
            base_hard = hard_range_violation_np(base_np, action_min_np, action_max_np)
            refined_hard = hard_range_violation_np(refined_np, action_min_np, action_max_np)
            values = {
                "base_score": base_score,
                "refined_score": refined_score,
                "score_delta": refined_score - base_score,
                "base_smooth": base_smooth,
                "refined_smooth": refined_smooth,
                "delta_norm": delta_norm,
                "base_barrier": base_barrier,
                "refined_barrier": refined_barrier,
                "base_hard_violation": base_hard,
                "refined_hard_violation": refined_hard,
            }
            for key, value in values.items():
                accum[key].append(value)
            accept_rates.extend([x["accept_rate"] for x in logs])
            grad_norms.extend([x["grad_norm_mean"] for x in logs])
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "base_score_mean": float(base_score.mean()),
                    "refined_score_mean": float(refined_score.mean()),
                    "score_delta_mean": float((refined_score - base_score).mean()),
                    "refined_beats_base_rate": float(np.mean(refined_score > base_score)),
                    "delta_norm_mean": float(delta_norm.mean()),
                    "refined_hard_range_violation_max": float(refined_hard.max()),
                    "n_refine_steps": len(logs),
                }
            )
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")
    if not rows:
        raise RuntimeError(f"No valid rows for {scorer_name}")
    merged = {key: np.concatenate(value, axis=0) for key, value in accum.items()}
    improved_rate = float(np.mean(merged["score_delta"] > 0))
    return {
        "scorer": scorer_name,
        "n_frames": len(rows),
        "n_action_samples": int(len(merged["score_delta"])),
        "summary": {
            "base_score": summarize(merged["base_score"]),
            "refined_score": summarize(merged["refined_score"]),
            "score_delta": summarize(merged["score_delta"]),
            "refined_beats_base_rate": improved_rate,
            "base_smoothness": summarize(merged["base_smooth"]),
            "refined_smoothness": summarize(merged["refined_smooth"]),
            "smoothness_delta": summarize(merged["refined_smooth"] - merged["base_smooth"]),
            "action_delta_norm": summarize(merged["delta_norm"]),
            "base_limit_barrier": summarize(merged["base_barrier"]),
            "refined_limit_barrier": summarize(merged["refined_barrier"]),
            "limit_barrier_delta": summarize(merged["refined_barrier"] - merged["base_barrier"]),
            "base_hard_range_violation": summarize(merged["base_hard_violation"]),
            "refined_hard_range_violation": summarize(merged["refined_hard_violation"]),
            "accept_rate_per_step": summarize(accept_rates),
            "grad_norm_mean_per_step": summarize(grad_norms),
        },
        "passes_insertion_clean_refine_smoke": bool(
            improved_rate > args.pass_improved_rate
            and float(np.mean(merged["score_delta"])) > 0
            and float(np.percentile(merged["delta_norm"], 95)) <= args.max_total_delta + 1e-5
            and float(np.percentile(merged["refined_hard_violation"], 95)) <= 1e-5
        ),
        "rows": rows,
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Insertion Distilled Clean-Refine Comparison",
        "",
        "Offline insertion DP/Foresight clean-action trust-region comparison. This is not a robot rollout.",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- n_frames: `{result['n_frames']}`",
        "",
        "| scorer | pass | improved | score delta mean | smooth delta mean | action delta p95 | range violation p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["insertion_risk", "distilled_energy"]:
        row = result[key]
        s = row["summary"]
        lines.append(
            f"| {key} | {row['passes_insertion_clean_refine_smoke']} | "
            f"{s['refined_beats_base_rate']:.4f} | {s['score_delta']['mean']:.6f} | "
            f"{s['smoothness_delta']['mean']:.6f} | {s['action_delta_norm']['p95']:.6f} | "
            f"{s['refined_hard_range_violation']['p95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Scores are scorer-internal and not directly comparable by absolute scale across scorers.",
            "- This gate checks whether each scorer can provide useful full-chain action gradients under the same trust region.",
            "- Passing this gate promotes a scorer for real rollout ablation; it does not prove real task improvement.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    action_min, action_max = get_action_limits(reranker)
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

    insertion = InsertionRiskScorerRuntime(args.insertion_ckpt, device=args.device)
    distilled = DistilledTacQualityEnergyRuntime(args.distilled_ckpt, device=args.device)
    freeze(insertion)
    freeze(distilled)

    insertion_result = run_one(args, reranker, "insertion_risk", insertion, frames, vae_mean, vae_std, action_min, action_max)
    distilled_result = run_one(args, reranker, "distilled_energy", distilled, frames, vae_mean, vae_std, action_min, action_max)
    result = {
        "purpose": "Compare insertion default risk scorer and distilled TacQualityEnergy in clean-action refinement.",
        "scope": "Offline insertion DP/Foresight clean-action trust-region chain; not robot rollout.",
        "config": vars(args),
        "n_frames": len(frames),
        "insertion_risk": insertion_result,
        "distilled_energy": distilled_result,
        "overall_pass": bool(
            insertion_result["passes_insertion_clean_refine_smoke"]
            and distilled_result["passes_insertion_clean_refine_smoke"]
        ),
        "recommendation": (
            "Keep InsertionRiskScorerRuntime as insertion default; promote DistilledTacQualityEnergyRuntime "
            "as cross-task ablation candidate if both pass."
        ),
    }
    json_path = out_dir / "insertion_distilled_clean_refine_comparison.json"
    md_path = out_dir / "insertion_distilled_clean_refine_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "insertion_risk": {
                    "pass": insertion_result["passes_insertion_clean_refine_smoke"],
                    "improved": insertion_result["summary"]["refined_beats_base_rate"],
                    "delta": insertion_result["summary"]["score_delta"]["mean"],
                },
                "distilled_energy": {
                    "pass": distilled_result["passes_insertion_clean_refine_smoke"],
                    "improved": distilled_result["summary"]["refined_beats_base_rate"],
                    "delta": distilled_result["summary"]["score_delta"]["mean"],
                },
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--insertion_ckpt", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
    parser.add_argument("--distilled_ckpt", default="/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0414")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--n_eval", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--insertion_score_mode", default="energy_clipped", choices=["quality", "log_p_good", "risk_guidance", "energy", "energy_clipped"])
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--joint_limit_weight", type=float, default=0.0)
    parser.add_argument("--joint_margin_frac", type=float, default=0.03)
    parser.add_argument("--refine_steps", type=int, default=4)
    parser.add_argument("--refine_step_size", type=float, default=0.02)
    parser.add_argument("--max_total_delta", type=float, default=0.08)
    parser.add_argument("--pass_improved_rate", type=float, default=0.95)
    parser.add_argument("--output_dir", default=str(OUT_DIR / "n24_k4"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
