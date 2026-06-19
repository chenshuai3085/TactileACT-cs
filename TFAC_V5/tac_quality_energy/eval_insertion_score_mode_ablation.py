#!/usr/bin/env python3
"""Cross-score insertion guidance modes inside the DP sampler.

The previous p_good ablation showed that probability scores can saturate.
This script compares insertion score modes by using each mode as the DDPM-step
guidance objective, then rescoring the same base/guided final actions with all
candidate modes.  This avoids judging a mode only by the score it optimized.

The output is offline sampler evidence only; it does not prove real robot
success or bounce reduction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from for_show_xiaomi.serve_dp_tac_quality_guided import GuidedDPStack  # noqa: E402
from TFAC_V5.tac_quality_energy.eval_ddpm_step_guidance_audit import (  # noqa: E402
    load_episode_obs,
    predict_x0_from_eps,
    summarize,
    unit_update,
)
from TFAC_V5.tac_quality_energy.sweep_board_ddpm_step_guidance import (  # noqa: E402
    choose_contact_starts,
    episode_id_from_path,
    has_required_keys,
    write_csv as write_rows_csv,
)
from TFAC_V5.tac_quality_energy.sweep_insertion_ddpm_step_guidance import parse_ints  # noqa: E402


DEFAULT_DATASET = Path("/home/chenshuai/data/dataset/260401_k14_truncated")
DEFAULT_DP_RUN = Path("/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210")
DEFAULT_VAE = Path("/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
DEFAULT_FORESIGHT_DIR = Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401")
DEFAULT_FORESIGHT_CKPT = DEFAULT_FORESIGHT_DIR / "foresight_best.ckpt"
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_marker_joint_20260619_insertion_logit_ablation.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/chenshuai/Project/output/tac_quality_score_mode_ablation/"
    "insertion_0401_profile_pgood_energy_goodmargin_cross_score"
)
DEFAULT_GUIDANCE_MODES = "profile,p_good,energy,good_margin"
DEFAULT_EVAL_MODES = "profile,p_good,log_p_good,energy,good_margin,quality_logit,neg_risk"


def select_eval_points(
    stack: GuidedDPStack,
    dataset_dir: Path,
    *,
    max_episodes: int,
    starts_per_episode: int,
    min_start: int,
    max_start: int | None,
    contact_quantile: float,
    contact_min: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    paths = sorted(dataset_dir.glob("episode_*.hdf5"), key=episode_id_from_path)
    for path in paths:
        ep = episode_id_from_path(path)
        ok, reason = has_required_keys(
            path,
            str(stack.config.get("proprio_key", "proprio_joint")),
            str(stack.config.get("tac_side", "left")),
            stack.camera_names,
        )
        if not ok:
            skipped.append({"episode_id": ep, "path": str(path), "reason": reason})
            continue
        starts, info = choose_contact_starts(
            path,
            tac_side=str(stack.config.get("tac_side", "left")),
            min_start=min_start,
            max_start=max_start,
            n_starts=starts_per_episode,
            contact_quantile=contact_quantile,
            contact_min=contact_min,
        )
        if not starts:
            skipped.append({"episode_id": ep, "path": str(path), "reason": info.get("reason", "no_starts")})
            continue
        for start_idx, start in enumerate(starts):
            points.append(
                {
                    "episode_id": ep,
                    "start": int(start),
                    "start_index": int(start_idx),
                    "selection": info,
                }
            )
        if len({p["episode_id"] for p in points}) >= max_episodes:
            break
    return points, skipped


def score_x0_mode(stack: GuidedDPStack, x0_norm: torch.Tensor, bridge, mode: str) -> torch.Tensor:
    if stack.guidance is None:
        raise RuntimeError("TacQuality guidance must be enabled for scoring")
    old_mode = stack.guidance.adapter.score_mode
    stack.guidance.adapter.score_mode = mode
    try:
        action_raw = stack.guidance.adapter.action_normalizer.denormalize(x0_norm)
        tactile = bridge(action_raw)
        return stack.guidance.adapter.score_from_prediction(tactile, action_raw)
    finally:
        stack.guidance.adapter.score_mode = old_mode


def mean_score(stack: GuidedDPStack, x0_norm: torch.Tensor, bridge, mode: str) -> float:
    score = score_x0_mode(stack, x0_norm.detach().clone().requires_grad_(True), bridge, mode)
    return float(score.mean().detach().cpu())


def run_guided_sample(
    stack: GuidedDPStack,
    obs_buffer: deque[dict[str, Any]],
    marker_buffer: list[np.ndarray],
    *,
    seed: int,
    guidance_mode: str,
    eval_modes: list[str],
    guidance_steps: int,
    guidance_scale: float,
    max_delta_norm: float,
    sample_clip: float,
    accept_only_improved: bool,
    final_accept_only: bool,
) -> dict[str, Any]:
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

    def denoise(enable_guidance: bool) -> tuple[torch.Tensor, dict[str, Any]]:
        action = initial.detach().clone()
        guide_start = max(0, len(timesteps) - int(guidance_steps))
        logs: list[dict[str, Any]] = []
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
                score_before = score_x0_mode(stack, x0_before, bridge, guidance_mode)
                grad = torch.autograd.grad(score_before.mean(), action_for_grad, retain_graph=False)[0]
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
                    score_after = score_x0_mode(stack, x0_after, bridge, guidance_mode)
                    accepted = bool(
                        (not accept_only_improved)
                        or torch.all(score_after >= score_before.detach()).item()
                    )
                logs.append(
                    {
                        "step_idx": int(step_idx),
                        "timestep": int(t.item()),
                        "score_before": float(score_before.mean().detach().cpu()),
                        "score_after": float(score_after.mean().detach().cpu()),
                        "score_delta": float((score_after - score_before.detach()).mean().detach().cpu()),
                        "accepted": accepted,
                        "contact_gate_value": gate_value,
                        **grad_report,
                    }
                )
                if accepted:
                    action = guided_action.detach()
                    eps = eps_after.detach()
            with torch.no_grad():
                action = stack.noise_scheduler.step(eps, t, action).prev_sample.detach()
        return action.detach(), {"logs": logs}

    base_action, _ = denoise(enable_guidance=False)
    raw_guided_action, guided_info = denoise(enable_guidance=True)
    base_guidance_score = mean_score(stack, base_action, bridge, guidance_mode)
    raw_guided_guidance_score = mean_score(stack, raw_guided_action, bridge, guidance_mode)
    final_accepted = bool((not final_accept_only) or raw_guided_guidance_score >= base_guidance_score)
    guided_action = raw_guided_action if final_accepted else base_action
    guided_guidance_score = raw_guided_guidance_score if final_accepted else base_guidance_score

    cross_scores: dict[str, dict[str, float]] = {}
    for mode in eval_modes:
        base = mean_score(stack, base_action, bridge, mode)
        raw = mean_score(stack, raw_guided_action, bridge, mode)
        final = mean_score(stack, guided_action, bridge, mode)
        cross_scores[mode] = {
            "base": base,
            "raw_guided": raw,
            "guided": final,
            "raw_delta": raw - base,
            "delta": final - base,
        }

    logs = guided_info["logs"]
    action_delta = (guided_action - base_action).flatten(1).norm(dim=1)
    raw_action_delta = (raw_guided_action - base_action).flatten(1).norm(dim=1)
    return {
        "seed": int(seed),
        "guidance_mode": guidance_mode,
        "base_guidance_score": base_guidance_score,
        "raw_guided_guidance_score": raw_guided_guidance_score,
        "guided_guidance_score": guided_guidance_score,
        "guided_minus_base_guidance_score": guided_guidance_score - base_guidance_score,
        "raw_guided_minus_base_guidance_score": raw_guided_guidance_score - base_guidance_score,
        "final_accepted": final_accepted,
        "guided_steps": len(logs),
        "guided_action_delta_norm": float(action_delta.mean().detach().cpu()),
        "raw_guided_action_delta_norm": float(raw_action_delta.mean().detach().cpu()),
        "finite_grad_rate": float(np.mean([row["finite_grad_rate"] for row in logs])) if logs else 1.0,
        "positive_grad_rate": float(np.mean([row["grad_norm"] > 1e-8 for row in logs])) if logs else 1.0,
        "accept_rate": float(np.mean([row["accepted"] for row in logs])) if logs else 1.0,
        "per_step_score_delta": summarize([row["score_delta"] for row in logs]),
        "cross_scores": cross_scores,
        "logs": logs,
    }


def flatten_rows(rows: list[dict[str, Any]], eval_modes: list[str]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        base = {
            "episode_id": row["episode_id"],
            "start": row["start"],
            "seed": row["seed"],
            "guidance_mode": row["guidance_mode"],
            "final_accepted": row["final_accepted"],
            "guided_action_delta_norm": row["guided_action_delta_norm"],
            "raw_guided_action_delta_norm": row["raw_guided_action_delta_norm"],
            "accept_rate": row["accept_rate"],
            "finite_grad_rate": row["finite_grad_rate"],
        }
        for mode in eval_modes:
            scores = row["cross_scores"][mode]
            flat.append(
                {
                    **base,
                    "eval_mode": mode,
                    "base_score": scores["base"],
                    "guided_score": scores["guided"],
                    "delta": scores["delta"],
                    "raw_delta": scores["raw_delta"],
                }
            )
    return flat


def aggregate(rows: list[dict[str, Any]], guidance_modes: list[str], eval_modes: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for guidance_mode in guidance_modes:
        subset = [r for r in rows if r["guidance_mode"] == guidance_mode]
        mode_summary: dict[str, Any] = {
            "n_rows": len(subset),
            "final_accept_rate": float(np.mean([r["final_accepted"] for r in subset])) if subset else 0.0,
            "accept_rate": summarize([r["accept_rate"] for r in subset]),
            "finite_grad_rate": summarize([r["finite_grad_rate"] for r in subset]),
            "guided_action_delta_norm": summarize([r["guided_action_delta_norm"] for r in subset]),
            "cross_scores": {},
        }
        for eval_mode in eval_modes:
            deltas = [r["cross_scores"][eval_mode]["delta"] for r in subset]
            raw_deltas = [r["cross_scores"][eval_mode]["raw_delta"] for r in subset]
            mode_summary["cross_scores"][eval_mode] = {
                "delta": summarize(deltas),
                "raw_delta": summarize(raw_deltas),
                "improve_rate": float(np.mean([d > 0.0 for d in deltas])) if deltas else 0.0,
            }
        out[guidance_mode] = mode_summary
    return out


def render_markdown(result: dict[str, Any]) -> str:
    guidance_modes = result["guidance_modes"]
    eval_modes = result["eval_modes"]
    summary = result["summary"]
    lines = [
        "# Insertion Score-Mode Cross-Score Ablation",
        "",
        f"- evidence boundary: {result['evidence_boundary']}",
        f"- dataset: `{result['inputs']['dataset_dir']}`",
        f"- DP ckpt: `{result['inputs']['ckpt_dir']}/{result['inputs']['ckpt_name']}`",
        f"- Foresight: `{result['inputs']['foresight_ckpt']}`",
        "",
        "## Guidance-Mode Summary",
        "",
        "| guidance mode | rows | final accept | action norm | finite grad | own-score delta | own improve | profile delta | p_good delta | energy delta | good_margin delta | quality_logit delta | neg_risk delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in guidance_modes:
        s = summary[mode]
        own = s["cross_scores"][mode if mode in eval_modes else "profile"]
        def d(eval_mode: str) -> float:
            return s["cross_scores"][eval_mode]["delta"].get("mean", 0.0)
        lines.append(
            f"| {mode} | {s['n_rows']} | {s['final_accept_rate']:.4f} | "
            f"{s['guided_action_delta_norm'].get('mean', 0.0):.6f} | "
            f"{s['finite_grad_rate'].get('mean', 0.0):.4f} | "
            f"{own['delta'].get('mean', 0.0):.6f} | {own['improve_rate']:.4f} | "
            f"{d('profile'):.6f} | {d('p_good'):.6f} | {d('energy'):.6f} | "
            f"{d('good_margin'):.6f} | {d('quality_logit'):.6f} | {d('neg_risk'):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `p_good` is a bounded probability and can saturate; zero or tiny p_good delta does not necessarily mean the scorer has no useful logit geometry.",
            "- `good_margin` tests the unsaturated binary logit margin directly.",
            "- `energy` tests the existing unsaturated mixed quality/binary logit.",
            "- A deployable insertion guidance score should improve its own score without degrading profile/quality/risk semantics, and still requires real paired robot rollouts.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="default_guided")
    parser.add_argument("--ckpt_dir", default=str(DEFAULT_DP_RUN))
    parser.add_argument("--ckpt_name", default="dp_final.pth")
    parser.add_argument("--vae_checkpoint_override", default=str(DEFAULT_VAE))
    parser.add_argument("--foresight_dir", default=str(DEFAULT_FORESIGHT_DIR))
    parser.add_argument("--foresight_ckpt", default=str(DEFAULT_FORESIGHT_CKPT))
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_CONFIG))
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--max_episodes", type=int, default=8)
    parser.add_argument("--starts_per_episode", type=int, default=2)
    parser.add_argument("--min_start", type=int, default=32)
    parser.add_argument("--max_start", type=int, default=-1)
    parser.add_argument("--contact_quantile", type=float, default=0.60)
    parser.add_argument("--contact_min", type=float, default=1.5)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_steps", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.001)
    parser.add_argument("--max_delta_norm", type=float, default=0.005)
    parser.add_argument("--sample_clip", type=float, default=1.0)
    parser.add_argument("--dp_norm_mode", choices=["minmax", "standard", "identity"], default="minmax")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--disable_guidance", action="store_true")
    parser.add_argument("--disable_contact_gate", action="store_true")
    parser.add_argument("--contact_gate_low", type=float, default=1.8)
    parser.add_argument("--contact_gate_high", type=float, default=2.3)
    parser.add_argument("--guidance_modes", default=DEFAULT_GUIDANCE_MODES)
    parser.add_argument("--eval_modes", default=DEFAULT_EVAL_MODES)
    parser.add_argument("--disable_accept_only", action="store_true")
    parser.add_argument("--disable_final_accept_only", action="store_true")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.task = "insertion"
    args.disable_guidance = False
    guidance_modes = [x.strip() for x in str(args.guidance_modes).split(",") if x.strip()]
    eval_modes = [x.strip() for x in str(args.eval_modes).split(",") if x.strip()]

    stack = GuidedDPStack(args)
    stack.num_inference_steps = int(args.num_inference_steps)
    max_start = None if int(args.max_start) < 0 else int(args.max_start)
    points, skipped = select_eval_points(
        stack,
        Path(args.dataset_dir),
        max_episodes=int(args.max_episodes),
        starts_per_episode=int(args.starts_per_episode),
        min_start=int(args.min_start),
        max_start=max_start,
        contact_quantile=float(args.contact_quantile),
        contact_min=float(args.contact_min),
    )
    seeds = parse_ints(args.seeds)
    rows: list[dict[str, Any]] = []
    for point in points:
        obs_buffer, marker_buffer = load_episode_obs(
            stack,
            Path(args.dataset_dir),
            int(point["episode_id"]),
            int(point["start"]),
        )
        for seed in seeds:
            for guidance_mode in guidance_modes:
                sample = run_guided_sample(
                    stack,
                    obs_buffer,
                    marker_buffer,
                    seed=seed,
                    guidance_mode=guidance_mode,
                    eval_modes=eval_modes,
                    guidance_steps=int(args.guidance_steps),
                    guidance_scale=float(args.guidance_scale),
                    max_delta_norm=float(args.max_delta_norm),
                    sample_clip=float(args.sample_clip),
                    accept_only_improved=not bool(args.disable_accept_only),
                    final_accept_only=not bool(args.disable_final_accept_only),
                )
                row = {**point, **sample}
                rows.append(row)
                own_delta = sample["cross_scores"][guidance_mode]["delta"] if guidance_mode in eval_modes else sample["guided_minus_base_guidance_score"]
                print(
                    f"mode={guidance_mode} ep={point['episode_id']} start={point['start']} "
                    f"seed={seed} own_delta={own_delta:.6f}"
                )

    result = {
        "purpose": "Insertion score-mode ablation with cross-scoring across TacQuality heads.",
        "evidence_boundary": (
            "Offline sampler/cross-score evidence only. It verifies local score behavior "
            "under recorded observations and matched Foresight; it does not prove real robot improvement."
        ),
        "inputs": {
            "dataset_dir": str(args.dataset_dir),
            "ckpt_dir": str(args.ckpt_dir),
            "ckpt_name": str(args.ckpt_name),
            "vae_checkpoint_override": str(args.vae_checkpoint_override),
            "foresight_dir": str(args.foresight_dir),
            "foresight_ckpt": str(args.foresight_ckpt),
            "rollout_arm_config": str(args.rollout_arm_config),
        },
        "selection": {
            "max_episodes": int(args.max_episodes),
            "starts_per_episode": int(args.starts_per_episode),
            "min_start": int(args.min_start),
            "max_start": max_start,
            "contact_quantile": float(args.contact_quantile),
            "contact_min": float(args.contact_min),
            "points": points,
            "skipped": skipped,
        },
        "guidance": {
            "scheduler": args.scheduler,
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_steps": int(args.guidance_steps),
            "guidance_scale": float(args.guidance_scale),
            "max_delta_norm": float(args.max_delta_norm),
            "sample_clip": float(args.sample_clip),
            "accept_only_improved": not bool(args.disable_accept_only),
            "final_accept_only": not bool(args.disable_final_accept_only),
            "seeds": seeds,
        },
        "guidance_modes": guidance_modes,
        "eval_modes": eval_modes,
        "summary": aggregate(rows, guidance_modes, eval_modes),
        "rows": rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "insertion_score_mode_ablation.json"
    csv_path = out_dir / "insertion_score_mode_ablation_rows.csv"
    flat_csv_path = out_dir / "insertion_score_mode_ablation_cross_scores.csv"
    md_path = out_dir / "insertion_score_mode_ablation.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_rows_csv(csv_path, rows)
    write_rows_csv(flat_csv_path, flatten_rows(rows, eval_modes))
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "cross_score_csv": str(flat_csv_path),
                "markdown": str(md_path),
                "n_rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
