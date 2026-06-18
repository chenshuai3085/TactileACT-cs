#!/usr/bin/env python3
"""Insertion DDPM-step TacQuality guidance sweep over real 0401 episodes.

This is the insertion counterpart of the board DDPM-step sweep.  It evaluates
late-step classifier/energy guidance over multiple recorded observations while
using the matched 0401 Foresight checkpoint.  It is offline sampler evidence
only and does not prove real insertion improvement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from for_show_xiaomi.serve_dp_tac_quality_guided import GuidedDPStack  # noqa: E402
from TFAC_V5.tac_quality_energy.eval_ddpm_step_guidance_audit import (  # noqa: E402
    load_episode_obs,
    run_sample,
    summarize,
)
from TFAC_V5.tac_quality_energy.sweep_board_ddpm_step_guidance import (  # noqa: E402
    choose_contact_starts,
    episode_id_from_path,
    flatten_row,
    group_summary,
    has_required_keys,
    mean_by_key,
    write_csv,
)


DEFAULT_DATASET = Path("/home/chenshuai/data/dataset/260401_k14_truncated")
DEFAULT_DP_RUN = Path("/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210")
DEFAULT_VAE = Path("/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
DEFAULT_FORESIGHT_DIR = Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401")
DEFAULT_FORESIGHT_CKPT = DEFAULT_FORESIGHT_DIR / "foresight_best.ckpt"
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_marker_joint_20260618.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "insertion_0401_default_multiep8_start2_seed2_t0_s001"
)


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


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Insertion DDPM-Step TacQuality Guidance Sweep",
        "",
        f"- evidence boundary: {result['evidence_boundary']}",
        f"- dataset: `{result['inputs']['dataset_dir']}`",
        f"- DP ckpt: `{result['inputs']['ckpt_dir']}/{result['inputs']['ckpt_name']}`",
        f"- Foresight: `{result['inputs']['foresight_ckpt']}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| eval points | {summary['n_points']} |",
        f"| rows | {summary['n_rows']} |",
        f"| skipped episodes | {summary['n_skipped_episodes']} |",
        f"| final improve rate | {summary['final_score_improve_rate']:.4f} |",
        f"| final score delta mean | {summary['final_score_delta']['mean']:.6f} |",
        f"| final score delta min | {summary['final_score_delta']['min']:.6f} |",
        f"| finite grad mean | {summary['finite_grad_rate']['mean']:.4f} |",
        f"| accept rate mean | {summary['accept_rate']['mean']:.4f} |",
        f"| final accept rate mean | {summary['final_accept_rate']['mean']:.4f} |",
        f"| action delta norm mean | {summary['guided_action_delta_norm']['mean']:.6f} |",
        "",
        "## By Episode",
        "",
        "| episode | n | improve | score delta mean | action delta mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_episode"]:
        lines.append(
            f"| {row['episode_id']} | {row['n']} | {row['improve_rate']:.4f} | "
            f"{row['final_score_delta']['mean']:.6f} | "
            f"{row['action_delta_norm']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This sweep tests whether the insertion risk scorer remains a finite and score-improving local gradient source inside the DP sampler.",
            "- It uses matched `latent_foresight_0401` with `0 missing / 0 unexpected` loading behavior.",
            "- It is offline sampler evidence only; real success/bounce/retry claims still require paired robot rollouts.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in str(value).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["insertion"], default="insertion")
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
    parser.add_argument("--max_delta_norm", type=float, default=0.02)
    parser.add_argument("--sample_clip", type=float, default=1.0)
    parser.add_argument("--disable_accept_only", action="store_true")
    parser.add_argument("--disable_final_accept_only", action="store_true")
    parser.add_argument("--dp_norm_mode", choices=["minmax", "standard", "identity"], default="minmax")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--disable_guidance", action="store_true")
    parser.add_argument("--disable_contact_gate", action="store_true")
    parser.add_argument("--contact_gate_low", type=float, default=1.8)
    parser.add_argument("--contact_gate_high", type=float, default=2.3)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    args.disable_guidance = False
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
    detailed: list[dict[str, Any]] = []
    for point in points:
        obs_buffer, marker_buffer = load_episode_obs(
            stack,
            Path(args.dataset_dir),
            int(point["episode_id"]),
            int(point["start"]),
        )
        for seed in seeds:
            sample = run_sample(
                stack,
                obs_buffer,
                marker_buffer,
                seed=int(seed),
                guidance_steps=int(args.guidance_steps),
                guidance_scale=float(args.guidance_scale),
                max_delta_norm=float(args.max_delta_norm),
                sample_clip=float(args.sample_clip),
                accept_only_improved=not bool(getattr(args, "disable_accept_only", False)),
                final_accept_only=not bool(getattr(args, "disable_final_accept_only", False)),
            )
            flat = flatten_row(point, int(seed), sample)
            rows.append(flat)
            detailed.append({"point": point, "seed": int(seed), "sample": sample})
            print(
                f"ep={point['episode_id']} start={point['start']} seed={seed} "
                f"delta={flat['guided_minus_base_final_score']:.6f}"
            )

    final_deltas = [float(row["guided_minus_base_final_score"]) for row in rows]
    raw_final_deltas = [float(row["raw_guided_minus_base_final_score"]) for row in rows]
    return {
        "purpose": "Multi-episode insertion DDPM-step TacQuality guidance sweep.",
        "evidence_boundary": (
            "Offline sampler sweep only. It verifies local DDPM-step score behavior "
            "under recorded observations; it does not prove real robot improvement."
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
            "accept_only_improved": not bool(getattr(args, "disable_accept_only", False)),
            "final_accept_only": not bool(getattr(args, "disable_final_accept_only", False)),
            "seeds": seeds,
        },
        "summary": {
            "n_points": len(points),
            "n_rows": len(rows),
            "n_skipped_episodes": len(skipped),
            "final_score_delta": summarize(final_deltas),
            "raw_final_score_delta": summarize(raw_final_deltas),
            "final_score_improve_rate": float(np.mean([x > 0.0 for x in final_deltas])) if rows else 0.0,
            "finite_grad_rate": mean_by_key(rows, "finite_grad_rate"),
            "positive_grad_rate": mean_by_key(rows, "positive_grad_rate"),
            "accept_rate": mean_by_key(rows, "accept_rate"),
            "final_accept_rate": mean_by_key(rows, "final_accepted"),
            "guided_action_delta_norm": mean_by_key(rows, "guided_action_delta_norm"),
            "raw_guided_action_delta_norm": mean_by_key(rows, "raw_guided_action_delta_norm"),
            "per_step_score_delta_mean": mean_by_key(rows, "per_step_score_delta_mean"),
        },
        "by_episode": group_summary(rows, "episode_id"),
        "rows": rows,
        "details": detailed,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_sweep(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "insertion_ddpm_step_guidance_sweep.json"
    csv_path = out_dir / "insertion_ddpm_step_guidance_sweep_rows.csv"
    md_path = out_dir / "insertion_ddpm_step_guidance_sweep.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(csv_path, result["rows"])
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "n_rows": result["summary"]["n_rows"],
        "final_score_improve_rate": result["summary"]["final_score_improve_rate"],
        "final_score_delta_mean": result["summary"]["final_score_delta"].get("mean"),
    }, indent=2))


if __name__ == "__main__":
    main()
