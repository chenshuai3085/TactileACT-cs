#!/usr/bin/env python3
"""Audit TacQuality guidance robustness around noisy action chunks.

This is an intermediate check between clean-action gradient audits and true
DDPM-step classifier guidance.  It perturbs recorded action chunks by several
noise levels, then checks whether the deployed-style trust-region update still
provides finite gradients and improves the TacQuality score.

It is not a real rollout metric and does not prove robot improvement.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.eval_guidance_gradient_audit import (  # noqa: E402
    build_bridge,
    load_foresight,
    load_scorer,
    refiner_config,
    sample_episode_windows,
    score_from_prediction,
)
from TFAC_V5.tac_quality_energy.serving_guidance import ActionNormalizer, load_rollout_arm_config  # noqa: E402
from TFAC_V5.tac_quality_energy.trust_region import TacQualityTrustRegionRefiner  # noqa: E402


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit")
DEFAULT_BOARD_DATASET = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609")
DEFAULT_BOARD_FORESIGHT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload"
)
DEFAULT_BOARD_FORESIGHT_CKPT = DEFAULT_BOARD_FORESIGHT_DIR / "foresight_best.ckpt"
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json"
)


def summarize_np(values: List[float] | np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _task_id(task: str) -> int:
    return 1 if task == "board" else 0


def _noise_like_action(
    clean_action: torch.Tensor,
    fs_norm: Mapping[str, Any],
    level: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if level <= 0:
        return clean_action.clone()
    std = torch.as_tensor(
        fs_norm["action_std"],
        dtype=clean_action.dtype,
        device=clean_action.device,
    ).view(1, 1, -1)
    noise = torch.randn(
        clean_action.shape,
        dtype=clean_action.dtype,
        device=clean_action.device,
        generator=generator,
    )
    return clean_action + float(level) * std * noise


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    rollout = load_rollout_arm_config(Path(args.rollout_arm_config))
    arm = rollout["tasks"][args.task][args.arm]
    scorer_runtime = args.scorer_runtime or arm["scorer_runtime"]
    scorer_checkpoint = args.scorer_checkpoint or arm["checkpoint"]["path"]
    score_mode = str(args.score_mode or arm.get("refiner", {}).get("score_mode", "energy_clipped"))
    profile_energy = arm.get("refiner", {}).get("energy", {})

    scorer = load_scorer(scorer_runtime, scorer_checkpoint, str(device))
    refiner = TacQualityTrustRegionRefiner(refiner_config(arm))
    foresight, fs_cfg, fs_norm, fs_info = load_foresight(Path(args.foresight_dir), Path(args.foresight_ckpt), device)
    windows = sample_episode_windows(args, fs_cfg)
    normalizer = ActionNormalizer(mode="identity")
    task_id_value = _task_id(args.task)
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    rows: List[Dict[str, Any]] = []
    for sample_idx, row in enumerate(windows):
        if args.progress:
            print(f"[noisy-audit] sample {sample_idx + 1}/{len(windows)}", flush=True)
        clean_action = torch.tensor(row["action"], dtype=torch.float32, device=device).unsqueeze(0)
        qpos_raw = torch.tensor(row["qpos"], dtype=torch.float32, device=device).view(1, -1)
        marker_raw = torch.tensor(row["marker_window"], dtype=torch.float32, device=device).unsqueeze(0)
        foresight_images = [img.to(device).unsqueeze(0) for img in row.get("foresight_images", [])]
        if clean_action.shape[1] < args.action_horizon:
            continue
        if clean_action.shape[1] < args.action_chunk:
            pad = clean_action[:, -1:].expand(-1, args.action_chunk - clean_action.shape[1], -1)
            clean_action = torch.cat([clean_action, pad], dim=1)

        bridge = build_bridge(foresight, fs_norm, fs_cfg, qpos_raw, marker_raw, foresight_images, args.task)
        task_id = torch.full((1,), task_id_value, dtype=torch.long, device=device)

        def score_fn(candidate_raw: torch.Tensor) -> torch.Tensor:
            tactile = bridge(candidate_raw)
            return score_from_prediction(scorer, score_mode, profile_energy, tactile, candidate_raw, task_id)

        with torch.enable_grad():
            clean_base = normalizer.denormalize(clean_action[:, : args.action_horizon])
            clean_score = score_fn(clean_base.detach().clone().requires_grad_(True)).detach()
            for noise_level in noise_levels:
                if args.progress:
                    print(f"[noisy-audit]   noise={noise_level}", flush=True)
                noisy_action = _noise_like_action(clean_base, fs_norm, noise_level, generator)
                noisy_score = score_fn(noisy_action.detach().clone().requires_grad_(True)).detach()
                guided, report = refiner.refine(noisy_action, score_fn)
                final_score = report["final_score"]["mean"]
                noisy_score_mean = float(noisy_score.mean().cpu())
                clean_score_mean = float(clean_score.mean().cpu())
                row_report = {
                    "sample": int(sample_idx),
                    "episode": row["path"],
                    "start": int(row["start"]),
                    "noise_level_action_std": float(noise_level),
                    "clean_score": clean_score_mean,
                    "noisy_score": noisy_score_mean,
                    "final_score": float(final_score),
                    "noisy_minus_clean": noisy_score_mean - clean_score_mean,
                    "final_minus_noisy": float(final_score) - noisy_score_mean,
                    "final_minus_clean": float(final_score) - clean_score_mean,
                    "finite_grad_rate": float(report["finite_grad_rate"]),
                    "positive_grad_rate": float(report["positive_grad_rate"]),
                    "accept_rate": float(report["accept_rate"]),
                    "improved_rate": float(report["improved_rate"]),
                    "max_delta_within_trust_region": bool(report["max_delta_within_trust_region"]),
                    "action_delta_norm": float(report["delta_norm"]["mean"]),
                    "noise_delta_norm": float((noisy_action - clean_base).flatten(1).norm(dim=1).mean().detach().cpu()),
                    "guided_action_delta_from_clean": float((guided - clean_base).flatten(1).norm(dim=1).mean().detach().cpu()),
                }
                rows.append(row_report)

    by_level: Dict[str, Dict[str, Any]] = {}
    for level in noise_levels:
        level_rows = [r for r in rows if abs(r["noise_level_action_std"] - level) < 1e-12]
        if not level_rows:
            continue
        by_level[str(level)] = {
            "n": len(level_rows),
            "clean_score": summarize_np([r["clean_score"] for r in level_rows]),
            "noisy_score": summarize_np([r["noisy_score"] for r in level_rows]),
            "final_score": summarize_np([r["final_score"] for r in level_rows]),
            "noisy_minus_clean": summarize_np([r["noisy_minus_clean"] for r in level_rows]),
            "final_minus_noisy": summarize_np([r["final_minus_noisy"] for r in level_rows]),
            "final_minus_clean": summarize_np([r["final_minus_clean"] for r in level_rows]),
            "finite_grad_rate_mean": float(np.mean([r["finite_grad_rate"] for r in level_rows])),
            "positive_grad_rate_mean": float(np.mean([r["positive_grad_rate"] for r in level_rows])),
            "accept_rate_mean": float(np.mean([r["accept_rate"] for r in level_rows])),
            "improved_rate_mean": float(np.mean([r["improved_rate"] for r in level_rows])),
            "trust_region_pass_rate": float(np.mean([r["max_delta_within_trust_region"] for r in level_rows])),
            "score_improve_rate": float(np.mean([r["final_minus_noisy"] > 0.0 for r in level_rows])),
            "score_not_worse_than_clean_rate": float(np.mean([r["final_minus_clean"] >= 0.0 for r in level_rows])),
            "action_delta_norm": summarize_np([r["action_delta_norm"] for r in level_rows]),
            "noise_delta_norm": summarize_np([r["noise_delta_norm"] for r in level_rows]),
        }

    pass_by_level = {
        level: bool(
            stats["finite_grad_rate_mean"] >= args.min_finite_grad_rate
            and stats["positive_grad_rate_mean"] >= args.min_positive_grad_rate
            and stats["trust_region_pass_rate"] >= args.min_trust_region_pass_rate
            and stats["score_improve_rate"] >= args.min_score_improve_rate
        )
        for level, stats in by_level.items()
    }
    result = {
        "purpose": "Noisy-action TacQuality guidance robustness audit through Foresight.",
        "evidence_boundary": (
            "This verifies local score-gradient robustness around perturbed action chunks. "
            "It does not prove true DDPM-step guidance or real robot improvement."
        ),
        "task": args.task,
        "arm": args.arm,
        "device": str(device),
        "dataset_dir": args.dataset_dir,
        "noise_levels_action_std": noise_levels,
        "n_rows": len(rows),
        "n_samples": len({r["sample"] for r in rows}),
        "foresight": {"dir": args.foresight_dir, "ckpt": args.foresight_ckpt, **fs_info},
        "scorer": {
            "runtime": scorer_runtime,
            "checkpoint": scorer_checkpoint,
            "score_mode": score_mode,
            "profile_energy": profile_energy,
        },
        "thresholds": {
            "min_finite_grad_rate": args.min_finite_grad_rate,
            "min_positive_grad_rate": args.min_positive_grad_rate,
            "min_trust_region_pass_rate": args.min_trust_region_pass_rate,
            "min_score_improve_rate": args.min_score_improve_rate,
        },
        "by_noise_level": by_level,
        "pass_by_noise_level": pass_by_level,
        "overall_pass": bool(rows and all(pass_by_level.values())),
        "rows": rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "noisy_action_guidance_audit.json"
    md_path = out_dir / "noisy_action_guidance_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "overall_pass": result["overall_pass"]}, indent=2))
    return result


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Noisy-Action TacQuality Guidance Audit",
        "",
        "This audit perturbs recorded action chunks and checks whether TacQuality trust-region guidance still improves score.",
        "",
        "## Setup",
        "",
        f"- task: `{result['task']}`",
        f"- arm: `{result['arm']}`",
        f"- samples: `{result['n_samples']}`",
        f"- rows: `{result['n_rows']}`",
        f"- scorer: `{result['scorer']['runtime']}`",
        f"- score mode: `{result['scorer']['score_mode']}`",
        f"- foresight: `{result['foresight']['dir']}`",
        f"- overall pass: `{result['overall_pass']}`",
        "",
        "## By Noise Level",
        "",
        "| action noise std scale | pass | n | score improve rate | finite grad | positive grad | trust-region | noisy-clean | final-noisy | final-clean |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level, stats in result["by_noise_level"].items():
        lines.append(
            f"| {level} | {result['pass_by_noise_level'].get(level)} | {stats['n']} | "
            f"{fmt(stats['score_improve_rate'])} | {fmt(stats['finite_grad_rate_mean'])} | "
            f"{fmt(stats['positive_grad_rate_mean'])} | {fmt(stats['trust_region_pass_rate'])} | "
            f"{fmt(stats['noisy_minus_clean']['mean'])} | {fmt(stats['final_minus_noisy']['mean'])} | "
            f"{fmt(stats['final_minus_clean']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Evidence Boundary",
            "",
            "- This is stronger than a clean-action-only gradient audit because it tests perturbed action chunks.",
            "- It is still not true DDPM-step guidance; the DP sampler is not modified here.",
            "- It does not prove real robot improvement; matched baseline/guided rollouts remain required.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["board", "insertion"], default="board")
    parser.add_argument("--arm", default="marker_joint_guided")
    parser.add_argument("--dataset_dir", default=str(DEFAULT_BOARD_DATASET))
    parser.add_argument("--foresight_dir", default=str(DEFAULT_BOARD_FORESIGHT_DIR))
    parser.add_argument("--foresight_ckpt", default=str(DEFAULT_BOARD_FORESIGHT_CKPT))
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--scorer_runtime", default=None)
    parser.add_argument("--scorer_checkpoint", default=None)
    parser.add_argument("--score_mode", default=None)
    parser.add_argument("--noise_levels", default="0,0.05,0.1,0.2,0.4")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--action_key", default="actions/joint_abs")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--max_episodes", type=int, default=4)
    parser.add_argument("--samples_per_episode", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_trust_region_pass_rate", type=float, default=0.999)
    parser.add_argument("--min_score_improve_rate", type=float, default=0.80)
    parser.add_argument("--torch_num_threads", type=int, default=4)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
