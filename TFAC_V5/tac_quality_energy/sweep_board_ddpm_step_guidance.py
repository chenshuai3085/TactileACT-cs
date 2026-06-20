#!/usr/bin/env python3
"""Board DDPM-step TacQuality guidance sweep over real 260617 episodes.

This expands the single-frame smoke in ``eval_ddpm_step_guidance_audit.py`` to
multiple episodes, contact-phase start points, and random seeds while reusing a
single loaded DP/Foresight/scorer stack.  It is still offline evidence only:
positive score deltas here do not prove real robot improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from for_show_xiaomi.serve_dp_tac_quality_guided import GuidedDPStack  # noqa: E402
from TFAC_V5.tac_quality_energy.eval_ddpm_step_guidance_audit import (  # noqa: E402
    DEFAULT_BOARD_FORESIGHT_CKPT,
    DEFAULT_BOARD_FORESIGHT_DIR,
    DEFAULT_ROLLOUT_CONFIG,
    load_episode_obs,
    run_sample,
    summarize,
)


DEFAULT_DATASET = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617")
DEFAULT_DP_RUN = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_"
    "20260620_rerun"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "board_marker_joint_s12_260617_20260620_rerun_multiep_t0_s001"
)


def episode_id_from_path(path: Path) -> int:
    match = re.search(r"episode_(\d+)\.hdf5$", path.name)
    if not match:
        raise ValueError(f"Cannot parse episode id from {path}")
    return int(match.group(1))


def has_required_keys(path: Path, proprio_key: str, tac_side: str, camera_names: Sequence[str]) -> tuple[bool, str]:
    try:
        with h5py.File(path, "r") as f:
            required = [f"observations/{proprio_key}", f"observations/tac/{tac_side}/marker_offset"]
            required.extend(
                f"observations/images/{cam}" for cam in camera_names if cam != "gelsight"
            )
            for key in required:
                if key not in f:
                    return False, f"missing:{key}"
            length = int(f[f"observations/{proprio_key}"].shape[0])
            if length < 32:
                return False, f"too_short:{length}"
    except Exception as exc:
        return False, f"error:{type(exc).__name__}:{exc}"
    return True, "ok"


def marker_contact_metric(marker: np.ndarray) -> np.ndarray:
    mag = np.linalg.norm(marker.astype(np.float32), axis=-1)
    return mag.mean(axis=(1, 2))


def choose_contact_starts(
    path: Path,
    *,
    tac_side: str,
    min_start: int,
    max_start: int | None,
    n_starts: int,
    contact_quantile: float,
    contact_min: float,
) -> tuple[list[int], dict[str, Any]]:
    with h5py.File(path, "r") as f:
        marker = f[f"observations/tac/{tac_side}/marker_offset"][:]
    metric = marker_contact_metric(marker)
    valid = np.arange(len(metric))
    valid = valid[valid >= int(min_start)]
    if max_start is not None and max_start >= 0:
        valid = valid[valid <= int(max_start)]
    if valid.size == 0:
        return [], {"reason": "no_valid_indices", "length": int(len(metric))}

    threshold = max(float(np.quantile(metric[valid], contact_quantile)), float(contact_min))
    contact = valid[metric[valid] >= threshold]
    source = "contact_quantile"
    if contact.size < n_starts:
        threshold = float(contact_min)
        contact = valid[metric[valid] >= threshold]
        source = "contact_min"
    if contact.size < n_starts:
        contact = valid
        source = "all_valid"

    if contact.size <= n_starts:
        starts = [int(x) for x in contact]
    else:
        positions = np.linspace(0, contact.size - 1, n_starts)
        starts = [int(contact[int(round(pos))]) for pos in positions]
    starts = sorted(set(starts))
    return starts, {
        "reason": source,
        "length": int(len(metric)),
        "threshold": float(threshold),
        "metric_mean": float(metric.mean()),
        "metric_p50": float(np.percentile(metric, 50)),
        "metric_p90": float(np.percentile(metric, 90)),
        "chosen_metric": [float(metric[s]) for s in starts],
    }


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


def flatten_row(point: dict[str, Any], seed: int, sample: dict[str, Any]) -> dict[str, Any]:
    gate = sample.get("contact_gate", {})
    per_step = sample.get("per_step_score_delta", {})
    return {
        "episode_id": point["episode_id"],
        "start": point["start"],
        "seed": int(seed),
        "base_final_score": sample.get("base_final_score"),
        "guided_final_score": sample.get("guided_final_score"),
        "guided_minus_base_final_score": sample.get("guided_minus_base_final_score"),
        "raw_guided_minus_base_final_score": sample.get("raw_guided_minus_base_final_score"),
        "final_accepted": sample.get("final_accepted"),
        "guided_steps": sample.get("guided_steps"),
        "guided_action_delta_norm": sample.get("guided_action_delta_norm"),
        "raw_guided_action_delta_norm": sample.get("raw_guided_action_delta_norm"),
        "finite_grad_rate": sample.get("finite_grad_rate"),
        "positive_grad_rate": sample.get("positive_grad_rate"),
        "accept_rate": sample.get("accept_rate"),
        "per_step_score_delta_mean": per_step.get("mean"),
        "contact_gate_value": gate.get("contact_gate_value"),
        "contact_gate_metric": gate.get("contact_gate_metric"),
        "contact_gate_reason": gate.get("contact_gate_reason"),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_by_key(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    return summarize([float(row[key]) for row in rows if row.get(key) is not None])


def group_summary(rows: Sequence[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    out = []
    keys = sorted({row.get(group_key) for row in rows})
    for key in keys:
        group = [row for row in rows if row.get(group_key) == key]
        deltas = [float(row["guided_minus_base_final_score"]) for row in group]
        out.append(
            {
                group_key: key,
                "n": len(group),
                "improve_rate": float(np.mean([x > 0.0 for x in deltas])) if group else 0.0,
                "final_score_delta": summarize(deltas),
                "action_delta_norm": mean_by_key(group, "guided_action_delta_norm"),
                "final_accept_rate": mean_by_key(group, "final_accepted"),
                "accept_rate": mean_by_key(group, "accept_rate"),
                "contact_gate_value": mean_by_key(group, "contact_gate_value"),
            }
        )
    return out


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Board DDPM-Step TacQuality Guidance Sweep",
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
        f"| contact gate mean | {summary['contact_gate_value']['mean']:.4f} |",
        "",
        "## By Episode",
        "",
        "| episode | n | improve | score delta mean | action delta mean | gate mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_episode"]:
        lines.append(
            f"| {row['episode_id']} | {row['n']} | {row['improve_rate']:.4f} | "
            f"{row['final_score_delta']['mean']:.6f} | "
            f"{row['action_delta_norm']['mean']:.6f} | "
            f"{row['contact_gate_value']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This sweep tests whether late-step TacQuality gradients remain finite and locally score-improving across multiple real 260617 board observations.",
            "- It is a stronger offline check than a single-frame smoke, but it still does not prove real wiping improvement.",
            "- Real claims still require paired baseline/guided robot rollouts with server-side force traces.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_ints(value: str) -> list[int]:
    return [int(x) for x in str(value).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["board"], default="board")
    parser.add_argument("--arm", default="marker_joint_s12_guided")
    parser.add_argument("--ckpt_dir", default=str(DEFAULT_DP_RUN))
    parser.add_argument("--ckpt_name", default="dp_best.pth")
    parser.add_argument("--vae_checkpoint_override", default=None)
    parser.add_argument("--foresight_dir", default=str(DEFAULT_BOARD_FORESIGHT_DIR))
    parser.add_argument("--foresight_ckpt", default=str(DEFAULT_BOARD_FORESIGHT_CKPT))
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_CONFIG))
    parser.add_argument("--dataset_dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--max_episodes", type=int, default=6)
    parser.add_argument("--starts_per_episode", type=int, default=2)
    parser.add_argument("--min_start", type=int, default=32)
    parser.add_argument("--max_start", type=int, default=-1)
    parser.add_argument("--contact_quantile", type=float, default=0.60)
    parser.add_argument("--contact_min", type=float, default=1.8)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_steps", type=int, default=1)
    parser.add_argument("--guidance_scale", type=float, default=0.001)
    parser.add_argument("--max_delta_norm", type=float, default=0.01)
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
                f"delta={flat['guided_minus_base_final_score']:.6f} "
                f"gate={flat['contact_gate_value']:.3f}"
            )

    final_deltas = [float(row["guided_minus_base_final_score"]) for row in rows]
    raw_final_deltas = [float(row["raw_guided_minus_base_final_score"]) for row in rows]
    return {
        "purpose": "Multi-episode board DDPM-step TacQuality guidance sweep.",
        "evidence_boundary": (
            "Offline sampler sweep only. It verifies local DDPM-step score behavior "
            "under recorded observations; it does not prove real robot improvement."
        ),
        "inputs": {
            "dataset_dir": str(args.dataset_dir),
            "ckpt_dir": str(args.ckpt_dir),
            "ckpt_name": str(args.ckpt_name),
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
            "contact_gate_low": float(args.contact_gate_low),
            "contact_gate_high": float(args.contact_gate_high),
            "disable_contact_gate": bool(args.disable_contact_gate),
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
            "contact_gate_value": mean_by_key(rows, "contact_gate_value"),
            "contact_gate_metric": mean_by_key(rows, "contact_gate_metric"),
        },
        "by_episode": group_summary(rows, "episode_id"),
        "by_contact_gate_reason": group_summary(rows, "contact_gate_reason"),
        "rows": rows,
        "details": detailed,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = run_sweep(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "board_ddpm_step_guidance_sweep.json"
    csv_path = out_dir / "board_ddpm_step_guidance_sweep_rows.csv"
    md_path = out_dir / "board_ddpm_step_guidance_sweep.md"
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
