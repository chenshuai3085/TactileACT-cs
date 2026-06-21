#!/usr/bin/env python3
"""Audit force-aware board denoising-step guidance on real HDF5 windows.

This extends the single synthetic dry-run smoke:

    real HDF5 obs window -> DP obs_cond -> DP denoising with x0 guidance
        -> ForceAwareForesightGuidanceRuntime score/gradient report

It is still not a real robot rollout.  It verifies that the deployed serving
stack can apply force-aware TacQuality gradients inside DP denoising on multiple
real board windows, using real images, qpos, and tactile marker histories.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.pretrain_latent_foresight_multistep_force import (  # noqa: E402
    parse_dataset_roots,
    scan_labeled_episodes,
    split_episodes,
)
from for_show_xiaomi.serve_dp_tac_quality_guided import GuidedDPStack  # noqa: E402


DEFAULT_DP_DIR = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_"
    "20260621_codex"
)
DEFAULT_FORESIGHT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0"
)
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
DEFAULT_OUT_ROOT = Path("/home/chenshuai/Project/output/force_aware_denoising_real_window_audit")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def summary(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def h5_array(root: h5py.File, key: str) -> np.ndarray:
    if key in root:
        return root[key][()]
    if key.startswith("/") and key[1:] in root:
        return root[key[1:]][()]
    raise KeyError(key)


def marker_window(marker: np.ndarray, end_t: int, window: int) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for i in range(window):
        ts = max(0, end_t - (window - 1 - i))
        frames.append(marker[ts].astype(np.float32))
    return frames


def contact_metric(marker_frames: Sequence[np.ndarray]) -> float:
    marker_win = np.stack(marker_frames, axis=0)
    return float(np.linalg.norm(marker_win.reshape(marker_win.shape[0], -1, 2), axis=-1).mean())


def choose_starts(length: int, window: int, chunk: int, count: int, rng: np.random.Generator) -> np.ndarray:
    lo = max(window - 1, int(length * 0.20))
    hi = min(int(length * 0.95), length - chunk - 1)
    if hi <= lo:
        return np.asarray([], dtype=np.int64)
    candidates = np.arange(lo, hi, dtype=np.int64)
    if count > 0 and count < len(candidates):
        return np.sort(rng.choice(candidates, size=count, replace=False))
    return candidates


def scan_forceaware_episodes(forceaware_dir: Path, split: str) -> Tuple[List[Any], Dict[str, Any]]:
    cfg = load_json(forceaware_dir / "args.json")
    roots = parse_dataset_roots(cfg)
    episodes = scan_labeled_episodes(roots)
    train_eps, val_eps = split_episodes(episodes, float(cfg.get("train_ratio", 0.9)), int(cfg.get("seed", 42)))
    if split == "train":
        selected = train_eps
    elif split == "all":
        selected = episodes
    else:
        selected = val_eps
    return selected, {
        "all": len(episodes),
        "train": len(train_eps),
        "val": len(val_eps),
        "selected": len(selected),
        "split_seed": int(cfg.get("seed", 42)),
    }


def select_stratified_episodes(
    episodes: List[Any],
    max_episodes: int,
    rng: np.random.Generator,
) -> Tuple[List[Any], Dict[str, int]]:
    by_label: Dict[str, List[Any]] = {}
    for ref in episodes:
        by_label.setdefault(ref.label, []).append(ref)
    for refs in by_label.values():
        order = rng.permutation(len(refs))
        refs[:] = [refs[int(i)] for i in order]
    label_counts = {label: len(refs) for label, refs in sorted(by_label.items())}
    if max_episodes <= 0 or max_episodes >= len(episodes):
        selected = [ref for label in sorted(by_label) for ref in by_label[label]]
        return selected, label_counts

    selected: List[Any] = []
    cursors = {label: 0 for label in by_label}
    labels = sorted(by_label)
    while len(selected) < max_episodes:
        added = False
        for label in labels:
            if len(selected) >= max_episodes:
                break
            idx = cursors[label]
            refs = by_label[label]
            if idx < len(refs):
                selected.append(refs[idx])
                cursors[label] = idx + 1
                added = True
        if not added:
            break
    return selected, label_counts


def make_stack_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        task="board",
        arm=args.arm,
        ckpt_dir=str(args.dp_dir),
        ckpt_name=args.ckpt_name,
        vae_checkpoint_override=None,
        foresight_dir=str(args.foresight_dir),
        foresight_ckpt=str(args.foresight_ckpt),
        rollout_arm_config=str(args.rollout_config),
        gpu=args.gpu,
        scheduler=args.scheduler,
        num_inference_steps=args.num_inference_steps,
        action_horizon=8,
        action_skip=0,
        max_timesteps=300,
        seed=args.seed,
        dp_norm_mode=args.dp_norm_mode,
        no_ema=False,
        send_guidance_report=False,
        guidance_location="denoising_step",
        ddpm_guidance_steps=args.ddpm_guidance_steps,
        ddpm_guidance_scale=args.ddpm_guidance_scale,
        ddpm_max_delta_norm=args.ddpm_max_delta_norm,
        ddpm_sample_clip=args.ddpm_sample_clip,
        ddpm_min_grad_norm=args.ddpm_min_grad_norm,
        disable_ddpm_accept_only=False,
        disable_ddpm_x0_clip=False,
        server_rollout_log_dir=None,
        disable_server_rollout_log=True,
        disable_guidance=False,
        disable_contact_gate=args.disable_contact_gate,
        contact_gate_low=args.contact_gate_low,
        contact_gate_high=args.contact_gate_high,
        dry_run_guidance_smoke=False,
        synthetic_foresight_for_smoke=False,
        smoke_marker_value=3.0,
        smoke_output="",
        min_finite_grad_rate=args.min_finite_grad_rate,
        min_positive_grad_rate=args.min_positive_grad_rate,
    )


def build_obs_for_timestep(
    stack: GuidedDPStack,
    h5: h5py.File,
    t: int,
    marker: np.ndarray,
    qpos: np.ndarray,
    image_cache: Dict[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], Dict[str, Any]]:
    obs_buffer: deque[Dict[str, Any]] = deque(maxlen=stack.obs_horizon)
    marker_buffer: List[np.ndarray] = []
    processed_last: Dict[str, Any] | None = None
    marker_start = max(0, t - stack.tac_history - stack.obs_horizon)
    marker_buffer.extend([m.astype(np.float32) for m in marker[marker_start : t + 1]])

    for obs_i in range(stack.obs_horizon):
        obs_t = max(0, t - (stack.obs_horizon - 1 - obs_i))
        obs = {
            "images": {cam: image_cache[cam][obs_t] for cam in stack.camera_names if cam != "gelsight"},
            "qpos": qpos[obs_t],
            "tac": {"left": {"marker_offset": marker[obs_t]}},
        }
        processed = stack.preprocess_obs(obs)
        processed["_marker_idx"] = len(marker_buffer) - (t - obs_t) - 1
        obs_buffer.append(processed)
        processed_last = processed
    if processed_last is None:
        raise RuntimeError("empty obs buffer")
    return list(obs_buffer), marker_buffer, processed_last


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    summary_data = result["summary"]
    checks = result["checks"]
    lines = [
        "# Force-Aware Denoising Real-Window Audit",
        "",
        "Purpose: validate `force_aware_guided` denoising-step guidance on real board HDF5 windows.",
        "",
        "## Setup",
        f"- arm: `{result['setup']['arm']}`",
        f"- split: `{result['setup']['split']}`",
        f"- windows: `{result['setup']['num_windows']}`",
        f"- scheduler / steps: `{result['setup']['scheduler']}` / `{result['setup']['num_inference_steps']}`",
        f"- guided steps: `{result['setup']['ddpm_guidance_steps']}`",
        f"- rollout_config: `{result['setup']['rollout_config']}`",
        "",
        "## Checks",
        f"- pass: `{summary_data['pass']}`",
        f"- runtime_is_force_aware: `{checks['runtime_is_force_aware']}`",
        f"- adapter_policy_ok: `{checks['adapter_policy_ok']}`",
        f"- denoising_location_ok: `{checks['denoising_location_ok']}`",
        f"- not_reranking: `{checks['not_reranking']}`",
        "",
        "## Metrics",
        f"- finite_grad_rate_mean: `{summary_data['finite_grad_rate_mean']:.4f}`",
        f"- positive_grad_rate_mean: `{summary_data['positive_grad_rate_mean']:.4f}`",
        f"- accept_rate_mean: `{summary_data['accept_rate_mean']:.4f}`",
        f"- trust_region_pass_rate: `{summary_data['trust_region_pass_rate']:.4f}`",
        f"- score_delta_mean: `{summary_data['score_delta']['mean']:.6f}`",
        f"- normalized_action_delta_mean: `{summary_data['normalized_action_delta']['mean']:.6f}`",
        "",
        "## Evidence Boundary",
        result["evidence_boundary"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    stack = GuidedDPStack(make_stack_args(args))
    if not stack.uses_force_aware_guidance:
        raise RuntimeError("Expected force-aware guidance runtime")

    episodes, split_counts = scan_forceaware_episodes(args.forceaware_dir, args.split)
    selected, available_label_counts = select_stratified_episodes(episodes, args.max_episodes, rng)

    records: List[Dict[str, Any]] = []
    for ref in selected:
        try:
            with h5py.File(ref.path, "r") as f:
                qpos = h5_array(f, f"observations/{args.proprio_key}").astype(np.float32)
                marker = h5_array(f, f"observations/tac/{args.tac_side}/marker_offset").astype(np.float32)
                image_cache = {
                    cam: h5_array(f, f"observations/images/{cam}")
                    for cam in stack.camera_names
                    if cam != "gelsight"
                }
                length = min([len(qpos), len(marker)] + [len(v) for v in image_cache.values()])
                starts = choose_starts(length, stack.tac_history, stack.pred_horizon, args.samples_per_episode, rng)
                for start in starts:
                    marker_frames = marker_window(marker, int(start), stack.tac_history)
                    metric = contact_metric(marker_frames)
                    if metric < args.min_contact_metric:
                        continue
                    obs_buffer, marker_buffer, processed = build_obs_for_timestep(
                        stack,
                        f,
                        int(start),
                        marker,
                        qpos,
                        image_cache,
                    )
                    obs_cond = stack.build_obs_cond(obs_buffer, marker_buffer)
                    contact_gate = stack.contact_gate_report(marker_buffer)
                    with torch.inference_mode():
                        _, report = stack.ddpm_inference_with_force_aware_tac_guidance(
                            obs_cond,
                            processed,
                            marker_buffer,
                            contact_gate,
                        )
                    records.append(
                        {
                            "path": ref.path,
                            "label": ref.label,
                            "start": int(start),
                            "contact_metric": metric,
                            "score_delta": float(report["score_delta"]["mean"]),
                            "final_score": float(report["final_score"]["mean"]),
                            "finite_grad_rate": float(report["finite_grad_rate"]),
                            "positive_grad_rate": float(report["positive_grad_rate"]),
                            "accept_rate": float(report["accept_rate"]),
                            "trust_region_pass": bool(report["max_delta_within_trust_region"]),
                            "normalized_action_delta": float(report["normalized_action_delta"]["mean"]),
                            "adapter_policy": report.get("adapter_policy"),
                            "runtime": report.get("scorer_runtime"),
                            "guidance_location": report.get("guidance_location"),
                            "every_step_ddpm_guidance": bool(report.get("every_step_ddpm_guidance")),
                            "not_reranking": report.get("reranking") is False,
                            "guided_steps_executed": int(report.get("ddpm_guidance", {}).get("guided_steps_executed", 0)),
                            "contact_gate_value": float(report.get("contact_gate_value", 1.0)),
                        }
                    )
                    if args.max_windows > 0 and len(records) >= args.max_windows:
                        break
        except (OSError, KeyError, RuntimeError) as exc:
            if args.fail_on_episode_error:
                raise
            print(f"[warn] skip {ref.path}: {exc}", file=sys.stderr)
        if args.max_windows > 0 and len(records) >= args.max_windows:
            break

    if not records:
        raise RuntimeError("No real windows audited")

    finite = [r["finite_grad_rate"] for r in records]
    positive = [r["positive_grad_rate"] for r in records]
    accept = [r["accept_rate"] for r in records]
    trust = [1.0 if r["trust_region_pass"] else 0.0 for r in records]
    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "setup": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "arm": args.arm,
            "split": args.split,
            "split_counts": split_counts,
            "available_label_counts": available_label_counts,
            "selected_label_counts": {
                label: sum(1 for ref in selected if ref.label == label)
                for label in sorted(set(ref.label for ref in selected))
            },
            "num_windows": len(records),
            "max_episodes": args.max_episodes,
            "samples_per_episode": args.samples_per_episode,
            "scheduler": args.scheduler,
            "num_inference_steps": args.num_inference_steps,
            "ddpm_guidance_steps": args.ddpm_guidance_steps,
            "ddpm_guidance_scale": args.ddpm_guidance_scale,
            "ddpm_max_delta_norm": args.ddpm_max_delta_norm,
            "dp_dir": str(args.dp_dir),
            "ckpt_name": args.ckpt_name,
            "rollout_config": str(args.rollout_config),
            "forceaware_dir": str(args.forceaware_dir),
            "foresight_ckpt": str(args.foresight_ckpt),
            "device": str(stack.device),
        },
        "summary": {
            "pass": bool(
                np.mean(finite) >= args.min_finite_grad_rate
                and np.mean(positive) >= args.min_positive_grad_rate
                and np.mean(accept) >= args.min_accept_rate
                and np.mean(trust) >= args.min_trust_region_pass_rate
            ),
            "finite_grad_rate_mean": float(np.mean(finite)),
            "positive_grad_rate_mean": float(np.mean(positive)),
            "accept_rate_mean": float(np.mean(accept)),
            "trust_region_pass_rate": float(np.mean(trust)),
            "score_delta": summary(r["score_delta"] for r in records),
            "final_score": summary(r["final_score"] for r in records),
            "normalized_action_delta": summary(r["normalized_action_delta"] for r in records),
            "contact_metric": summary(r["contact_metric"] for r in records),
            "labels": {
                label: sum(1 for r in records if r["label"] == label)
                for label in sorted(set(r["label"] for r in records))
            },
        },
        "checks": {
            "runtime_is_force_aware": all(r["runtime"] == "ForceAwareForesightGuidanceRuntime" for r in records),
            "adapter_policy_ok": all(r["adapter_policy"] == "denoising_step_force_aware_tac_quality_guidance" for r in records),
            "denoising_location_ok": all(
                str(r["guidance_location"]).startswith("inside DP denoising loop") for r in records
            ),
            "every_step_ddpm_guidance": all(r["every_step_ddpm_guidance"] for r in records),
            "not_reranking": all(r["not_reranking"] for r in records),
            "guided_steps_executed_positive": all(r["guided_steps_executed"] > 0 for r in records),
        },
        "records": records,
        "evidence_boundary": (
            "This validates force-aware denoising-step guidance on real HDF5 windows "
            "through the serving DP stack. It is not an online real robot improvement claim."
        ),
        "paths": {
            "json": str(out_dir / "force_aware_denoising_real_window_audit.json"),
            "markdown": str(out_dir / "force_aware_denoising_real_window_audit.md"),
        },
    }
    save_json(result, out_dir / "force_aware_denoising_real_window_audit.json")
    write_markdown(result, out_dir / "force_aware_denoising_real_window_audit.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_dir", type=Path, default=DEFAULT_DP_DIR)
    parser.add_argument("--ckpt_name", default="dp_best.pth")
    parser.add_argument("--foresight_dir", type=Path, default=DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--foresight_ckpt", type=Path, default=DEFAULT_FORESIGHT_DIR / "foresight_force_best.ckpt")
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--forceaware_dir", type=Path, default=DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--arm", default="force_aware_guided")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max_episodes", type=int, default=4)
    parser.add_argument("--samples_per_episode", type=int, default=2)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--ddpm_guidance_steps", type=int, default=1)
    parser.add_argument("--ddpm_guidance_scale", type=float, default=0.001)
    parser.add_argument("--ddpm_max_delta_norm", type=float, default=0.01)
    parser.add_argument("--ddpm_sample_clip", type=float, default=1.0)
    parser.add_argument("--ddpm_min_grad_norm", type=float, default=1e-8)
    parser.add_argument("--dp_norm_mode", choices=["minmax", "standard", "identity"], default="minmax")
    parser.add_argument("--disable_contact_gate", action="store_true")
    parser.add_argument("--contact_gate_low", type=float, default=1.8)
    parser.add_argument("--contact_gate_high", type=float, default=2.3)
    parser.add_argument("--min_contact_metric", type=float, default=1.8)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_accept_rate", type=float, default=0.999)
    parser.add_argument("--min_trust_region_pass_rate", type=float, default=0.999)
    parser.add_argument("--fail_on_episode_error", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "json": result["paths"]["json"],
        "markdown": result["paths"]["markdown"],
        "summary": result["summary"],
        "checks": result["checks"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
