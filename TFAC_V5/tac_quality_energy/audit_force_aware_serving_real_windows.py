#!/usr/bin/env python3
"""Audit the force-aware board serving arm on real HDF5 windows.

This is the deployment-contract check between synthetic dry-run and real robot
rollout:

    real board HDF5 window -> build_serving_guidance_from_arm(force_aware_guided)
        -> guide_force_aware_action_chunk -> score/gradient/action-delta report

It uses the serving arm, DP action normalizer, and rollout config.  It does not
claim robot improvement because actions are replayed dataset future qpos chunks,
not online rollouts.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

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
from TFAC_V5.tac_quality_energy.serving_guidance import (  # noqa: E402
    build_serving_guidance_from_arm,
    load_rollout_arm_config,
)


DEFAULT_DP_CONFIG = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun/"
    "config.json"
)
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
DEFAULT_FORCE_AWARE_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_forceaware_multistep16_boardvae_e100_bs16_0"
)
DEFAULT_OUT_ROOT = Path("/home/chenshuai/Project/output/force_aware_serving_real_window_audit")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def pad_chunk(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(len(arr), start + length)
    chunk = arr[start:end]
    if len(chunk) <= 0:
        idx = max(0, min(start, len(arr) - 1))
        chunk = arr[idx : idx + 1]
    if len(chunk) < length:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], length - len(chunk), axis=0)], axis=0)
    return chunk.astype(np.float32)


def marker_window(marker: np.ndarray, end_t: int, window: int) -> np.ndarray:
    frames = []
    for i in range(window):
        ts = max(0, end_t - (window - 1 - i))
        frames.append(marker[ts])
    return np.stack(frames, axis=0).astype(np.float32)


def contact_metric(marker_win: np.ndarray) -> float:
    return float(np.linalg.norm(marker_win.reshape(marker_win.shape[0], -1, 2), axis=-1).mean())


def choose_starts(length: int, window: int, chunk: int, count: int, rng: np.random.Generator) -> np.ndarray:
    lo = max(window - 1, int(length * 0.2))
    hi = min(int(length * 0.95), length - chunk - 1)
    if hi <= lo:
        return np.asarray([], dtype=np.int64)
    candidates = np.arange(lo, hi, dtype=np.int64)
    if count > 0 and count < len(candidates):
        return np.sort(rng.choice(candidates, size=count, replace=False))
    return candidates


def scan_forceaware_val_episodes(forceaware_dir: Path, split: str) -> Tuple[List[Any], Dict[str, Any]]:
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
    """Select episodes while preserving label coverage where possible."""

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


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Force-Aware Serving Real-Window Audit",
        "",
        "Purpose: validate the `force_aware_guided` serving arm on real board HDF5 windows.",
        "",
        "## Setup",
        f"- arm: `{result['setup']['arm']}`",
        f"- split: `{result['setup']['split']}`",
        f"- windows: `{result['setup']['num_windows']}`",
        f"- rollout_config: `{result['setup']['rollout_config']}`",
        "",
        "## Metrics",
        f"- pass: `{result['summary']['pass']}`",
        f"- finite_grad_rate_mean: `{result['summary']['finite_grad_rate_mean']:.4f}`",
        f"- positive_grad_rate_mean: `{result['summary']['positive_grad_rate_mean']:.4f}`",
        f"- improved_rate_mean: `{result['summary']['improved_rate_mean']:.4f}`",
        f"- trust_region_pass_rate: `{result['summary']['trust_region_pass_rate']:.4f}`",
        f"- score_delta_mean: `{result['summary']['score_delta']['mean']:.6f}`",
        f"- raw_action_delta_mean: `{result['summary']['raw_action_delta']['mean']:.6f}`",
        "",
        "## Evidence Boundary",
        result["evidence_boundary"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    dp_cfg = load_json(args.dp_config)
    rollout_cfg = load_rollout_arm_config(args.rollout_config)
    guidance = build_serving_guidance_from_arm(
        "board",
        args.arm,
        dp_norm_stats=dp_cfg["norm_stats"],
        rollout_config=rollout_cfg,
        device=str(device),
        norm_mode=args.dp_norm_mode,
    )
    runtime_summary = {}
    if hasattr(guidance.adapter, "runtime") and hasattr(guidance.adapter.runtime, "summary"):
        runtime_summary = guidance.adapter.runtime.summary()
    episodes, split_counts = scan_forceaware_val_episodes(args.forceaware_dir, args.split)
    rng = np.random.default_rng(args.seed)
    selected, available_label_counts = select_stratified_episodes(episodes, args.max_episodes, rng)

    records: List[Dict[str, Any]] = []
    for ref in selected:
        try:
            with h5py.File(ref.path, "r") as f:
                qpos = h5_array(f, f"observations/{args.proprio_key}").astype(np.float32)
                marker = h5_array(f, f"observations/tac/{args.tac_side}/marker_offset").astype(np.float32)
        except (OSError, KeyError):
            continue
        length = min(len(qpos), len(marker))
        starts = choose_starts(length, args.window, args.chunk_size, args.samples_per_episode, rng)
        for start in starts:
            marker_win = marker_window(marker, int(start), args.window)
            if contact_metric(marker_win) < args.min_contact_metric:
                continue
            qpos_now = qpos[int(start)]
            action_raw = pad_chunk(qpos, int(start) + 1, args.chunk_size)
            action = torch.tensor(action_raw, dtype=torch.float32, device=device).unsqueeze(0)
            qpos_t = torch.tensor(qpos_now, dtype=torch.float32, device=device).view(1, -1)
            marker_t = torch.tensor(marker_win, dtype=torch.float32, device=device).unsqueeze(0)
            action_norm = guidance.adapter.action_normalizer.normalize(action)
            with torch.inference_mode():
                guided_norm, report = guidance.guide_force_aware_action_chunk(
                    action_norm,
                    qpos_raw=qpos_t,
                    marker_window_raw=marker_t,
                )
            records.append(
                {
                    "path": ref.path,
                    "label": ref.label,
                    "start": int(start),
                    "contact_metric": contact_metric(marker_win),
                    "score_delta": float(report["score_delta"]["mean"]),
                    "base_score": float(report["base_score"]["mean"]),
                    "final_score": float(report["final_score"]["mean"]),
                    "finite_grad_rate": float(report["finite_grad_rate"]),
                    "positive_grad_rate": float(report["positive_grad_rate"]),
                    "improved": float(report["improved_rate"]),
                    "accept_rate": float(report["accept_rate"]),
                    "trust_region_pass": bool(report["max_delta_within_trust_region"]),
                    "raw_action_delta": float(report["raw_action_delta"]["mean"]),
                    "normalized_action_delta": float(report["normalized_action_delta"]["mean"]),
                    "adapter_policy": report.get("adapter_policy"),
                    "runtime": report.get("scorer_runtime"),
                    "not_reranking": report.get("integration_contract", {}).get("reranking") is False,
                }
            )
            if args.max_windows > 0 and len(records) >= args.max_windows:
                break
        if args.max_windows > 0 and len(records) >= args.max_windows:
            break

    if not records:
        raise RuntimeError("No real windows audited")

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    finite = [r["finite_grad_rate"] for r in records]
    positive = [r["positive_grad_rate"] for r in records]
    improved = [r["improved"] for r in records]
    trust = [1.0 if r["trust_region_pass"] else 0.0 for r in records]
    result = {
        "setup": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "arm": args.arm,
            "device": str(device),
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
            "rollout_config": str(args.rollout_config),
            "dp_config": str(args.dp_config),
            "forceaware_dir": str(args.forceaware_dir),
            "score_weights": runtime_summary.get("score_weights", {}),
            "runtime_summary": runtime_summary,
        },
        "summary": {
            "pass": bool(
                np.mean(finite) >= args.min_finite_grad_rate
                and np.mean(positive) >= args.min_positive_grad_rate
                and np.mean(improved) >= args.min_improved_rate
                and np.mean(trust) >= args.min_trust_region_pass_rate
            ),
            "finite_grad_rate_mean": float(np.mean(finite)),
            "positive_grad_rate_mean": float(np.mean(positive)),
            "improved_rate_mean": float(np.mean(improved)),
            "accept_rate_mean": float(np.mean([r["accept_rate"] for r in records])),
            "trust_region_pass_rate": float(np.mean(trust)),
            "score_delta": summary(r["score_delta"] for r in records),
            "base_score": summary(r["base_score"] for r in records),
            "final_score": summary(r["final_score"] for r in records),
            "raw_action_delta": summary(r["raw_action_delta"] for r in records),
            "normalized_action_delta": summary(r["normalized_action_delta"] for r in records),
            "contact_metric": summary(r["contact_metric"] for r in records),
            "labels": {label: sum(1 for r in records if r["label"] == label) for label in sorted(set(r["label"] for r in records))},
        },
        "checks": {
            "runtime_is_force_aware": all(r["runtime"] == "ForceAwareForesightGuidanceRuntime" for r in records),
            "adapter_policy_ok": all(r["adapter_policy"] == "force_aware_foresight_trust_region_refinement" for r in records),
            "not_reranking": all(r["not_reranking"] for r in records),
        },
        "records": records,
        "evidence_boundary": (
            "This validates real HDF5 windows through the serving arm and DP action normalizer. "
            "It is still not a real online robot improvement claim."
        ),
        "paths": {
            "json": str(out_dir / "force_aware_serving_real_window_audit.json"),
            "markdown": str(out_dir / "force_aware_serving_real_window_audit.md"),
        },
    }
    save_json(result, out_dir / "force_aware_serving_real_window_audit.json")
    write_markdown(result, out_dir / "force_aware_serving_real_window_audit.md")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_config", type=Path, default=DEFAULT_DP_CONFIG)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--forceaware_dir", type=Path, default=DEFAULT_FORCE_AWARE_DIR)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--arm", default="force_aware_guided")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max_episodes", type=int, default=8)
    parser.add_argument("--samples_per_episode", type=int, default=4)
    parser.add_argument("--max_windows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--dp_norm_mode", choices=["minmax", "standard", "identity"], default="minmax")
    parser.add_argument("--min_contact_metric", type=float, default=1.8)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_improved_rate", type=float, default=0.80)
    parser.add_argument("--min_trust_region_pass_rate", type=float, default=0.999)
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
