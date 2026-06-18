#!/usr/bin/env python3
"""Build board TacQuality features in the actual Foresight-predicted domain.

The deploy-time guidance chain scores:

    action chunk -> Foresight -> predicted future marker -> TacQuality scorer

Older board scorer training used GT future marker features.  This builder
materializes the Foresight-predicted marker features first, so the downstream
ForceBandTacQualityEnergy training distribution matches serving-time inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.eval_board_force_band_scorer import (  # noqa: E402
    action_proxy_np,
    force_proxy_np,
    marker_proxy_np,
    summarize,
    window_ending,
)
from TFAC_V5.tac_quality_energy.eval_foresight_score_alignment import (  # noqa: E402
    build_bridge,
    future_marker,
    load_foresight,
    marker_window,
)


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618")
DEFAULT_FORESIGHT_DIR = Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload")
DEFAULT_FORESIGHT_CKPT = DEFAULT_FORESIGHT_DIR / "foresight_best.ckpt"

DATASETS = {
    "positive_old": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "positive_260617": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617",
    "too_small": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high",
    "too_large": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low",
    "oscillate": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate",
}

REASON4 = {"too_small": 0, "positive_old": 1, "positive_260617": 1, "too_large": 2, "oscillate": 3}
REASON5 = {"too_small": 0, "positive_old": 1, "positive_260617": 2, "too_large": 3, "oscillate": 4}


def chunk_from(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(len(arr), start + length)
    chunk = arr[start:end]
    if len(chunk) == 0:
        idx = max(0, min(start, len(arr) - 1))
        chunk = arr[idx : idx + 1]
    if len(chunk) < length:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], length - len(chunk), axis=0)], axis=0)
    return chunk.astype(np.float32)


def force_band_quality(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = np.asarray([r["label"] for r in rows], dtype=str)
    force_feat = np.stack([force_proxy_np(r["force_window"]) for r in rows])
    marker_delta = np.asarray([marker_proxy_np(r["gt_marker"])[12] for r in rows], dtype=np.float64)
    force_mag = force_feat[:, 0].astype(np.float64)
    force_delta = force_feat[:, 7].astype(np.float64)
    pos = np.char.startswith(labels, "positive")
    center = float(np.median(force_mag[pos]))
    mad = float(np.median(np.abs(force_mag[pos] - center)))
    sigma = max(1.4826 * mad, float(np.std(force_mag[pos])), 0.75)
    delta_ref = max(float(np.quantile(force_delta[pos], 0.75)), 0.05)
    marker_ref = max(float(np.quantile(marker_delta[pos], 0.75)), 1e-4)

    band = np.exp(-0.5 * ((force_mag - center) / sigma) ** 2)
    smooth = np.exp(-force_delta / delta_ref)
    marker_smooth = np.exp(-marker_delta / marker_ref)
    physical = np.clip(0.62 * band + 0.25 * smooth + 0.13 * marker_smooth, 0.0, 1.0).astype(np.float32)
    binary = np.asarray([1 if r["label"].startswith("positive") else 0 for r in rows], dtype=np.float32)
    guidance = np.clip(0.70 * binary + 0.30 * physical, 0.0, 1.0).astype(np.float32)
    ref = {
        "physical_definition": "0.62*positive_union_force_band + 0.25*force_smooth + 0.13*marker_smooth",
        "guidance_definition": "0.70*binary_good + 0.30*physical_quality",
        "force_mag_center_positive_union_median": center,
        "force_mag_sigma": sigma,
        "force_delta_ref_positive_union_q75": delta_ref,
        "marker_delta_ref_positive_union_q75": marker_ref,
        "physical_quality_by_label": {label: summarize(physical[labels == label]) for label in sorted(set(labels.tolist()))},
        "guidance_quality_by_label": {label: summarize(guidance[labels == label]) for label in sorted(set(labels.tolist()))},
        "force_mag_by_label": {label: summarize(force_mag[labels == label]) for label in sorted(set(labels.tolist()))},
        "force_delta_by_label": {label: summarize(force_delta[labels == label]) for label in sorted(set(labels.tolist()))},
    }
    return guidance, ref


def feature_sets(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    pred_left = np.stack([marker_proxy_np(r["pred_marker"]) for r in rows]).astype(np.float32)
    pred_right = pred_left.copy()
    pred_both = np.concatenate([pred_left, pred_right, np.abs(pred_left - pred_right)], axis=1)
    gt_left = np.stack([marker_proxy_np(r["gt_marker"]) for r in rows]).astype(np.float32)
    gt_right = gt_left.copy()
    gt_both = np.concatenate([gt_left, gt_right, np.abs(gt_left - gt_right)], axis=1)
    joint = np.stack([action_proxy_np(r["joint_action"], 7) for r in rows]).astype(np.float32)
    eef = np.stack([action_proxy_np(r["eef_action"], 6) for r in rows]).astype(np.float32)
    force = np.stack([force_proxy_np(r["force_window"]) for r in rows]).astype(np.float32)
    return {
        "marker_left": pred_left,
        "left_marker_action": np.concatenate([pred_left, joint, eef], axis=1),
        "left_marker_joint_action": np.concatenate([pred_left, joint], axis=1),
        "marker_action": np.concatenate([pred_both, joint, eef], axis=1),
        "marker_joint_action": np.concatenate([pred_both, joint], axis=1),
        "gt_marker_action": np.concatenate([gt_both, joint, eef], axis=1),
        "gt_marker_joint_action": np.concatenate([gt_both, joint], axis=1),
        "force_oracle": force,
        "pred_marker_action_force_oracle": np.concatenate([pred_both, joint, eef, force], axis=1),
    }


def collect_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    foresight, fs_cfg, fs_norm, fs_info = load_foresight(Path(args.foresight_dir), Path(args.foresight_ckpt), device)
    rows: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}

    for label, root in DATASETS.items():
        paths = sorted(Path(root).glob("episode_*.hdf5"))
        if args.max_episodes_per_class > 0:
            paths = paths[: args.max_episodes_per_class]
        for path in tqdm(paths, desc=label):
            try:
                with h5py.File(path, "r") as f:
                    marker = f[f"observations/tac/{args.tac_side}/marker_offset"][()].astype(np.float32)
                    qpos = f[f"observations/{args.proprio_key}"][()].astype(np.float32)
                    joint = f[args.joint_action_key][()].astype(np.float32)
                    eef = f[args.eef_action_key][()].astype(np.float32) if args.eef_action_key in f else joint[:, :6]
                    force = f[f"observations/tac/{args.tac_side}/force6d"][()].astype(np.float32)
            except (OSError, KeyError) as exc:
                skipped[str(path)] = skipped.get(str(path), 0) + 1
                continue

            length = min(len(marker), len(qpos), len(joint), len(eef), len(force))
            lo = max(args.window - 1, int(length * args.phase_start_frac))
            hi = min(int(length * args.phase_end_frac), length - args.action_chunk - args.horizon - 1)
            if hi <= lo:
                continue
            candidates = np.arange(lo, hi, dtype=np.int64)
            if args.contact_only:
                mag = np.linalg.norm(marker.reshape(len(marker), -1, 2), axis=-1).mean(axis=1)
                thr = np.quantile(mag, args.contact_quantile)
                candidates = candidates[mag[candidates] >= thr]
                if len(candidates) == 0:
                    continue
            n = len(candidates) if args.samples_per_episode <= 0 else min(args.samples_per_episode, len(candidates))
            starts = np.sort(rng.choice(candidates, size=n, replace=False))
            for start in starts:
                start = int(start)
                final = min(start + args.horizon, length - 1)
                marker_win = marker_window(marker, start, args.window)
                gt_marker = future_marker(marker, start, args.horizon, args.window)
                action_chunk = chunk_from(joint, start, args.action_chunk)
                eef_chunk = chunk_from(eef, start, args.action_chunk)
                force_window = window_ending(force, final, args.window)

                qpos_t = torch.tensor(qpos[start], dtype=torch.float32, device=device).view(1, -1)
                marker_t = torch.tensor(marker_win, dtype=torch.float32, device=device).unsqueeze(0)
                action_t = torch.tensor(action_chunk, dtype=torch.float32, device=device).unsqueeze(0)
                bridge = build_bridge(foresight, fs_norm, fs_cfg, qpos_t, marker_t)
                with torch.no_grad():
                    pred_marker = bridge(action_t)["left_marker_seq"][0].detach().cpu().numpy().astype(np.float32)

                rows.append(
                    {
                        "label": label,
                        "binary": 1 if label.startswith("positive") else 0,
                        "reason": REASON4[label],
                        "reason5": REASON5[label],
                        "group": f"{label}:{path.stem}",
                        "episode": str(path),
                        "start": start,
                        "end": final,
                        "pred_marker": pred_marker,
                        "gt_marker": gt_marker,
                        "joint_action": action_chunk,
                        "eef_action": eef_chunk,
                        "force_window": force_window,
                        "marker_mae": float(np.mean(np.abs(pred_marker - gt_marker))),
                    }
                )
    if not rows:
        raise RuntimeError("No predicted-domain rows collected")
    args._foresight_info = fs_info
    args._skipped = skipped
    return rows


def plot_feature_summary(labels: np.ndarray, values: np.ndarray, title: str, path: Path) -> None:
    uniq = sorted(set(labels.tolist()))
    data = [values[labels == label] for label in uniq]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    ax.boxplot(data, labels=uniq, showfliers=False)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args)
    labels = np.asarray([r["label"] for r in rows], dtype=str)
    binary = np.asarray([r["binary"] for r in rows], dtype=np.int64)
    reason = np.asarray([r["reason"] for r in rows], dtype=np.int64)
    reason5 = np.asarray([r["reason5"] for r in rows], dtype=np.int64)
    groups = np.asarray([r["group"] for r in rows], dtype=str)
    quality, quality_ref = force_band_quality(rows)
    features = feature_sets(rows)
    marker_mae = np.asarray([r["marker_mae"] for r in rows], dtype=np.float32)

    feature_path = out_dir / "board_predicted_domain_force_band_features.npz"
    np.savez_compressed(
        feature_path,
        labels=labels,
        binary=binary,
        reason=reason,
        reason5=reason5,
        quality=quality,
        groups=groups,
        episode=np.asarray([r["episode"] for r in rows], dtype=object),
        start=np.asarray([r["start"] for r in rows], dtype=np.int64),
        end=np.asarray([r["end"] for r in rows], dtype=np.int64),
        marker_mae=marker_mae,
        **features,
    )

    label_counts = {label: int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))}
    result: Dict[str, Any] = {
        "purpose": "Train/evaluate board scorer features in Foresight-predicted marker domain.",
        "feature_path": str(feature_path),
        "datasets": DATASETS,
        "n_samples": int(len(rows)),
        "n_groups": int(len(np.unique(groups))),
        "label_counts": label_counts,
        "feature_shapes": {name: list(value.shape) for name, value in features.items()},
        "quality_reference": quality_ref,
        "marker_mae_by_label": {label: summarize(marker_mae[labels == label]) for label in sorted(label_counts)},
        "foresight": {
            "dir": str(args.foresight_dir),
            "ckpt": str(args.foresight_ckpt),
            **getattr(args, "_foresight_info", {}),
        },
        "sampling": {
            "max_episodes_per_class": args.max_episodes_per_class,
            "samples_per_episode": args.samples_per_episode,
            "contact_only": args.contact_only,
            "contact_quantile": args.contact_quantile,
            "phase_start_frac": args.phase_start_frac,
            "phase_end_frac": args.phase_end_frac,
            "window": args.window,
            "horizon": args.horizon,
            "action_chunk": args.action_chunk,
        },
        "evidence_boundary": [
            "Feature cache uses Foresight-predicted marker fields, not GT marker fields, for deployable marker/action features.",
            "Labels and force-quality targets still come from offline collection regimes and future force windows.",
            "This does not prove real robot guidance improvement until rollout force curves are evaluated.",
        ],
    }
    (out_dir / "build_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "quality_target_reference.json").write_text(json.dumps(quality_ref, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_feature_summary(labels, marker_mae, "Foresight marker MAE by label", out_dir / "marker_mae_by_label.png")

    md = [
        "# Board Predicted-Domain ForceBand Features",
        "",
        "This cache uses `action -> Foresight -> predicted marker` before feature extraction.",
        "",
        f"- samples: `{result['n_samples']}`",
        f"- groups: `{result['n_groups']}`",
        f"- feature path: `{feature_path}`",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in label_counts.items():
        md.append(f"- {label}: `{count}`")
    md.extend(["", "## Marker Prediction MAE", "", "| label | mean | std | min | max |", "|---|---:|---:|---:|---:|"])
    for label, payload in result["marker_mae_by_label"].items():
        md.append(
            f"| {label} | {payload.get('mean', float('nan')):.4f} | "
            f"{payload.get('std', float('nan')):.4f} | {payload.get('min', float('nan')):.4f} | "
            f"{payload.get('max', float('nan')):.4f} |"
        )
    md.extend(["", "## Evidence Boundary", ""])
    for item in result["evidence_boundary"]:
        md.append(f"- {item}")
    (out_dir / "build_result.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"feature_path": str(feature_path), "result": str(out_dir / "build_result.json"), "n": len(rows)}, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--foresight_dir", type=Path, default=DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--foresight_ckpt", type=Path, default=DEFAULT_FORESIGHT_CKPT)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--joint_action_key", default="actions/joint_abs")
    parser.add_argument("--eef_action_key", default="actions/eef_abs")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--samples_per_episode", type=int, default=6)
    parser.add_argument("--max_episodes_per_class", type=int, default=0)
    parser.add_argument("--phase_start_frac", type=float, default=0.25)
    parser.add_argument("--phase_end_frac", type=float, default=0.85)
    parser.add_argument("--contact_only", action="store_true", default=True)
    parser.add_argument("--include_noncontact", dest="contact_only", action="store_false")
    parser.add_argument("--contact_quantile", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
