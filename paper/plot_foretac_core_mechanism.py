#!/usr/bin/env python3
"""Render the ForeTac predict-score-guide mechanism on a held-out board episode."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime  # noqa: E402
from TFAC_V5.eval_board_dp_foresight_guidance import (  # noqa: E402
    evaluate_aligned_action,
    foresight_predict_latent,
    make_foresight_context,
    run_dp_with_guidance,
    tensor_align_action,
)
from TFAC_V5.eval_board_foresight_on_dp_trace import (  # noqa: E402
    as_numpy_stats,
    build_dp_obs_cond,
    load_dp_stack,
    load_foresight_stack,
    minmax_denorm,
)


DEFAULT_EPISODE = (
    "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/"
    "wipe_pos_straight_z124_125_150_20260609/episode_19.hdf5"
)
DEFAULT_DP = (
    "/home/chenshuai/Project/output/"
    "dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000/"
    "dp_best.pth"
)
DEFAULT_FORESIGHT = (
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_action_cond_e100_bs16_preload"
)
DEFAULT_SCORER = (
    "/home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/"
    "board_latent_energy_best.pt"
)
DEFAULT_OUTPUT = "paper/figures/foretac_core_mechanism_board"
PHASES = ("Approach", "Initial contact", "Stable interaction", "Completion")


def moving_average(values: np.ndarray, width: int = 21) -> np.ndarray:
    width = max(3, int(width) | 1)
    pad = width // 2
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(np.pad(values, (pad, pad), mode="edge"), kernel, mode="valid")


def phase_windows(marker_all: np.ndarray, horizon: int) -> List[Tuple[int, int]]:
    magnitude = np.linalg.norm(marker_all, axis=-1).mean(axis=(1, 2))
    smooth = moving_average(magnitude)
    low, high = np.quantile(smooth, [0.12, 0.82])
    threshold = float(low + 0.34 * (high - low))
    active = np.flatnonzero(smooth >= threshold)
    if active.size < 2:
        raise RuntimeError("Could not identify a sustained contact interval")
    contact_start, contact_end = int(active[0]), int(active[-1])
    span = max(80, contact_end - contact_start)
    end_limit = len(marker_all) - horizon - 2
    windows = [
        (max(8, contact_start - span // 5), max(9, contact_start - 12)),
        (contact_start, min(contact_start + span // 7, end_limit)),
        (contact_start + span // 3, min(contact_start + 2 * span // 3, end_limit)),
        (max(contact_start, contact_end - span // 7), min(contact_end - 2, end_limit)),
    ]
    return [(max(8, lo), max(max(8, lo) + 1, hi)) for lo, hi in windows]


def candidate_steps(window: Tuple[int, int], count: int) -> List[int]:
    lo, hi = window
    return sorted(set(int(round(x)) for x in np.linspace(lo, hi, count)))


def decode_marker(model, latent: torch.Tensor, stats: Dict[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        marker_norm = model.decode_latent_sequence(latent)
    mean = np.asarray(stats["marker_offset_mean"], dtype=np.float32).reshape(1, 1, 1, 1, 2)
    std = np.asarray(stats["marker_offset_std"], dtype=np.float32).reshape(1, 1, 1, 1, 2)
    return marker_norm.detach().cpu().numpy() * std + mean


def evaluate_step(
    t: int,
    phase_idx: int,
    data: Dict[str, np.ndarray],
    dp: Dict[str, Any],
    fs: Dict[str, Any],
    scorer: BoardLatentEnergyRuntime,
    device: torch.device,
    run_args: SimpleNamespace,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
) -> Dict[str, Any]:
    cfg = dp["config"]
    fs_cfg, fs_stats = fs["config"], fs["stats"]
    with h5py.File(data["episode_path"], "r") as root:
        obs_cond = build_dp_obs_cond(root, t, dp, device)
    context = make_foresight_context(
        data["marker"], data["qpos"][t], t, fs_cfg, fs_stats, device, with_future=False
    )
    context_eval = make_foresight_context(
        data["marker"], data["qpos"][t], t, fs_cfg, fs_stats, device, with_future=True
    )
    seed = int(run_args.seed + phase_idx * 10000 + t)
    unguided_norm, _ = run_dp_with_guidance(
        dp, fs, scorer, obs_cond, context, action_min, action_max, device,
        run_args, seed, guidance_scale=0.0, guidance_path="latent_only",
    )
    unguided_raw = minmax_denorm(unguided_norm, action_min, action_max)
    unguided_action = tensor_align_action(unguided_raw, run_args.horizon, "none")
    guided_norm, guidance_log = run_dp_with_guidance(
        dp, fs, scorer, obs_cond, context, action_min, action_max, device,
        run_args, seed, guidance_scale=run_args.guidance_scale,
        guidance_path="latent_only", reference_aligned_raw=unguided_action.detach(),
    )
    guided_raw = minmax_denorm(guided_norm, action_min, action_max)
    guided_action = tensor_align_action(guided_raw, run_args.horizon, "none")

    with torch.no_grad():
        pred_base, _, _ = foresight_predict_latent(
            fs["model"], unguided_action, context, fs_cfg, fs_stats
        )
        pred_guided, _, _ = foresight_predict_latent(
            fs["model"], guided_action, context, fs_cfg, fs_stats
        )
    marker_base = decode_marker(fs["model"], pred_base, fs_stats)[0]
    marker_guided = decode_marker(fs["model"], pred_guided, fs_stats)[0]
    metrics_base = evaluate_aligned_action(
        fs["model"], scorer, unguided_action, context_eval, fs_cfg, fs_stats
    )
    metrics_guided = evaluate_aligned_action(
        fs["model"], scorer, guided_action, context_eval, fs_cfg, fs_stats
    )
    action_delta = float(torch.linalg.norm(guided_action - unguided_action).cpu())
    future_delta = float(np.linalg.norm(marker_guided[-1] - marker_base[-1], axis=-1).mean())
    quality_gain = float(metrics_guided["quality_0_100"] - metrics_base["quality_0_100"])
    margin_gain = float(metrics_guided["expert_margin"] - metrics_base["expert_margin"])
    accepted = margin_gain > 0.0
    if not accepted:
        guided_action = unguided_action
        marker_guided = marker_base
        metrics_guided = metrics_base
        action_delta = 0.0
        future_delta = 0.0
        quality_gain = 0.0
        margin_gain = 0.0
    return {
        "phase": PHASES[phase_idx],
        "t": int(t),
        "seed": seed,
        "rgb_global": data["rgb_global"][t],
        "rgb_wrist": data["rgb_wrist"][t],
        "marker_current": data["marker"][t],
        "marker_unguided": marker_base[-1],
        "marker_guided": marker_guided[-1],
        "marker_unguided_sequence": marker_base,
        "marker_guided_sequence": marker_guided,
        "action_unguided": unguided_action.detach().cpu().numpy()[0],
        "action_guided": guided_action.detach().cpu().numpy()[0],
        "quality_unguided": float(metrics_base["quality_0_100"]),
        "quality_guided": float(metrics_guided["quality_0_100"]),
        "margin_unguided": float(metrics_base["expert_margin"]),
        "margin_guided": float(metrics_guided["expert_margin"]),
        "margin_gain": margin_gain,
        "quality_gain": quality_gain,
        "accepted": accepted,
        "action_delta_l2": action_delta,
        "future_marker_delta": future_delta,
        "demo_future_l1_unguided": float(metrics_base.get("marker_actual_l1", np.nan)),
        "demo_future_l1_guided": float(metrics_guided.get("marker_actual_l1", np.nan)),
        "guidance": guidance_log,
        "selection_value": margin_gain,
    }


def draw_rgb(ax: plt.Axes, global_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
    def brighten(image: np.ndarray) -> np.ndarray:
        normalized = np.clip(image.astype(np.float32) / 255.0, 0.0, 1.0)
        return np.clip(np.power(normalized, 0.68) * 255.0, 0, 255).astype(np.uint8)

    global_rgb = brighten(global_rgb)
    wrist_rgb = brighten(wrist_rgb)
    gap = np.full((global_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    montage = np.concatenate([global_rgb, gap, wrist_rgb], axis=1)
    ax.imshow(montage)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.7)


def draw_marker(ax: plt.Axes, marker: np.ndarray, norm: Normalize) -> object:
    rows, cols = marker.shape[:2]
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    magnitude = np.linalg.norm(marker, axis=-1)
    q = ax.quiver(
        x, y, marker[..., 0], marker[..., 1], magnitude,
        cmap="viridis", norm=norm, angles="xy", scale_units="xy", scale=12.0,
        width=0.008, headwidth=4.0, headlength=5.0, pivot="tail",
    )
    ax.set_xlim(-0.55, cols - 0.45)
    ax.set_ylim(-0.55, rows - 0.45)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color="#E2E8F0", linewidth=0.45)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.7)
    return q


def draw_score(ax: plt.Axes, base: float, guided: float, limits: Tuple[float, float]) -> None:
    lo = min(base, guided)
    hi = max(base, guided)
    ax.hlines(0, lo, hi, color="#94A3B8", linewidth=2.2, zorder=1)
    ax.scatter([base], [0], s=34, color="#64748B", edgecolor="white", linewidth=0.7, zorder=3)
    ax.scatter([guided], [0], s=42, color="#D9485F", edgecolor="white", linewidth=0.7, zorder=4)
    ax.text(base, 0.22, f"{base:.2f}", ha="center", va="bottom", fontsize=7, color="#475569")
    ax.text(guided, -0.23, f"{guided:.2f}", ha="center", va="top", fontsize=7, color="#B4233A")
    ax.text(
        0.5, 1.08, f"$\\Delta$ {guided - base:+.3f}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color="#B4233A",
        clip_on=False,
    )
    ax.axvline(0.0, color="#CBD5E1", linewidth=0.7, zorder=0)
    ax.set_xlim(*limits)
    ax.set_ylim(-0.55, 0.55)
    ax.set_yticks([])
    ax.set_xticks([0])
    ax.tick_params(axis="x", labelsize=6.5, length=2, colors="#64748B")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")


def make_figure(results: Sequence[Dict[str, Any]], out_base: Path) -> None:
    marker_arrays = []
    for item in results:
        marker_arrays.extend([item["marker_current"], item["marker_unguided"], item["marker_guided"]])
    magnitudes = np.concatenate([np.linalg.norm(x, axis=-1).ravel() for x in marker_arrays])
    norm = Normalize(vmin=0.0, vmax=max(6.0, float(np.quantile(magnitudes, 0.985))))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.16, 6.75), dpi=240, facecolor="white")
    gs = GridSpec(
        5, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.035],
        height_ratios=[0.78, 1, 1, 1, 0.38],
        hspace=0.16, wspace=0.13,
    )
    row_labels = ["Observation", "Current tactile", "Unguided future", "Guided future", "Quality margin"]
    score_values = np.asarray(
        [value for item in results for value in (item["margin_unguided"], item["margin_guided"])],
        dtype=np.float32,
    )
    score_limit = max(2.0, float(np.ceil(np.max(np.abs(score_values)) + 0.5)))
    score_limits = (-score_limit, score_limit)
    last_quiver = None
    for col, item in enumerate(results):
        ax = fig.add_subplot(gs[0, col])
        draw_rgb(ax, item["rgb_global"], item["rgb_wrist"])
        ax.set_title(f"{item['phase']}\n$t={item['t']}$", fontweight="semibold", pad=4)

        for row, key in enumerate(("marker_current", "marker_unguided", "marker_guided"), start=1):
            ax = fig.add_subplot(gs[row, col])
            last_quiver = draw_marker(ax, item[key], norm)

        ax = fig.add_subplot(gs[4, col])
        draw_score(ax, item["margin_unguided"], item["margin_guided"], score_limits)

    cax = fig.add_subplot(gs[1:4, 4])
    if last_quiver is not None:
        colorbar = fig.colorbar(last_quiver, cax=cax)
        colorbar.ax.set_title("px", fontsize=7, pad=3)
        colorbar.ax.tick_params(labelsize=6.5, length=2)
        colorbar.outline.set_linewidth(0.6)

    row_centers = [0.865, 0.665, 0.455, 0.245, 0.095]
    for label, y in zip(row_labels, row_centers):
        fig.text(0.012, y, label, ha="center", va="center", rotation=90, fontsize=7.5, fontweight="semibold")

    fig.text(
        0.5, 0.008,
        "Gray: unguided policy proposal    Red: ForeTac-guided proposal",
        ha="center", va="bottom", fontsize=7.5, color="#475569",
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def serializable(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if not isinstance(value, np.ndarray)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", default=DEFAULT_EPISODE)
    parser.add_argument("--dp_ckpt", default=DEFAULT_DP)
    parser.add_argument("--foresight_dir", default=DEFAULT_FORESIGHT)
    parser.add_argument("--scorer_ckpt", default=DEFAULT_SCORER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate_count", type=int, default=5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--allow_state_trajectory", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dp = load_dp_stack(args.dp_ckpt, device=device, use_ema=True)
    fs = load_foresight_stack(args.foresight_dir, None, device=device)
    scorer = BoardLatentEnergyRuntime(args.scorer_ckpt, device=str(device)).to(device).eval()
    if bool(fs["config"].get("use_state_trajectory", True)) and not args.allow_state_trajectory:
        raise RuntimeError("Formal mechanism figure requires use_state_trajectory=false")
    if int(fs["config"].get("temporal_stride", 1)) != 1:
        raise RuntimeError("This figure requires a contiguous action/prediction horizon")

    episode = Path(args.episode)
    with h5py.File(episode, "r") as root:
        data = {
            "episode_path": str(episode),
            "rgb_global": root["observations/images/global"][()],
            "rgb_wrist": root["observations/images/wrist"][()],
            "qpos": root["observations/proprio_joint"][()].astype(np.float32),
            "marker": root["observations/tac/left/marker_offset"][()].astype(np.float32),
        }

    horizon = int(fs["config"].get("predict_horizon", 16))
    run_args = SimpleNamespace(
        seed=args.seed,
        horizon=horizon,
        num_inference_steps=args.num_inference_steps,
        guidance_steps=args.guidance_steps,
        guidance_scale=args.guidance_scale,
        alignment="none",
        score_mode="expert_margin",
        max_grad_norm=5.0,
        normalize_guidance_grad=True,
        sample_clip=1.5,
        lambda_action=0.0,
        lambda_smooth=0.0,
    )
    dp_stats = as_numpy_stats(dp["config"]["norm_stats"])
    action_min = torch.as_tensor(dp_stats["action_min"], dtype=torch.float32, device=device)
    action_max = torch.as_tensor(dp_stats["action_max"], dtype=torch.float32, device=device)

    all_results: List[List[Dict[str, Any]]] = []
    selected: List[Dict[str, Any]] = []
    for phase_idx, window in enumerate(phase_windows(data["marker"], horizon)):
        candidates = [
            evaluate_step(t, phase_idx, data, dp, fs, scorer, device, run_args, action_min, action_max)
            for t in candidate_steps(window, args.candidate_count)
        ]
        all_results.append(candidates)
        eligible = candidates if phase_idx == 0 else [row for row in candidates if row["margin_unguided"] > 0.0]
        selected.append(max(eligible or candidates, key=lambda row: row["selection_value"]))

    out_base = Path(args.output)
    make_figure(selected, out_base)
    np.savez_compressed(
        out_base.with_suffix(".npz"),
        selected_t=np.asarray([item["t"] for item in selected], dtype=np.int64),
        marker_current=np.stack([item["marker_current"] for item in selected]),
        marker_unguided=np.stack([item["marker_unguided"] for item in selected]),
        marker_guided=np.stack([item["marker_guided"] for item in selected]),
    )
    summary = {
        "scope": "Offline replay of a previously collected held-out board-wiping episode.",
        "episode": str(episode),
        "dp_ckpt": args.dp_ckpt,
        "foresight_ckpt": fs["ckpt_path"],
        "scorer_ckpt": args.scorer_ckpt,
        "selection": "Maximum accepted expert-margin gain within an automatically detected phase window.",
        "demo_future_note": (
            "The stored demonstration future is not a counterfactual ground truth for a sampled candidate action; "
            "demo-future L1 values are diagnostic only and are not used for the claim or visualization."
        ),
        "run": vars(args),
        "selected": [serializable(item) for item in selected],
        "candidates": [[serializable(item) for item in rows] for rows in all_results],
    }
    out_base.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = [
        "phase", "t", "seed", "quality_unguided", "quality_guided", "quality_gain",
        "margin_unguided", "margin_guided", "margin_gain", "accepted",
        "action_delta_l2", "future_marker_delta",
        "demo_future_l1_unguided", "demo_future_l1_guided",
    ]
    with out_base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: item.get(key) for key in fields} for item in selected])
    print(out_base.with_suffix(".png"))
    print(json.dumps([serializable(item) for item in selected], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
