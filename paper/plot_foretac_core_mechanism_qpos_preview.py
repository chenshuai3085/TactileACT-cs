#!/usr/bin/env python3
"""Render a v8j board mechanism preview with legacy qpos-conditioned foresight."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import h5py
import matplotlib
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.plot_foretac_core_mechanism import (
    PHASES,
    candidate_steps,
    phase_windows,
)
from paper.plot_foretac_multitask_qpos_preview import draw_marker
from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime
from TFAC_V5.eval_board_dp_foresight_guidance import (
    foresight_predict_latent,
    make_foresight_context,
)
from TFAC_V5.eval_board_foresight_on_dp_trace import load_foresight_stack

EPISODE = (
    "/media/chenshuai/EXTERNAL_USB/pih_dataset/260625_v8j_caheiban/"
    "peg_in_hole_0625/episode_3.hdf5"
)
FORESIGHT_DIR = (
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_caheiban_260609_260610_260625_"
    "nostride_h16_future1_exhaustive_e10_bs32_0"
)
SCORER_CKPT = (
    "/home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/"
    "board_latent_energy_best.pt"
)
OUTPUT_BASE = ROOT / "paper/figures/foretac_core_mechanism_board_v8j_qpos_preview"
MULTITIME_BASE = ROOT / "paper/figures/foretac_guidance_multitime_board_v8j_qpos_preview"
HORIZON_STEPS = (1, 4, 8, 12, 16)


def configure_reproducibility(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def brighten(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, [2.0, 98.5])
    image = np.clip((image - lo) / max(float(hi - lo), 1.0), 0.0, 1.0)
    return np.clip(np.power(image, 0.72) * 255.0, 0, 255).astype(np.uint8)


def draw_rgb(ax: plt.Axes, global_rgb: np.ndarray, wrist_rgb: np.ndarray) -> None:
    global_rgb, wrist_rgb = brighten(global_rgb), brighten(wrist_rgb)
    gap = np.full((global_rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    ax.imshow(np.concatenate([global_rgb, gap, wrist_rgb], axis=1))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.65)


def action_change(action_base: np.ndarray, action_guided: np.ndarray) -> np.ndarray:
    return np.linalg.norm(action_guided - action_base, axis=-1)


def decode_marker(model: torch.nn.Module, latent: torch.Tensor, stats: Dict[str, np.ndarray]) -> np.ndarray:
    with torch.no_grad():
        marker_norm = model.decode_latent_sequence(latent)
    mean = np.asarray(stats["marker_offset_mean"], dtype=np.float32).reshape(1, 1, 1, 1, 2)
    std = np.asarray(stats["marker_offset_std"], dtype=np.float32).reshape(1, 1, 1, 1, 2)
    return marker_norm.detach().cpu().numpy() * std + mean


def tactile_score(
    fs: Dict[str, Any],
    scorer: BoardLatentEnergyRuntime,
    qpos_chunk: torch.Tensor,
    scorer_reference: torch.Tensor,
    context: Dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    pred, _, _ = foresight_predict_latent(
        fs["model"], qpos_chunk, context, fs["config"], fs["stats"]
    )
    # Keep the scorer action branch fixed so improvement is caused by predicted tactile only.
    out = scorer(scorer_reference.detach(), pred, normalized=False)
    return out["expert_margin"].mean(), pred, out


def evaluate_qpos_step(
    t: int,
    phase_idx: int,
    data: Dict[str, np.ndarray],
    fs: Dict[str, Any],
    scorer: BoardLatentEnergyRuntime,
    device: torch.device,
    optimization_steps: int = 30,
) -> Dict[str, Any]:
    horizon = int(fs["config"].get("predict_horizon", 16))
    base_np = data["qpos"][t + 1:t + 1 + horizon].copy()
    if len(base_np) != horizon:
        raise ValueError(f"Incomplete qpos horizon at t={t}")
    base = torch.as_tensor(base_np, dtype=torch.float32, device=device).unsqueeze(0)
    context = make_foresight_context(
        data["marker"], data["qpos"][t], t,
        fs["config"], fs["stats"], device, with_future=False,
    )
    with torch.no_grad():
        base_score, pred_base, base_out = tactile_score(fs, scorer, base, base, context)

    guided = base.clone().detach().requires_grad_(True)
    max_delta = (0.02 * scorer.action_std).clamp(min=0.03, max=0.30).view(1, 1, -1)
    optimizer = torch.optim.Adam([guided], lr=0.035)
    score_trace = [float(base_score.cpu())]
    for _ in range(optimization_steps):
        optimizer.zero_grad(set_to_none=True)
        score, _, _ = tactile_score(fs, scorer, guided, base, context)
        proximity = ((guided - base) / max_delta).pow(2).mean()
        smoothness = (guided[:, 1:] - guided[:, :-1]).pow(2).mean()
        objective = score - 0.01 * proximity - 1e-4 * smoothness
        (-objective).backward()
        torch.nn.utils.clip_grad_norm_([guided], 5.0)
        optimizer.step()
        with torch.no_grad():
            guided.copy_(torch.maximum(torch.minimum(guided, base + max_delta), base - max_delta))
            current_score, _, _ = tactile_score(fs, scorer, guided, base, context)
            score_trace.append(float(current_score.cpu()))

    with torch.no_grad():
        guided_score, pred_guided, guided_out = tactile_score(fs, scorer, guided, base, context)
    marker_base = decode_marker(fs["model"], pred_base, fs["stats"])[0]
    marker_guided = decode_marker(fs["model"], pred_guided, fs["stats"])[0]
    margin_base = float(base_score.cpu())
    margin_guided = float(guided_score.cpu())
    return {
        "phase": PHASES[phase_idx],
        "t": int(t),
        "rgb_global": data["rgb_global"][t],
        "rgb_wrist": data["rgb_wrist"][t],
        "marker_current": data["marker"][t],
        "marker_unguided_sequence": marker_base,
        "marker_guided_sequence": marker_guided,
        "marker_unguided": marker_base[-1],
        "marker_guided": marker_guided[-1],
        "action_unguided": base.detach().cpu().numpy()[0],
        "action_guided": guided.detach().cpu().numpy()[0],
        "margin_unguided": margin_base,
        "margin_guided": margin_guided,
        "margin_gain": margin_guided - margin_base,
        "quality_unguided": float(base_out["quality_0_100"].mean().cpu()),
        "quality_guided": float(guided_out["quality_0_100"].mean().cpu()),
        "action_delta_l2": float(torch.linalg.norm(guided - base).cpu()),
        "future_marker_delta": float(np.linalg.norm(marker_guided - marker_base, axis=-1).mean()),
        "score_trace": score_trace,
    }


def make_figure(selected: List[Dict[str, Any]], key: Dict[str, Any]) -> None:
    marker_arrays = [key["marker_current"], key["marker_unguided_sequence"], key["marker_guided_sequence"]]
    marker_values = np.concatenate([
        np.linalg.norm(value, axis=-1).reshape(-1) for value in marker_arrays
    ])
    norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.quantile(marker_values, 0.99))))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.16, 5.65), dpi=260, facecolor="white")
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.9, 2.35], hspace=0.18)
    stages = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.08)
    for idx, item in enumerate(selected):
        ax = fig.add_subplot(stages[0, idx])
        draw_rgb(ax, item["rgb_global"], item["rgb_wrist"])
        phase_name = "Contact" if item["phase"] == "Initial contact" else item["phase"]
        ax.set_title(f"{phase_name}\n$t={item['t']}$", fontweight="semibold", pad=3)

    detail = GridSpecFromSubplotSpec(
        3, 8, subplot_spec=outer[1],
        width_ratios=[0.92, 0.14, 0.82, 0.82, 0.82, 0.82, 0.82, 1.15],
        height_ratios=[1, 1, 0.58], hspace=0.16, wspace=0.14,
    )
    current_ax = fig.add_subplot(detail[0:2, 0])
    last_image = draw_marker(current_ax, key["marker_current"], norm)
    current_ax.set_title(f"Current tactile\n$t={key['t']}$", pad=3)

    for row, label, color in ((0, "Unguided", "#475569"), (1, "ForeTac", "#B4233A")):
        label_ax = fig.add_subplot(detail[row, 1])
        label_ax.axis("off")
        label_ax.text(
            0.5, 0.5, label, rotation=90, ha="center", va="center",
            color=color, fontsize=7, fontweight="semibold",
        )
    for col_idx, horizon in enumerate(HORIZON_STEPS, start=2):
        base_ax = fig.add_subplot(detail[0, col_idx])
        guided_ax = fig.add_subplot(detail[1, col_idx])
        last_image = draw_marker(base_ax, key["marker_unguided_sequence"][horizon - 1], norm)
        draw_marker(guided_ax, key["marker_guided_sequence"][horizon - 1], norm)
        base_ax.set_title(f"$H={horizon}$", pad=3)

    score_ax = fig.add_subplot(detail[0:2, 7])
    base_score, guided_score = key["margin_unguided"], key["margin_guided"]
    score_ax.plot(
        [base_score, guided_score], [0, 0],
        color="#94A3B8", linewidth=2.2, zorder=1,
    )
    score_ax.annotate(
        "", xy=(guided_score, 0), xytext=(base_score, 0),
        arrowprops={"arrowstyle": "->", "color": "#D1495B", "lw": 1.4},
    )
    score_ax.scatter([base_score], [0], s=38, color="#94A3B8", edgecolor="white", linewidth=0.6, zorder=3)
    score_ax.scatter([guided_score], [0], s=46, color="#D1495B", edgecolor="white", linewidth=0.6, zorder=4)
    score_ax.axvline(0, color="#CBD5E1", linewidth=0.7)
    score_ax.set_yticks([])
    score_ax.set_ylim(-0.65, 0.65)
    score_ax.tick_params(labelsize=6.4, length=2)
    score_ax.grid(axis="x", color="#E2E8F0", linewidth=0.55)
    score_ax.spines[["top", "right", "left"]].set_visible(False)
    score_ax.spines["bottom"].set_color("#CBD5E1")
    score_ax.set_title(f"Quality margin\n$\\Delta={key['margin_gain']:+.3f}$", pad=3)
    score_ax.text(base_score, 0.16, f"Unguided\n{base_score:.2f}", ha="center", va="bottom", fontsize=6.4, color="#475569")
    score_ax.text(guided_score, -0.16, f"ForeTac\n{guided_score:.2f}", ha="center", va="top", fontsize=6.4, color="#B4233A")

    action_ax = fig.add_subplot(detail[2, 0:7])
    change = action_change(key["action_unguided"], key["action_guided"])
    steps = np.arange(1, len(change) + 1)
    action_ax.plot(steps, change, color="#D1495B", linewidth=1.35)
    action_ax.fill_between(steps, change, color="#D1495B", alpha=0.10)
    action_ax.scatter(steps, change, color="#D1495B", s=9, edgecolor="white", linewidth=0.35)
    action_ax.set_xlim(1, len(change))
    action_ax.set_ylim(bottom=0)
    action_ax.set_xticks([1, 4, 8, 12, 16])
    action_ax.set_ylabel("Joint $\\Delta$ L2", fontsize=6.7)
    action_ax.set_xlabel("Chunk step", fontsize=6.7, labelpad=1)
    action_ax.tick_params(labelsize=6.2, length=2)
    action_ax.grid(axis="y", color="#E2E8F0", linewidth=0.55)
    action_ax.spines[["top", "right"]].set_visible(False)
    action_ax.spines[["left", "bottom"]].set_color("#CBD5E1")

    cax = fig.add_subplot(detail[2, 7])
    colorbar = fig.colorbar(last_image, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5.8, length=1.5)
    colorbar.set_label("Marker displacement (px)", fontsize=6.2, labelpad=1)
    colorbar.outline.set_linewidth(0.5)

    fig.text(
        0.5, 0.008,
        "Before execution, ForeTac forecasts the tactile consequence of a candidate joint trajectory and locally improves its predicted contact quality.",
        ha="center", va="bottom", fontsize=6.8, color="#475569",
    )
    OUTPUT_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_BASE.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUTPUT_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_margin(ax: plt.Axes, item: Dict[str, Any], show_title: bool) -> None:
    base = float(item["margin_unguided"])
    guided = float(item["margin_guided"])
    padding = max(0.8, 0.18 * abs(guided - base))
    lo, hi = min(base, guided) - padding, max(base, guided) + padding
    ax.plot([base, guided], [0, 0], color="#94A3B8", linewidth=2.2, zorder=1)
    ax.annotate(
        "", xy=(guided, 0), xytext=(base, 0),
        arrowprops={"arrowstyle": "->", "color": "#D1495B", "lw": 1.4},
    )
    ax.scatter([base], [0], s=34, color="#94A3B8", edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter([guided], [0], s=42, color="#D1495B", edgecolor="white", linewidth=0.5, zorder=4)
    ax.text(base, 0.16, f"Before\n{base:.2f}", ha="center", va="bottom", fontsize=6.2, color="#475569")
    ax.text(guided, -0.16, f"After\n{guided:.2f}", ha="center", va="top", fontsize=6.2, color="#B4233A")
    ax.text(
        0.5, 0.05,
        f"$\\Delta$ margin {item['margin_gain']:+.2f}\nJoint $\\Delta$ L2 {item['action_delta_l2']:.2f}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2, color="#475569",
    )
    ax.axvline(0, color="#CBD5E1", linewidth=0.65)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.72, 0.72)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    if show_title:
        ax.set_title("Quality margin", pad=4)


def make_multitime_figure(items: List[Dict[str, Any]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.titlesize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.16, 6.35), dpi=260, facecolor="white")
    gs = GridSpec(
        6, 10, figure=fig,
        width_ratios=[1.48, 0.78, 0.14, 0.78, 0.78, 0.78, 0.78, 0.78, 1.26, 0.035],
        height_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.12, wspace=0.15,
    )
    for idx, item in enumerate(items):
        row_base, row_guided = idx * 2, idx * 2 + 1
        arrays = [
            item["marker_current"],
            item["marker_unguided_sequence"],
            item["marker_guided_sequence"],
        ]
        marker_values = np.concatenate([
            np.linalg.norm(value, axis=-1).reshape(-1) for value in arrays
        ])
        norm = Normalize(vmin=0.0, vmax=max(1.0, float(np.quantile(marker_values, 0.99))))

        rgb_ax = fig.add_subplot(gs[row_base:row_guided + 1, 0])
        draw_rgb(rgb_ax, item["rgb_global"], item["rgb_wrist"])
        phase_name = "Contact" if item["phase"] == "Initial contact" else item["phase"]
        rgb_ax.set_title(
            f"{phase_name}\n$t={item['t']}$",
            loc="left", fontweight="semibold", pad=3,
        )

        current_ax = fig.add_subplot(gs[row_base:row_guided + 1, 1])
        last_image = draw_marker(current_ax, item["marker_current"], norm)
        if idx == 0:
            current_ax.set_title("Current", pad=3)

        for row, label, color in (
            (row_base, "Unguided", "#475569"),
            (row_guided, "ForeTac", "#B4233A"),
        ):
            label_ax = fig.add_subplot(gs[row, 2])
            label_ax.axis("off")
            label_ax.text(
                0.5, 0.5, label, rotation=90, ha="center", va="center",
                fontsize=7, fontweight="semibold", color=color,
            )

        for col, horizon in enumerate(HORIZON_STEPS, start=3):
            base_ax = fig.add_subplot(gs[row_base, col])
            guided_ax = fig.add_subplot(gs[row_guided, col])
            last_image = draw_marker(base_ax, item["marker_unguided_sequence"][horizon - 1], norm)
            draw_marker(guided_ax, item["marker_guided_sequence"][horizon - 1], norm)
            if idx == 0:
                base_ax.set_title(f"$H={horizon}$", pad=3)

        margin_ax = fig.add_subplot(gs[row_base:row_guided + 1, 8])
        draw_margin(margin_ax, item, show_title=idx == 0)
        cax = fig.add_subplot(gs[row_base:row_guided + 1, 9])
        colorbar = fig.colorbar(last_image, cax=cax)
        colorbar.ax.tick_params(labelsize=5.8, length=1.5)
        colorbar.outline.set_linewidth(0.5)

    fig.text(
        0.5, 0.008,
        "Qpos-conditioned tactile consequences before and after bounded joint-trajectory guidance on a held-out v8j board episode.",
        ha="center", va="bottom", fontsize=6.8, color="#475569",
    )
    fig.savefig(MULTITIME_BASE.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(MULTITIME_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def serializable(item: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in item.items() if not isinstance(v, np.ndarray)}


def main() -> None:
    configure_reproducibility()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = load_foresight_stack(FORESIGHT_DIR, None, device=device)
    scorer = BoardLatentEnergyRuntime(SCORER_CKPT, device=str(device)).to(device).eval()
    if not bool(fs["config"].get("use_state_trajectory", False)):
        raise RuntimeError("This preview intentionally uses the existing qpos-conditioned checkpoint")
    val_paths = Path(FORESIGHT_DIR, "val_episode_paths.txt").read_text().splitlines()
    if EPISODE not in val_paths:
        raise RuntimeError("The selected v8j episode must be held out")

    with h5py.File(EPISODE, "r") as root:
        data = {
            "episode_path": EPISODE,
            "rgb_global": root["observations/images/global"][()],
            "rgb_wrist": root["observations/images/wrist"][()],
            "qpos": root["observations/proprio_joint"][()].astype(np.float32),
            "marker": root["observations/tac/left/marker_offset"][()].astype(np.float32),
        }

    horizon = int(fs["config"].get("predict_horizon", 16))
    selected: List[Dict[str, Any]] = []
    candidates_by_phase: List[List[Dict[str, Any]]] = []
    for phase_idx, window in enumerate(phase_windows(data["marker"], horizon)):
        candidates = [
            evaluate_qpos_step(t, phase_idx, data, fs, scorer, device)
            for t in candidate_steps(window, 5)
        ]
        candidates_by_phase.append(candidates)
        selected.append(max(candidates, key=lambda row: row["margin_gain"]))
    contact_candidates = [row for rows in candidates_by_phase[1:] for row in rows]
    key = max(contact_candidates, key=lambda row: row["margin_gain"])
    make_figure(selected, key)
    multitime = selected[1:]
    make_multitime_figure(multitime)

    np.savez_compressed(
        OUTPUT_BASE.with_suffix(".npz"),
        phase_t=np.asarray([x["t"] for x in selected]),
        key_t=np.asarray(key["t"]),
        current=key["marker_current"],
        unguided=key["marker_unguided_sequence"],
        guided=key["marker_guided_sequence"],
        action_unguided=key["action_unguided"],
        action_guided=key["action_guided"],
    )
    np.savez_compressed(
        MULTITIME_BASE.with_suffix(".npz"),
        timestep=np.asarray([x["t"] for x in multitime]),
        current=np.stack([x["marker_current"] for x in multitime]),
        unguided=np.stack([x["marker_unguided_sequence"] for x in multitime]),
        guided=np.stack([x["marker_guided_sequence"] for x in multitime]),
        action_unguided=np.stack([x["action_unguided"] for x in multitime]),
        action_guided=np.stack([x["action_guided"] for x in multitime]),
    )
    OUTPUT_BASE.with_suffix(".json").write_text(
        json.dumps({
            "scope": "Offline qpos-trajectory mechanism preview on a held-out v8j board episode.",
            "evidence_boundary": (
                "The recorded future qpos chunk is treated as the candidate joint trajectory. "
                "No actions/joint_abs or DP-generated action is used in this visualization."
            ),
            "episode": EPISODE,
            "foresight_checkpoint": fs["ckpt_path"],
            "scorer_checkpoint": SCORER_CKPT,
            "shown_horizons": list(HORIZON_STEPS),
            "selected_phases": [serializable(x) for x in selected],
            "key_result": serializable(key),
            "candidates": [[serializable(x) for x in rows] for rows in candidates_by_phase],
        }, indent=2),
        encoding="utf-8",
    )
    MULTITIME_BASE.with_suffix(".json").write_text(
        json.dumps({
            "scope": "Three qpos-guidance stages from a held-out v8j board episode.",
            "episode": EPISODE,
            "foresight_checkpoint": fs["ckpt_path"],
            "scorer_checkpoint": SCORER_CKPT,
            "shown_horizons": list(HORIZON_STEPS),
            "results": [serializable(x) for x in multitime],
        }, indent=2),
        encoding="utf-8",
    )
    fields = [
        "phase", "t", "margin_unguided", "margin_guided", "margin_gain",
        "quality_unguided", "quality_guided", "action_delta_l2", "future_marker_delta",
    ]
    with OUTPUT_BASE.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: x[k] for k in fields} for x in selected])
    with MULTITIME_BASE.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{k: x[k] for k in fields} for x in multitime])
    print(OUTPUT_BASE.with_suffix(".png"))
    print(json.dumps({k: key[k] for k in fields}, indent=2))


if __name__ == "__main__":
    main()
