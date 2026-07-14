#!/usr/bin/env python3
"""Offline board stride-3 validation guidance diagnostics.

This script produces the four debug views we use for TouchGuide-style guidance:

1. score improvement per guided denoising step,
2. raw and effective guidance gradient magnitude,
3. action-dimension guidance heatmap on scorer validation rows,
4. DP denoising direction vs guidance direction alignment.

The DP-step traces come from previously logged stride-3 denoising audits.  The
action-dimension heatmap is recomputed on the scorer validation split using the
current stride-3 Foresight and latent energy checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime  # noqa: E402
from TFAC_V5.pretrain_latent_foresight_multistep import MultiStepLatentForesightModel  # noqa: E402


DEFAULT_FORESIGHT_DIR = (
    "/home/chenshuai/Project/output/foresight_ckpt/"
    "latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_temporalstride3_e100_bs16_preload"
)
DEFAULT_SCORER_DIR = "/home/chenshuai/Project/output/board_latent_energy/ce_margin_temporalstride3_e40"
DEFAULT_OUTPUT_DIR = "outputs/board_stride3_val_guidance_debug_20260714"
DEFAULT_DENSE_STEPS = "outputs/board_stride3_guidance_gradient_vis/ddpm_gradient_steps.csv"
DEFAULT_ALIGNMENT_JSON = "outputs/board_stride3_ddpm_stage_guidance_audit_20260706/board_ddpm_step_guidance_sweep.json"


ACTION_DIM_LABELS = ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]


def load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def as_np_stats(stats: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in stats.items()}


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def stat(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
    }


def sem(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / np.sqrt(arr.size))


def mean_sem(values: Iterable[float]) -> Tuple[float, float]:
    vals = list(values)
    if not vals:
        return float("nan"), 0.0
    return float(mean(vals)), sem(vals)


def resolve_path(path: str, remap_from: str, remap_to: str) -> str:
    if Path(path).exists():
        return path
    if remap_from and path.startswith(remap_from):
        alt = remap_to + path[len(remap_from):]
        if Path(alt).exists():
            return alt
    return path


def select_balanced_rows(rows: Sequence[Dict[str, Any]], per_class: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("class_name", "unknown"))].append(dict(row))
    selected: List[Dict[str, Any]] = []
    for cls in sorted(grouped):
        items = grouped[cls]
        if len(items) <= per_class:
            selected.extend(items)
            continue
        idx = rng.choice(len(items), size=per_class, replace=False)
        selected.extend(items[int(i)] for i in sorted(idx.tolist()))
    return selected


def load_foresight(foresight_dir: str, foresight_ckpt: str | None, device: torch.device):
    cfg = load_json(Path(foresight_dir) / "args.json")
    stats = as_np_stats(load_pickle(Path(foresight_dir) / "dataset_stats.pkl"))
    camera_names = list(cfg.get("camera_names", ["gelsight"]))
    model = MultiStepLatentForesightModel(
        camera_names=camera_names,
        cam_backbone_mapping={cam: 0 for cam in camera_names},
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        state_dim=int(cfg.get("state_dim", 7)),
        foresight_layers=int(cfg.get("foresight_layers", 3)),
        foresight_nheads=int(cfg.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(cfg.get("foresight_dim_feedforward", 2048)),
        dropout=float(cfg.get("dropout", 0.1)),
        tactile_mode=cfg.get("tactile_mode", "marker"),
        max_history=int(cfg.get("max_history", 8)),
        predict_horizon=int(cfg.get("predict_horizon", cfg.get("foresight_horizon", 16))),
        tactile_vae_ckpt=cfg.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(cfg.get("tactile_vae_latent_dim", 16)),
        tactile_vae_window=int(cfg.get("tactile_vae_window", 8)),
    ).to(device)
    ckpt_path = foresight_ckpt or str(Path(foresight_dir) / "foresight_best.ckpt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[foresight] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, stats, ckpt_path


def marker_history_norm(marker_all: np.ndarray, t: int, window: int, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    frames = []
    last = marker_all.shape[0] - 1
    for k in range(window):
        idx = max(0, min(t - (window - 1 - k), last))
        frames.append(marker_all[idx])
    raw = np.stack(frames, axis=0).astype(np.float32)
    return (raw - stats["marker_offset_mean"]) / stats["marker_offset_std"]


def gather_chunk(values: np.ndarray, start: int, horizon: int, temporal_stride: int) -> np.ndarray:
    last = values.shape[0] - 1
    idx = np.minimum(start + np.arange(horizon, dtype=np.int64) * temporal_stride, last)
    return values[idx].astype(np.float32)


def tensor_standard_norm(x: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=x.device).view(1, 1, -1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=x.device).view(1, 1, -1)
    return (x - mean_t) / std_t.clamp_min(1e-8)


def compute_val_action_gradients(
    rows: Sequence[Dict[str, Any]],
    *,
    foresight_model,
    foresight_cfg: Mapping[str, Any],
    foresight_stats: Mapping[str, np.ndarray],
    scorer: BoardLatentEnergyRuntime,
    device: torch.device,
    score_mode: str,
    guidance_path: str,
    remap_from: str,
    remap_to: str,
    guidance_scale: float,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    horizon = int(foresight_cfg.get("predict_horizon", foresight_cfg.get("foresight_horizon", 16)))
    temporal_stride = int(foresight_cfg.get("temporal_stride", 1))
    window = int(foresight_cfg.get("tactile_vae_window", 8))
    proprio_key = str(foresight_cfg.get("proprio_key", "proprio_joint"))
    action_key = str(foresight_cfg.get("action_key", "actions/joint_abs"))
    tac_side = str(foresight_cfg.get("tac_side", "left"))
    use_state_trajectory = bool(foresight_cfg.get("use_state_trajectory", False))

    sample_rows: List[Dict[str, Any]] = []
    grads: List[np.ndarray] = []
    for i, row in enumerate(rows):
        source_path = str(row["path"])
        ep_path = resolve_path(source_path, remap_from, remap_to)
        if not Path(ep_path).exists():
            sample_rows.append({
                "sample_idx": i,
                "source_path": source_path,
                "resolved_path": ep_path,
                "class_name": row.get("class_name", "unknown"),
                "start": int(row["start"]),
                "skipped": True,
                "skip_reason": "missing_hdf5",
            })
            continue

        start = int(row["start"])
        with h5py.File(ep_path, "r") as f:
            qpos_all = f[f"observations/{proprio_key}"][()].astype(np.float32)
            action_all = f[action_key][()].astype(np.float32)
            marker_all = f[f"observations/tac/{tac_side}/marker_offset"][()].astype(np.float32)

        qpos_raw = qpos_all[min(start, qpos_all.shape[0] - 1)]
        action_raw_np = gather_chunk(action_all, start, horizon, temporal_stride)
        marker_hist = marker_history_norm(marker_all, start, window, foresight_stats)
        qpos_norm = (qpos_raw - foresight_stats["qpos_mean"]) / (foresight_stats["qpos_std"] + 1e-8)

        images = [torch.as_tensor(marker_hist, dtype=torch.float32, device=device).unsqueeze(0)]
        qpos = torch.as_tensor(qpos_norm, dtype=torch.float32, device=device).unsqueeze(0)
        action_raw = torch.as_tensor(action_raw_np, dtype=torch.float32, device=device).unsqueeze(0)
        action_raw.requires_grad_(True)
        if use_state_trajectory:
            action_norm = tensor_standard_norm(action_raw, foresight_stats["qpos_mean"], foresight_stats["qpos_std"])
        else:
            action_norm = tensor_standard_norm(action_raw, foresight_stats["action_mean"], foresight_stats["action_std"])

        pred, _, _, _ = foresight_model(images, action_norm, future_images=None, qpos=qpos)
        score_action = action_raw.detach() if guidance_path == "latent_only" else action_raw
        out = scorer(score_action, pred, normalized=False)
        score = out[score_mode].mean()
        grad_score = torch.autograd.grad(score, action_raw, retain_graph=False)[0]
        grad_np = grad_score.detach().cpu().numpy()[0]
        grad_norm = float(np.linalg.norm(grad_np.reshape(-1)))
        per_dim_abs = np.abs(grad_np).mean(axis=0)
        top_dim = int(np.argmax(per_dim_abs))
        grads.append(grad_np)
        sample_rows.append({
            "sample_idx": i,
            "source_path": source_path,
            "resolved_path": ep_path,
            "class_name": row.get("class_name", "unknown"),
            "label": row.get("label"),
            "start": start,
            "score": float(score.detach().cpu()),
            "quality_0_100": float(out["quality_0_100"].mean().detach().cpu()),
            "p_expert": float(out["prob"][:, 0].mean().detach().cpu()),
            "grad_norm": grad_norm,
            "lambda_grad_norm": float(guidance_scale * grad_norm),
            "top_dim": ACTION_DIM_LABELS[top_dim] if top_dim < len(ACTION_DIM_LABELS) else f"dim{top_dim}",
            "top_dim_abs_grad": float(per_dim_abs[top_dim]),
            "skipped": False,
        })
    if not grads:
        return sample_rows, np.zeros((0, horizon, 7), dtype=np.float32)
    return sample_rows, np.stack(grads, axis=0).astype(np.float32)


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def dense_step_trace(rows: Sequence[Dict[str, str]], guidance_scale: float) -> Dict[str, np.ndarray]:
    by_step_delta: Dict[int, List[float]] = defaultdict(list)
    by_step_grad: Dict[int, List[float]] = defaultdict(list)
    by_step_update: Dict[int, List[float]] = defaultdict(list)
    by_step_no_delta: Dict[int, List[float]] = defaultdict(list)
    step_to_timestep: Dict[int, int] = {}
    for row in rows:
        step = int(float(row["ddpm_step_idx"]))
        step_to_timestep[step] = int(float(row["timestep"]))
        delta = f(row, "post_expert_margin") - f(row, "pre_expert_margin")
        if row["mode"] == "guided":
            by_step_delta[step].append(delta)
            by_step_grad[step].append(f(row, "grad_norm"))
            by_step_update[step].append(f(row, "applied_update_norm"))
        elif row["mode"] == "no_guidance":
            by_step_no_delta[step].append(delta)

    steps = np.asarray(sorted(by_step_delta), dtype=np.int64)
    guided_delta, guided_delta_sem = [], []
    no_delta, no_delta_sem = [], []
    grad, grad_sem = [], []
    lambda_grad, lambda_grad_sem = [], []
    update, update_sem = [], []
    for step in steps:
        m, e = mean_sem(by_step_delta[int(step)])
        guided_delta.append(m)
        guided_delta_sem.append(e)
        m, e = mean_sem(by_step_no_delta[int(step)])
        no_delta.append(m)
        no_delta_sem.append(e)
        m, e = mean_sem(by_step_grad[int(step)])
        grad.append(m)
        grad_sem.append(e)
        scaled = [guidance_scale * x for x in by_step_grad[int(step)]]
        m, e = mean_sem(scaled)
        lambda_grad.append(m)
        lambda_grad_sem.append(e)
        m, e = mean_sem(by_step_update[int(step)])
        update.append(m)
        update_sem.append(e)
    return {
        "step": steps,
        "local_step": np.arange(1, len(steps) + 1),
        "timestep": np.asarray([step_to_timestep[int(s)] for s in steps], dtype=np.int64),
        "guided_delta": np.asarray(guided_delta),
        "guided_delta_sem": np.asarray(guided_delta_sem),
        "no_delta": np.asarray(no_delta),
        "no_delta_sem": np.asarray(no_delta_sem),
        "grad": np.asarray(grad),
        "grad_sem": np.asarray(grad_sem),
        "lambda_grad": np.asarray(lambda_grad),
        "lambda_grad_sem": np.asarray(lambda_grad_sem),
        "update": np.asarray(update),
        "update_sem": np.asarray(update_sem),
    }


def alignment_trace(path: str | Path) -> Dict[str, Any]:
    data = load_json(path)
    traces: List[Dict[str, Any]] = []
    by_local: Dict[int, List[float]] = defaultdict(list)
    score_delta_by_local: Dict[int, List[float]] = defaultdict(list)
    for item in data.get("details", []):
        point = item.get("point", {})
        xs, ys = [], []
        for local_idx, row in enumerate(item.get("sample", {}).get("logs", []), start=1):
            if row.get("denoise_guidance_alignment") is None:
                continue
            align = float(row["denoise_guidance_alignment"])
            xs.append(local_idx)
            ys.append(align)
            by_local[local_idx].append(align)
            score_delta_by_local[local_idx].append(float(row.get("score_delta", 0.0)))
        if xs:
            traces.append({
                "episode_id": point.get("episode_id"),
                "start": point.get("start"),
                "x": np.asarray(xs, dtype=np.float64),
                "y": np.asarray(ys, dtype=np.float64),
            })

    steps = np.asarray(sorted(by_local), dtype=np.float64)
    means, errs, score_delta = [], [], []
    for step in steps.astype(int):
        m, e = mean_sem(by_local[int(step)])
        means.append(m)
        errs.append(e)
        score_delta.append(float(np.mean(score_delta_by_local[int(step)])))
    return {
        "raw": data,
        "traces": traces,
        "step": steps,
        "mean": np.asarray(means),
        "sem": np.asarray(errs),
        "score_delta": np.asarray(score_delta),
    }


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
    })


def plot_score(ax: plt.Axes, trace: Mapping[str, np.ndarray]) -> None:
    x = trace["local_step"]
    ax.plot(x, trace["guided_delta"], marker="o", color="#1864AB", linewidth=2.2, label="guided")
    ax.fill_between(
        x,
        trace["guided_delta"] - trace["guided_delta_sem"],
        trace["guided_delta"] + trace["guided_delta_sem"],
        color="#1864AB",
        alpha=0.16,
        linewidth=0,
    )
    ax.plot(x, trace["no_delta"], marker="s", color="#868E96", linewidth=1.8, label="no guidance")
    ax.axhline(0.0, color="#343A40", linewidth=0.9, alpha=0.6)
    ax.set_title("A. Score Before/After Guidance", loc="left", fontweight="bold")
    ax.set_xlabel("guided denoising step")
    ax.set_ylabel("score_after - score_before")
    ax.legend(frameon=False)


def plot_gradient(ax: plt.Axes, trace: Mapping[str, np.ndarray]) -> None:
    x = trace["local_step"]
    ax.plot(x, trace["grad"], marker="o", linewidth=2.2, color="#0B7285", label="||grad score||")
    ax.fill_between(x, np.maximum(1e-8, trace["grad"] - trace["grad_sem"]), trace["grad"] + trace["grad_sem"], color="#0B7285", alpha=0.14, linewidth=0)
    ax.plot(x, trace["lambda_grad"], marker="s", linewidth=2.0, color="#E67700", label="lambda ||grad score||")
    ax.plot(x, trace["update"], marker="^", linewidth=1.8, color="#862E9C", label="applied update norm")
    ax.set_yscale("log")
    ax.set_title("B. Gradient Norm And Effective Update", loc="left", fontweight="bold")
    ax.set_xlabel("guided denoising step")
    ax.set_ylabel("norm")
    ax.legend(frameon=False)


def plot_heatmap(ax: plt.Axes, grads: np.ndarray, guidance_scale: float) -> None:
    if grads.size == 0:
        heat = np.zeros((7, 16), dtype=np.float32)
    else:
        heat = guidance_scale * grads.mean(axis=0).T
    vmax = float(np.nanmax(np.abs(heat))) if heat.size else 1.0
    vmax = max(vmax, 1e-8)
    im = ax.imshow(heat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("C. Val Action-Dim Guidance Heatmap", loc="left", fontweight="bold")
    ax.set_xlabel("action horizon")
    ax.set_ylabel("action dim")
    ax.set_xticks(np.arange(0, heat.shape[1], max(1, heat.shape[1] // 8)))
    ax.set_yticks(np.arange(heat.shape[0]))
    labels = ACTION_DIM_LABELS[: heat.shape[0]]
    ax.set_yticklabels(labels)
    ax.grid(False)
    return im


def plot_alignment(ax: plt.Axes, align: Mapping[str, Any]) -> None:
    for tr in align["traces"]:
        ax.plot(tr["x"], tr["y"], color="#ADB5BD", linewidth=1.2, marker="o", markersize=3, alpha=0.7)
    x = align["step"]
    y = align["mean"]
    e = align["sem"]
    ax.plot(x, y, color="#C92A2A", linewidth=2.4, marker="o", label="mean")
    ax.fill_between(x, y - e, y + e, color="#C92A2A", alpha=0.16, linewidth=0)
    ax.axhline(0.0, color="#343A40", linewidth=0.9, alpha=0.7)
    ax.set_title("D. DP Direction vs Guidance Direction", loc="left", fontweight="bold")
    ax.set_xlabel("guided denoising step")
    ax.set_ylabel("cos(d_policy, d_guidance)")
    ax.set_ylim(-0.25, 0.25)
    ax.legend(frameon=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foresight_dir", default=DEFAULT_FORESIGHT_DIR)
    parser.add_argument("--foresight_ckpt", default=None)
    parser.add_argument("--scorer_ckpt", default=str(Path(DEFAULT_SCORER_DIR) / "board_latent_energy_best.pt"))
    parser.add_argument("--manifest", default=str(Path(DEFAULT_SCORER_DIR) / "manifest.json"))
    parser.add_argument("--train_metrics", default=str(Path(DEFAULT_SCORER_DIR) / "train_metrics.json"))
    parser.add_argument("--dense_steps_csv", default=DEFAULT_DENSE_STEPS)
    parser.add_argument("--alignment_json", default=DEFAULT_ALIGNMENT_JSON)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val_rows_per_class", type=int, default=4)
    parser.add_argument("--score_mode", default="expert_margin", choices=["score_good", "expert_margin", "quality_0_100", "p_expert"])
    parser.add_argument("--guidance_path", default="latent_only", choices=["full", "latent_only"])
    parser.add_argument("--guidance_scale", type=float, default=0.003)
    parser.add_argument("--remap_from", default="/home/chenshuai/data/dataset")
    parser.add_argument("--remap_to", default="/media/chenshuai/EXTERNAL_USB/pih_dataset")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(args.manifest)
    val_rows = select_balanced_rows(manifest["val_rows"], args.val_rows_per_class, args.seed)
    foresight_model, foresight_cfg, foresight_stats, foresight_ckpt = load_foresight(
        args.foresight_dir, args.foresight_ckpt, device
    )
    scorer = BoardLatentEnergyRuntime(args.scorer_ckpt, device=str(device)).to(device)
    scorer.eval()
    for p in scorer.parameters():
        p.requires_grad_(False)

    sample_rows, grads = compute_val_action_gradients(
        val_rows,
        foresight_model=foresight_model,
        foresight_cfg=foresight_cfg,
        foresight_stats=foresight_stats,
        scorer=scorer,
        device=device,
        score_mode=args.score_mode,
        guidance_path=args.guidance_path,
        remap_from=args.remap_from,
        remap_to=args.remap_to,
        guidance_scale=args.guidance_scale,
    )

    dense_trace = dense_step_trace(read_csv_rows(args.dense_steps_csv), args.guidance_scale)
    align = alignment_trace(args.alignment_json)

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2), constrained_layout=False)
    plot_score(axes[0, 0], dense_trace)
    plot_gradient(axes[0, 1], dense_trace)
    im = plot_heatmap(axes[1, 0], grads, args.guidance_scale)
    plot_alignment(axes[1, 1], align)
    cbar = fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbar.set_label("lambda * guidance direction")
    fig.suptitle("Blackboard Stride-3 Offline Guidance Diagnostics", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "board_stride3_val_guidance_debug_overview.png", bbox_inches="tight")
    fig.savefig(out_dir / "board_stride3_val_guidance_debug_overview.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    im = plot_heatmap(ax, grads, args.guidance_scale)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("lambda * guidance direction")
    fig.tight_layout()
    fig.savefig(out_dir / "val_action_dim_guidance_heatmap.png", bbox_inches="tight")
    fig.savefig(out_dir / "val_action_dim_guidance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    write_csv(out_dir / "val_action_gradient_samples.csv", sample_rows)
    np.savez_compressed(
        out_dir / "val_action_gradient_tensors.npz",
        grad_score=grads,
        lambda_grad_score=args.guidance_scale * grads,
    )

    train_metrics = load_json(args.train_metrics)
    final_epoch = train_metrics.get("history", [{}])[-1]
    val_metrics = final_epoch.get("val", {})
    successful = [r for r in sample_rows if not r.get("skipped")]
    summary = {
        "experiment": "board_stride3_val_guidance_debug",
        "device": str(device),
        "foresight_ckpt": foresight_ckpt,
        "scorer_ckpt": args.scorer_ckpt,
        "manifest": args.manifest,
        "dense_steps_csv": args.dense_steps_csv,
        "alignment_json": args.alignment_json,
        "score_mode": args.score_mode,
        "guidance_path": args.guidance_path,
        "guidance_scale": args.guidance_scale,
        "val_split": train_metrics.get("split", {}),
        "scorer_final_val_metrics": {
            "acc": val_metrics.get("acc"),
            "macro_f1": val_metrics.get("macro_f1"),
            "expert_vs_negative_auroc": val_metrics.get("expert_vs_negative_auroc"),
            "expert_margin_auroc": val_metrics.get("expert_margin_auroc"),
            "n": val_metrics.get("n"),
            "class_counts": val_metrics.get("class_counts"),
        },
        "selected_val_rows": {
            "requested_per_class": args.val_rows_per_class,
            "n_selected": len(val_rows),
            "n_successful_gradient": len(successful),
            "n_skipped": len(sample_rows) - len(successful),
            "grad_norm": stat([r["grad_norm"] for r in successful]),
            "lambda_grad_norm": stat([r["lambda_grad_norm"] for r in successful]),
            "score": stat([r["score"] for r in successful]),
            "quality_0_100": stat([r["quality_0_100"] for r in successful]),
            "p_expert": stat([r["p_expert"] for r in successful]),
        },
        "dense_denoising_logs": {
            "guided_step_delta_mean": stat(dense_trace["guided_delta"].tolist()),
            "grad_norm_mean_by_step": dense_trace["grad"].tolist(),
            "lambda_grad_norm_mean_by_step": dense_trace["lambda_grad"].tolist(),
            "applied_update_norm_mean_by_step": dense_trace["update"].tolist(),
        },
        "alignment_logs": {
            "summary": align["raw"].get("summary", {}),
            "mean_alignment_by_guided_step": align["mean"].tolist(),
            "mean_score_delta_by_guided_step": align["score_delta"].tolist(),
        },
        "artifacts": {
            "overview_png": str(out_dir / "board_stride3_val_guidance_debug_overview.png"),
            "overview_pdf": str(out_dir / "board_stride3_val_guidance_debug_overview.pdf"),
            "heatmap_png": str(out_dir / "val_action_dim_guidance_heatmap.png"),
            "sample_csv": str(out_dir / "val_action_gradient_samples.csv"),
            "gradient_npz": str(out_dir / "val_action_gradient_tensors.npz"),
            "summary_json": str(out_dir / "summary.json"),
        },
        "limitations": [
            "Current machine has no CUDA, so this run recomputes val action gradients on CPU.",
            "The original stride-3 DP checkpoint on /media/chenshuai/SANDISK ELE is not mounted; A/B/D reuse existing real stride-3 DP denoising logs.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# Blackboard Stride-3 Val Guidance Debug",
        "",
        f"- scorer final val acc: {val_metrics.get('acc')}",
        f"- scorer final val macro-F1: {val_metrics.get('macro_f1')}",
        f"- selected val rows with gradients: {len(successful)} / {len(sample_rows)}",
        f"- mean val grad norm: {summary['selected_val_rows']['grad_norm']['mean']:.6f}",
        f"- mean lambda grad norm: {summary['selected_val_rows']['lambda_grad_norm']['mean']:.6f}",
        f"- dense guided step score delta mean: {summary['dense_denoising_logs']['guided_step_delta_mean']['mean']:.6f}",
        f"- alignment mean by guided step: {summary['alignment_logs']['mean_alignment_by_guided_step']}",
        "",
        "Artifacts:",
        f"- `{summary['artifacts']['overview_png']}`",
        f"- `{summary['artifacts']['heatmap_png']}`",
        f"- `{summary['artifacts']['summary_json']}`",
    ]
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
