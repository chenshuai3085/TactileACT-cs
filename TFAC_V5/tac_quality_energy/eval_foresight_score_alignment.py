#!/usr/bin/env python3
"""Evaluate whether Foresight-predicted TacQuality scores align with reality.

The guidance chain uses score(Foresight(action)), not score(GT tactile).  This
audit checks whether predicted scores preserve useful ordering on held-out
blackboard-wiping windows with known positive/negative collection regimes.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel
from TFAC_V5.pretrain_latent_foresight_multistep import MultiStepLatentForesightModel
from TFAC_V5.tac_quality_energy.foresight_bridge import ForesightBridgeConfig, ForesightTacQualityBridge
from TFAC_V5.tac_quality_energy.board_proxy_energy import BoardProxyEnergyRuntime
from TFAC_V5.tac_quality_energy.ptg_proxy_runtime import PTGProxyScorerV2Runtime
from TFAC_V5.tac_quality_energy.force_band_runtime import ForceBandTacQualityEnergyRuntime


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_foresight_score_alignment")
DEFAULT_FORESIGHT_DIR = Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload")
DEFAULT_FORESIGHT_CKPT = DEFAULT_FORESIGHT_DIR / "foresight_best.ckpt"
DEFAULT_SCORER = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
DEFAULT_FORCE_BAND_SCORER = Path("/home/chenshuai/Project/output/board_force_band_tac_quality_energy/force_band_tac_quality_energy_best.pt")


BOARD_DATASETS = {
    "positive": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "too_small": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high",
    "too_large": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low",
    "oscillate": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate",
}
POSITIVE_260617_DATASET = "/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def finite_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_simple(x: np.ndarray, y: np.ndarray) -> float | None:
    return finite_corr(rankdata_simple(np.asarray(x)), rankdata_simple(np.asarray(y)))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def marker_delta_np(marker_seq: np.ndarray) -> float:
    marker_seq = np.asarray(marker_seq, dtype=np.float32)
    if len(marker_seq) < 2:
        return 0.0
    delta = np.linalg.norm(marker_seq[1:] - marker_seq[:-1], axis=-1)
    return float(delta.mean())


def force_band_quality(
    force_window: np.ndarray | None,
    marker_seq: np.ndarray,
    force_center: float,
    force_sigma: float,
    force_delta_ref: float,
    marker_delta_ref: float,
) -> float | None:
    if force_window is None or len(force_window) == 0:
        return None
    force_window = np.asarray(force_window, dtype=np.float32)
    linear_norm = np.linalg.norm(force_window[:, :3], axis=-1)
    force_mag = float(np.mean(linear_norm))
    fz = force_window[:, 2]
    force_delta = float(np.mean(np.abs(np.diff(fz)))) if len(fz) > 1 else 0.0
    marker_delta = marker_delta_np(marker_seq)
    band_score = np.exp(-0.5 * ((force_mag - force_center) / max(force_sigma, 1e-6)) ** 2)
    smooth_score = np.exp(-force_delta / max(force_delta_ref, 1e-6))
    marker_smooth = np.exp(-marker_delta / max(marker_delta_ref, 1e-6))
    return float(np.clip(0.62 * band_score + 0.25 * smooth_score + 0.13 * marker_smooth, 0.0, 1.0))


def load_foresight(foresight_dir: Path, foresight_ckpt: Path, device: torch.device):
    cfg = load_json(foresight_dir / "args.json")
    camera_names = cfg.get("camera_names", ["gelsight"])
    predict_horizon = int(cfg.get("predict_horizon", cfg.get("foresight_horizon", 1)))
    kwargs = dict(
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
        predict_horizon=predict_horizon,
        tactile_vae_ckpt=cfg.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(cfg.get("tactile_vae_latent_dim", 16)),
    )
    if predict_horizon > 1:
        model = MultiStepLatentForesightModel(**kwargs).to(device)
        kind = "multistep"
    else:
        model = LatentForesightPretrainModel(
            **kwargs,
            use_delta_pred=bool(cfg.get("use_delta_pred", False)),
            residual_prediction=bool(cfg.get("residual_prediction", False)),
        ).to(device)
        kind = "single_step"
    state = torch.load(foresight_ckpt, map_location=device, weights_only=False)
    if isinstance(state, Mapping) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    freeze(model)
    fs_norm = load_pickle(foresight_dir / "dataset_stats.pkl")
    return model, cfg, fs_norm, {"kind": kind, "missing": len(missing), "unexpected": len(unexpected)}


def marker_window(raw_marker: np.ndarray, start: int, window: int) -> np.ndarray:
    frames = []
    for i in range(window):
        t = max(0, start - (window - 1 - i))
        frames.append(raw_marker[t])
    return np.stack(frames).astype(np.float32)


def future_marker(raw_marker: np.ndarray, start: int, horizon: int, window: int) -> np.ndarray:
    frames = []
    # Score the final future window that the Foresight bridge also returns.
    final = min(start + horizon, len(raw_marker) - 1)
    for i in range(window):
        t = max(0, final - (window - 1 - i))
        frames.append(raw_marker[t])
    return np.stack(frames).astype(np.float32)


def build_bridge(foresight, fs_norm: Mapping[str, Any], fs_cfg: Mapping[str, Any], qpos_raw: torch.Tensor, marker_window_raw: torch.Tensor) -> ForesightTacQualityBridge:
    norm = fs_cfg.get("norm_stats", fs_cfg)
    mean = tuple(float(x) for x in norm.get("marker_offset_mean", [-0.33986145, -2.9208484]))
    std = tuple(float(x) for x in norm.get("marker_offset_std", [1.9804853, 2.7671177]))
    mean_t = torch.tensor(mean, dtype=torch.float32, device=marker_window_raw.device).view(1, 1, 1, 1, 2)
    std_t = torch.tensor(std, dtype=torch.float32, device=marker_window_raw.device).view(1, 1, 1, 1, 2)
    marker_norm = (marker_window_raw - mean_t) / std_t.clamp_min(1e-8)
    return ForesightTacQualityBridge(
        foresight,
        fs_norm,
        qpos_raw=qpos_raw,
        foresight_images=[],
        marker_window_norm=marker_norm,
        config=ForesightBridgeConfig(
            task="board",
            window=int(fs_cfg.get("tactile_vae_window", 8)),
            action_chunk=int(fs_cfg.get("chunk_size", 16)),
            latent_dim=int(fs_cfg.get("tactile_vae_latent_dim", 16)),
            marker_mean=mean,
            marker_std=std,
            residual_prediction=bool(fs_cfg.get("residual_prediction", False)),
        ),
    )


def selected_board_datasets(args: argparse.Namespace) -> Dict[str, str]:
    datasets = dict(BOARD_DATASETS)
    if args.include_260617_positive:
        datasets = {
            "positive": datasets["positive"],
            "positive_260617": args.positive_260617_dir,
            "too_small": datasets["too_small"],
            "too_large": datasets["too_large"],
            "oscillate": datasets["oscillate"],
        }
    return datasets


def sample_windows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []
    label_to_good = {"positive": 1, "positive_260617": 1, "too_small": 0, "too_large": 0, "oscillate": 0}
    datasets = selected_board_datasets(args)
    for label, root in datasets.items():
        paths = sorted(Path(root).glob("episode_*.hdf5"))
        if args.max_episodes_per_class > 0:
            paths = paths[: args.max_episodes_per_class]
        for path in paths:
            try:
                with h5py.File(path, "r") as f:
                    marker = f[f"observations/tac/{args.tac_side}/marker_offset"][()]
                    qpos = f[f"observations/{args.proprio_key}"][()]
                    action = f[args.action_key][()]
                    force = f.get(f"observations/tac/{args.tac_side}/force6d")
                    force_arr = force[()] if force is not None else None
            except (OSError, KeyError):
                continue
            length = min(len(marker), len(qpos), len(action))
            lo = args.window - 1
            hi = max(lo + 1, length - args.action_chunk - args.horizon - 1)
            if hi <= lo:
                continue
            candidates = np.arange(lo, hi, dtype=np.int64)
            if args.contact_only:
                mag = np.linalg.norm(marker.reshape(len(marker), -1, 2), axis=-1).mean(axis=1)
                thr = np.quantile(mag, args.contact_quantile)
                candidates = candidates[mag[candidates] >= thr]
                if len(candidates) == 0:
                    continue
            n = min(args.samples_per_episode, len(candidates))
            starts = np.sort(rng.choice(candidates, size=n, replace=False))
            for start in starts:
                final = min(int(start) + args.horizon, len(marker) - 1)
                force_window = None
                if force_arr is not None:
                    force_window = force_arr[int(start) : final + 1]
                rows.append(
                    {
                        "label": label,
                        "good": label_to_good[label],
                        "episode": str(path),
                        "start": int(start),
                        "qpos": qpos[int(start)].astype(np.float32),
                        "action": action[int(start) : int(start) + args.action_chunk].astype(np.float32),
                        "marker_window": marker_window(marker, int(start), args.window),
                        "future_marker": future_marker(marker, int(start), args.horizon, args.window),
                        "future_force_abs_mean": float(np.mean(np.abs(force_window[:, 2]))) if force_window is not None and len(force_window) else None,
                        "future_force_mag_mean": float(np.mean(np.linalg.norm(force_window[:, :3], axis=-1))) if force_window is not None and len(force_window) else None,
                        "future_force_delta_abs_mean": float(np.mean(np.abs(np.diff(force_window[:, 2])))) if force_window is not None and len(force_window) > 1 else None,
                        "future_force_window": force_window.astype(np.float32) if force_window is not None else None,
                    }
                )
    if args.max_samples > 0 and len(rows) > args.max_samples:
        idx = rng.choice(np.arange(len(rows)), size=args.max_samples, replace=False)
        rows = [rows[int(i)] for i in np.sort(idx)]
    if not rows:
        raise RuntimeError("No windows sampled")
    return rows


def load_scorer(args: argparse.Namespace, device: torch.device):
    if args.scorer_runtime == "PTGProxyScorerV2Runtime":
        return PTGProxyScorerV2Runtime(str(args.scorer_ckpt), device=str(device))
    if args.scorer_runtime == "ForceBandTacQualityEnergyRuntime":
        return ForceBandTacQualityEnergyRuntime(str(args.scorer_ckpt), device=str(device))
    if args.scorer_runtime == "BoardProxyEnergyRuntime":
        return BoardProxyEnergyRuntime(device=str(device))
    raise KeyError(f"Unsupported scorer runtime: {args.scorer_runtime}")


@torch.no_grad()
def score_marker(scorer, marker: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "profile":
        if not hasattr(scorer, "weighted_energy_score"):
            return scorer.score(marker, right_marker_seq=marker, joint_action_seq=action, task_id=task_id, mode="profile")
        return scorer.weighted_energy_score(
            marker,
            right_marker_seq=marker,
            joint_action_seq=action,
            task_id=task_id,
            quality_weight=0.75,
            binary_weight=0.10,
            reason_weight=0.0,
            clip=True,
        )
    return scorer.score(marker, right_marker_seq=marker, joint_action_seq=action, task_id=task_id, mode=mode)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    foresight, fs_cfg, fs_norm, fs_info = load_foresight(Path(args.foresight_dir), Path(args.foresight_ckpt), device)
    scorer = load_scorer(args, device)
    rows = sample_windows(args)
    task_id = torch.ones(1, dtype=torch.long, device=device)

    out_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        action = torch.tensor(row["action"], dtype=torch.float32, device=device).unsqueeze(0)
        if action.shape[1] < args.action_chunk:
            pad = action[:, -1:].expand(-1, args.action_chunk - action.shape[1], -1)
            action = torch.cat([action, pad], dim=1)
        action_score = action[:, : args.action_chunk]
        qpos = torch.tensor(row["qpos"], dtype=torch.float32, device=device).view(1, -1)
        cur_marker = torch.tensor(row["marker_window"], dtype=torch.float32, device=device).unsqueeze(0)
        gt_future_marker = torch.tensor(row["future_marker"], dtype=torch.float32, device=device).unsqueeze(0)
        bridge = build_bridge(foresight, fs_norm, fs_cfg, qpos, cur_marker)
        with torch.enable_grad():
            pred_tactile = bridge(action)
        pred_marker = pred_tactile["left_marker_seq"].detach()
        pred_score = score_marker(scorer, pred_marker, action_score, task_id, args.score_mode)
        gt_score = score_marker(scorer, gt_future_marker, action_score, task_id, args.score_mode)
        marker_mae = torch.mean(torch.abs(pred_marker - gt_future_marker)).detach().cpu().item()
        out = {
            "idx": i,
            "label": row["label"],
            "good": int(row["good"]),
            "episode": row["episode"],
            "start": int(row["start"]),
            "pred_score": float(pred_score.detach().cpu()[0]),
            "gt_score": float(gt_score.detach().cpu()[0]),
            "pred_minus_gt": float((pred_score - gt_score).detach().cpu()[0]),
            "marker_mae": float(marker_mae),
            "future_force_abs_mean": row["future_force_abs_mean"],
            "future_force_mag_mean": row["future_force_mag_mean"],
            "future_force_delta_abs_mean": row["future_force_delta_abs_mean"],
            "future_force_band_quality": force_band_quality(
                row.get("future_force_window"),
                row["future_marker"],
                args.force_center,
                args.force_sigma,
                args.force_delta_ref,
                args.marker_delta_ref,
            ),
        }
        out_rows.append(out)

    result = summarize_results(out_rows, args, fs_info)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "foresight_score_alignment_samples.csv"
    json_path = out_dir / "foresight_score_alignment.json"
    md_path = out_dir / "foresight_score_alignment.md"
    write_csv(out_rows, csv_path)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    plot_scores(out_rows, out_dir / "foresight_score_alignment.png")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "csv": str(csv_path), "summary": result["summary"]}, indent=2, ensure_ascii=False))
    return result


def summarize_results(rows: List[Dict[str, Any]], args: argparse.Namespace, fs_info: Dict[str, Any]) -> Dict[str, Any]:
    good = np.asarray([r["good"] for r in rows], dtype=np.int64)
    pred = np.asarray([r["pred_score"] for r in rows], dtype=np.float64)
    gt = np.asarray([r["gt_score"] for r in rows], dtype=np.float64)
    mae = np.asarray([r["marker_mae"] for r in rows], dtype=np.float64)
    pred_thr = float(np.median(pred))
    gt_thr = float(np.median(gt))
    force_abs = np.asarray([np.nan if r["future_force_abs_mean"] is None else r["future_force_abs_mean"] for r in rows], dtype=np.float64)
    force_mag = np.asarray([np.nan if r["future_force_mag_mean"] is None else r["future_force_mag_mean"] for r in rows], dtype=np.float64)
    force_delta = np.asarray([np.nan if r["future_force_delta_abs_mean"] is None else r["future_force_delta_abs_mean"] for r in rows], dtype=np.float64)
    force_quality = np.asarray([np.nan if r["future_force_band_quality"] is None else r["future_force_band_quality"] for r in rows], dtype=np.float64)
    by_label: Dict[str, Any] = {}
    for label in sorted({r["label"] for r in rows}):
        mask = np.asarray([r["label"] == label for r in rows], dtype=bool)
        by_label[label] = {
            "n": int(mask.sum()),
            "pred_score": summarize(pred[mask]),
            "gt_score": summarize(gt[mask]),
            "marker_mae": summarize(mae[mask]),
            "force_band_quality": summarize(force_quality[mask]),
        }
    summary = {
        "n": int(len(rows)),
        "pred_auc_good": safe_auc(good, pred),
        "pred_ap_good": safe_ap(good, pred),
        "pred_balanced_accuracy_at_median": float(balanced_accuracy_score(good, pred >= pred_thr)),
        "gt_auc_good": safe_auc(good, gt),
        "gt_ap_good": safe_ap(good, gt),
        "gt_balanced_accuracy_at_median": float(balanced_accuracy_score(good, gt >= gt_thr)),
        "pred_gt_pearson": finite_corr(pred, gt),
        "pred_gt_spearman": spearman_simple(pred, gt),
        "pred_score_vs_marker_mae_spearman": spearman_simple(pred, -mae),
        "gt_score_vs_marker_mae_spearman": spearman_simple(gt, -mae),
        "pred_score_vs_force_abs_spearman": spearman_simple(pred, -force_abs),
        "pred_score_vs_force_mag_spearman": spearman_simple(pred, -np.abs(force_mag - args.force_center)),
        "pred_score_vs_force_delta_spearman": spearman_simple(pred, -force_delta),
        "pred_score_vs_force_band_quality_spearman": spearman_simple(pred, force_quality),
        "gt_score_vs_force_band_quality_spearman": spearman_simple(gt, force_quality),
        "force_band_quality_auc_good": safe_auc(good, force_quality),
        "force_band_quality": summarize(force_quality),
        "marker_mae": summarize(mae),
    }
    return {
        "purpose": "Check whether score(Foresight(action)) aligns with GT future tactile quality labels.",
        "score_mode": args.score_mode,
        "foresight": {
            "dir": args.foresight_dir,
            "ckpt": args.foresight_ckpt,
            **fs_info,
        },
        "scorer_ckpt": str(args.scorer_ckpt),
        "scorer_runtime": args.scorer_runtime,
        "force_band_quality_params": {
            "force_center": args.force_center,
            "force_sigma": args.force_sigma,
            "force_delta_ref": args.force_delta_ref,
            "marker_delta_ref": args.marker_delta_ref,
        },
        "sampling": {
            "include_260617_positive": args.include_260617_positive,
            "max_episodes_per_class": args.max_episodes_per_class,
            "samples_per_episode": args.samples_per_episode,
            "max_samples": args.max_samples,
            "contact_only": args.contact_only,
            "contact_quantile": args.contact_quantile,
        },
        "summary": summary,
        "by_label": by_label,
        "dataset_labels": selected_board_datasets(args),
        "evidence_boundary": [
            "Uses offline collection-regime labels and real future marker as proxy ground truth.",
            "Does not execute guided robot actions.",
            "If predicted score aligns with GT score/labels, the Foresight scorer is more credible for online guidance.",
        ],
    }


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fmt(v: Any) -> str:
    if v is None:
        return "NA"
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.4f}"


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    s = result["summary"]
    lines = [
        "# Foresight Score Alignment",
        "",
        "This audit checks whether `score(Foresight(action))` preserves useful quality ordering.",
        "It is not a real robot rollout metric.",
        "",
        "## Summary",
        "",
        f"- samples: `{s['n']}`",
        f"- score mode: `{result['score_mode']}`",
        f"- predicted-score AUC(good): `{fmt(s['pred_auc_good'])}`",
        f"- GT-future-score AUC(good): `{fmt(s['gt_auc_good'])}`",
        f"- predicted vs GT score Spearman: `{fmt(s['pred_gt_spearman'])}`",
        f"- predicted vs GT score Pearson: `{fmt(s['pred_gt_pearson'])}`",
        f"- predicted score vs -marker MAE Spearman: `{fmt(s['pred_score_vs_marker_mae_spearman'])}`",
        f"- predicted score vs -force abs Spearman: `{fmt(s['pred_score_vs_force_abs_spearman'])}`",
        f"- predicted score vs force-band quality Spearman: `{fmt(s['pred_score_vs_force_band_quality_spearman'])}`",
        f"- predicted score vs -force delta Spearman: `{fmt(s['pred_score_vs_force_delta_spearman'])}`",
        "",
        "## By Label",
        "",
        "| label | n | pred score mean | GT score mean | marker MAE mean | force-band quality mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, payload in result["by_label"].items():
        lines.append(
            f"| {label} | {payload['n']} | {fmt(payload['pred_score']['mean'])} | "
            f"{fmt(payload['gt_score']['mean'])} | {fmt(payload['marker_mae']['mean'])} | "
            f"{fmt(payload['force_band_quality'].get('mean'))} |"
        )
    lines.extend(["", "## Evidence Boundary", ""])
    for item in result["evidence_boundary"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_scores(rows: List[Dict[str, Any]], path: Path) -> None:
    labels = sorted({r["label"] for r in rows})
    data_pred = [[r["pred_score"] for r in rows if r["label"] == label] for label in labels]
    data_gt = [[r["gt_score"] for r in rows if r["label"] == label] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].boxplot(data_pred, labels=labels, showfliers=False)
    axes[0].set_title("score(Foresight(action))")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].boxplot(data_gt, labels=labels, showfliers=False)
    axes[1].set_title("score(GT future marker)")
    axes[1].tick_params(axis="x", rotation=25)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_ylabel("TacQuality score")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--foresight_dir", default=str(DEFAULT_FORESIGHT_DIR))
    parser.add_argument("--foresight_ckpt", default=str(DEFAULT_FORESIGHT_CKPT))
    parser.add_argument("--scorer_ckpt", type=Path, default=DEFAULT_SCORER)
    parser.add_argument("--scorer_runtime", choices=["PTGProxyScorerV2Runtime", "ForceBandTacQualityEnergyRuntime", "BoardProxyEnergyRuntime"], default="PTGProxyScorerV2Runtime")
    parser.add_argument("--score_mode", choices=["profile", "quality", "energy_clipped", "p_good", "reason_good"], default="profile")
    parser.add_argument("--include_260617_positive", action="store_true",
                        help="Add the 260617 board dataset as an additional positive regime.")
    parser.add_argument("--positive_260617_dir", default=POSITIVE_260617_DATASET)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--action_key", default="actions/joint_abs")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--max_episodes_per_class", type=int, default=12)
    parser.add_argument("--samples_per_episode", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--contact_only", action="store_true", default=True)
    parser.add_argument("--include_noncontact", dest="contact_only", action="store_false")
    parser.add_argument("--contact_quantile", type=float, default=0.35)
    parser.add_argument("--force_center", type=float, default=11.8385)
    parser.add_argument("--force_sigma", type=float, default=3.0232)
    parser.add_argument("--force_delta_ref", type=float, default=0.5776)
    parser.add_argument("--marker_delta_ref", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
