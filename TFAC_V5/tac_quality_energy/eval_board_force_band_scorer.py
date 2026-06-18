#!/usr/bin/env python3
"""Evaluate force-band-aware board wiping scorer features.

This is a CPU-only audit for the next TacQuality board scorer.  It tests
whether held-out episodes can be classified/scored from:

1. predicted-deployable marker/action proxies only, and
2. oracle future force features that approximate the upper bound if Foresight
   is extended with a force/force-band head.

The split is episode-level GroupKFold to avoid frame/window leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import warnings


warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_board_force_band_eval")
BOARD_DATASETS = {
    "positive": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "too_small": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high",
    "too_large": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low",
    "oscillate": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate",
}
REASON_TO_ID = {"too_small": 0, "positive": 1, "too_large": 2, "oscillate": 3}
ID_TO_REASON = {v: k for k, v in REASON_TO_ID.items()}


def finite_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
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


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, score))


def best_f1_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def predict_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            clf = model.named_steps.get("clf")
            classes = getattr(clf, "classes_", None)
        if classes is not None:
            classes = np.asarray(classes)
            if 1 in classes:
                return proba[:, int(np.where(classes == 1)[0][0])].astype(np.float64)
        return proba[:, -1].astype(np.float64)
    score = model.decision_function(x)
    score = np.asarray(score)
    if score.ndim > 1:
        score = score[:, -1]
    return score.astype(np.float64)


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
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def marker_proxy_np(marker_seq: np.ndarray) -> np.ndarray:
    marker_seq = np.asarray(marker_seq, dtype=np.float32)
    timesteps = marker_seq.shape[0]
    mag = np.linalg.norm(marker_seq, axis=-1)
    flat = mag.reshape(timesteps, -1)
    mean_mag_t = flat.mean(axis=-1)
    max_mag_t = flat.max(axis=-1)
    p90_mag_t = np.quantile(flat, 0.90, axis=-1)
    thresh = np.maximum(0.25 * max_mag_t, 1e-6)
    area_t = (flat > thresh[:, None]).mean(axis=-1)

    yy, xx = np.meshgrid(np.arange(9, dtype=np.float32), np.arange(9, dtype=np.float32), indexing="ij")
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)
    weights = flat + 1e-6
    wsum = weights.sum(axis=-1, keepdims=True)
    centroid = weights @ coords / wsum
    centered = coords[None, :, :] - centroid[:, None, :]
    spread = np.sqrt((weights[:, :, None] * centered**2).sum(axis=1) / wsum)

    if timesteps > 1:
        d_marker = np.linalg.norm(marker_seq[1:] - marker_seq[:-1], axis=-1).reshape(timesteps - 1, -1)
        d_centroid = np.linalg.norm(centroid[1:] - centroid[:-1], axis=-1)
        d_mean_mag = np.abs(mean_mag_t[1:] - mean_mag_t[:-1])
    else:
        d_marker = np.zeros((1, 81), dtype=np.float32)
        d_centroid = np.zeros(1, dtype=np.float32)
        d_mean_mag = np.zeros(1, dtype=np.float32)

    half = max(1, timesteps // 2)
    first_mean = np.linalg.norm(marker_seq[:half], axis=-1).mean()
    second_mean = np.linalg.norm(marker_seq[half:], axis=-1).mean()

    return np.asarray(
        [
            mean_mag_t.mean(),
            mean_mag_t.std(),
            mean_mag_t[-1],
            max_mag_t.mean(),
            max_mag_t[-1],
            p90_mag_t.mean(),
            area_t.mean(),
            area_t[-1],
            centroid[:, 0].mean() / 8.0,
            centroid[:, 1].mean() / 8.0,
            spread[:, 0].mean() / 8.0,
            spread[:, 1].mean() / 8.0,
            d_marker.mean(),
            np.quantile(d_marker.reshape(-1), 0.90),
            d_centroid.mean(),
            d_mean_mag.mean(),
            second_mean - first_mean,
            np.linalg.norm((marker_seq[-1] - marker_seq[0]).reshape(-1)),
        ],
        dtype=np.float32,
    )


def action_proxy_np(action_seq: np.ndarray, action_dim: int) -> np.ndarray:
    action_seq = np.asarray(action_seq, dtype=np.float32)
    if action_seq.shape[-1] > action_dim:
        action_seq = action_seq[..., :action_dim]
    elif action_seq.shape[-1] < action_dim:
        pad = np.zeros((action_seq.shape[0], action_dim - action_seq.shape[-1]), dtype=np.float32)
        action_seq = np.concatenate([action_seq, pad], axis=-1)
    if len(action_seq) > 1:
        delta = action_seq[1:] - action_seq[:-1]
    else:
        delta = np.zeros((1, action_dim), dtype=np.float32)
    speed = np.linalg.norm(delta, axis=-1)
    if len(delta) > 1:
        accel = delta[1:] - delta[:-1]
    else:
        accel = np.zeros((1, action_dim), dtype=np.float32)
    accel_norm = np.linalg.norm(accel, axis=-1)
    return np.asarray(
        [
            np.linalg.norm(action_seq, axis=-1).mean(),
            np.linalg.norm(action_seq, axis=-1).std(),
            speed.mean(),
            speed.std(),
            np.quantile(speed, 0.90),
            accel_norm.mean(),
            np.quantile(accel_norm, 0.90),
            np.linalg.norm(action_seq[-1] - action_seq[0]),
            np.abs(delta).mean(),
            np.abs(delta).max(),
        ],
        dtype=np.float32,
    )


def force_proxy_np(force_seq: np.ndarray) -> np.ndarray:
    force_seq = np.asarray(force_seq, dtype=np.float32)
    if force_seq.ndim != 2 or force_seq.shape[0] == 0:
        return np.zeros(18, dtype=np.float32)
    linear = force_seq[:, :3]
    torque = force_seq[:, 3:6] if force_seq.shape[1] >= 6 else np.zeros_like(linear)
    linear_norm = np.linalg.norm(linear, axis=-1)
    torque_norm = np.linalg.norm(torque, axis=-1)
    fz = force_seq[:, 2]
    if len(force_seq) > 1:
        d_linear = np.linalg.norm(np.diff(linear, axis=0), axis=-1)
        d_fz = np.abs(np.diff(fz))
        d_torque = np.linalg.norm(np.diff(torque, axis=0), axis=-1)
    else:
        d_linear = np.zeros(1, dtype=np.float32)
        d_fz = np.zeros(1, dtype=np.float32)
        d_torque = np.zeros(1, dtype=np.float32)
    return np.asarray(
        [
            linear_norm.mean(),
            linear_norm.std(),
            np.quantile(linear_norm, 0.90),
            np.abs(fz).mean(),
            fz.mean(),
            fz.std(),
            np.quantile(np.abs(fz), 0.90),
            d_linear.mean(),
            np.quantile(d_linear, 0.90),
            d_fz.mean(),
            np.quantile(d_fz, 0.90),
            torque_norm.mean(),
            torque_norm.std(),
            d_torque.mean(),
            np.abs(linear[:, 0]).mean(),
            np.abs(linear[:, 1]).mean(),
            np.abs(linear[:, 2]).mean(),
            float(len(force_seq)),
        ],
        dtype=np.float32,
    )


def window_ending(arr: np.ndarray, end: int, window: int) -> np.ndarray:
    frames = []
    for i in range(window):
        t = max(0, end - (window - 1 - i))
        frames.append(arr[t])
    return np.stack(frames).astype(np.float32)


def chunk_from(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(len(arr), start + length)
    chunk = arr[start:end]
    if len(chunk) == 0:
        chunk = arr[max(0, min(start, len(arr) - 1)) : max(0, min(start, len(arr) - 1)) + 1]
    if len(chunk) < length:
        pad = np.repeat(chunk[-1:], length - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


@dataclass
class Sample:
    label: str
    binary: int
    reason: int
    group: str
    episode: str
    start: int
    end: int
    left_marker: np.ndarray
    right_marker: np.ndarray
    joint_action: np.ndarray
    eef_action: np.ndarray
    force: np.ndarray


def sample_dataset(args: argparse.Namespace) -> List[Sample]:
    rng = np.random.default_rng(args.seed)
    rows: List[Sample] = []
    for label, root in BOARD_DATASETS.items():
        paths = sorted(Path(root).glob("episode_*.hdf5"))
        if args.max_episodes_per_class > 0:
            paths = paths[: args.max_episodes_per_class]
        for path in paths:
            try:
                with h5py.File(path, "r") as f:
                    left = f[f"observations/tac/{args.tac_side}/marker_offset"][()]
                    right_path = "observations/tac/right/marker_offset" if args.tac_side == "left" else "observations/tac/left/marker_offset"
                    right = f[right_path][()] if right_path in f else left
                    force = f[f"observations/tac/{args.tac_side}/force6d"][()]
                    joint = f[args.joint_action_key][()]
                    eef = f[args.eef_action_key][()] if args.eef_action_key in f else joint[:, :6]
            except (OSError, KeyError):
                continue

            length = min(len(left), len(right), len(force), len(joint), len(eef))
            min_start = max(args.window - 1, int(length * args.phase_start_frac))
            max_start = min(
                int(length * args.phase_end_frac),
                length - args.horizon - 1,
                length - args.action_chunk - 1,
            )
            if max_start <= min_start:
                continue
            candidates = np.arange(min_start, max_start, dtype=np.int64)
            if args.samples_per_episode > 0 and len(candidates) > args.samples_per_episode:
                starts = np.sort(rng.choice(candidates, args.samples_per_episode, replace=False))
            else:
                starts = candidates

            for start in starts:
                end = min(int(start) + args.horizon, length - 1)
                rows.append(
                    Sample(
                        label=label,
                        binary=1 if label == "positive" else 0,
                        reason=REASON_TO_ID[label],
                        group=f"{label}:{path.stem}",
                        episode=str(path),
                        start=int(start),
                        end=int(end),
                        left_marker=window_ending(left, end, args.window),
                        right_marker=window_ending(right, end, args.window),
                        joint_action=chunk_from(joint, int(start), args.action_chunk),
                        eef_action=chunk_from(eef, int(start), args.action_chunk),
                        force=window_ending(force, end, args.window),
                    )
                )
    if args.max_samples > 0 and len(rows) > args.max_samples:
        idx = np.sort(rng.choice(len(rows), args.max_samples, replace=False))
        rows = [rows[int(i)] for i in idx]
    if not rows:
        raise RuntimeError("No samples collected.")
    return rows


def build_feature_tables(samples: List[Sample]) -> Dict[str, np.ndarray]:
    left = np.stack([marker_proxy_np(s.left_marker) for s in samples])
    right = np.stack([marker_proxy_np(s.right_marker) for s in samples])
    marker_both = np.concatenate([left, right, np.abs(left - right)], axis=1)
    joint = np.stack([action_proxy_np(s.joint_action, 7) for s in samples])
    eef = np.stack([action_proxy_np(s.eef_action, 6) for s in samples])
    force = np.stack([force_proxy_np(s.force) for s in samples])
    return {
        "marker_left": left,
        "marker_both": marker_both,
        "left_marker_action": np.concatenate([left, joint, eef], axis=1),
        "marker_action": np.concatenate([marker_both, joint, eef], axis=1),
        "force_oracle": force,
        "marker_action_force_oracle": np.concatenate([marker_both, joint, eef, force], axis=1),
    }


def make_quality_target(samples: List[Sample], out_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    force_feat = np.stack([force_proxy_np(s.force) for s in samples])
    labels = np.asarray([s.label for s in samples])
    pos = labels == "positive"
    force_mag = force_feat[:, 0]
    force_delta = force_feat[:, 7]
    fz_abs = force_feat[:, 3]

    pos_force = force_mag[pos]
    pos_delta = force_delta[pos]
    if len(pos_force) == 0:
        raise RuntimeError("No positive samples for force-band target.")

    center = float(np.median(pos_force))
    mad = float(np.median(np.abs(pos_force - center)))
    sigma = max(1.4826 * mad, float(np.std(pos_force)), 0.75)
    delta_ref = max(float(np.quantile(pos_delta, 0.75)), 0.05)

    band_score = np.exp(-0.5 * ((force_mag - center) / sigma) ** 2)
    smooth_score = np.exp(-force_delta / delta_ref)

    # Use marker temporal drift as a weak smoothness term, but do not let it
    # override the force-band target.
    marker_delta = np.asarray([marker_proxy_np(s.left_marker)[12] for s in samples], dtype=np.float64)
    marker_ref = max(float(np.quantile(marker_delta[pos], 0.75)), 1e-4)
    marker_smooth = np.exp(-marker_delta / marker_ref)

    quality = 0.62 * band_score + 0.25 * smooth_score + 0.13 * marker_smooth
    quality = np.clip(quality, 0.0, 1.0).astype(np.float32)

    ref = {
        "definition": "0.62*force_band + 0.25*force_smooth + 0.13*marker_smooth",
        "force_mag_center_positive_median": center,
        "force_mag_sigma": sigma,
        "force_delta_ref_positive_q75": delta_ref,
        "marker_delta_ref_positive_q75": marker_ref,
        "force_mag_by_label": {label: summarize(force_mag[labels == label]) for label in sorted(set(labels))},
        "force_delta_by_label": {label: summarize(force_delta[labels == label]) for label in sorted(set(labels))},
        "fz_abs_by_label": {label: summarize(fz_abs[labels == label]) for label in sorted(set(labels))},
        "quality_by_label": {label: summarize(quality[labels == label]) for label in sorted(set(labels))},
    }
    (out_dir / "quality_target_reference.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    return quality, ref


def make_models(seed: int, n_jobs: int, include_mlp: bool) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=260,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        ),
    }
    if include_mlp:
        models["mlp_small"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        early_stopping=True,
                        max_iter=400,
                        alpha=1e-4,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return models


def fold_splits(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError("Need at least two episode groups.")
    return list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))


def decile_summary(score: np.ndarray, quality: np.ndarray, binary: np.ndarray, n_bins: int) -> Dict[str, Any]:
    order = np.argsort(score, kind="mergesort")
    rows = []
    for i, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": int(i),
                "n": int(len(idx)),
                "score_mean": float(np.mean(score[idx])),
                "quality_mean": float(np.mean(quality[idx])),
                "good_rate": float(np.mean(binary[idx] == 1)),
            }
        )
    q = np.asarray([r["quality_mean"] for r in rows], dtype=np.float64)
    g = np.asarray([r["good_rate"] for r in rows], dtype=np.float64)
    return {
        "bins": rows,
        "quality_positive_step_rate": float(np.mean(np.diff(q) >= -1e-8)) if len(q) > 1 else None,
        "good_rate_positive_step_rate": float(np.mean(np.diff(g) >= -1e-8)) if len(g) > 1 else None,
        "top_bottom_quality_gap": float(q[-1] - q[0]) if len(q) else None,
        "top_bottom_good_rate_gap": float(g[-1] - g[0]) if len(g) else None,
    }


def evaluate_variant(
    name: str,
    x: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
    models: Mapping[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> Dict[str, Any]:
    variant_dir = out_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    splits = fold_splits(groups, args.n_splits)

    split_manifest = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        split_manifest.append(
            {
                "fold": int(fold),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_groups": sorted(np.unique(groups[train_idx]).tolist()),
                "test_groups": sorted(np.unique(groups[test_idx]).tolist()),
            }
        )
    (variant_dir / "group_splits.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    result: Dict[str, Any] = {
        "n": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "feature_dim": int(x.shape[1]),
        "models": {},
    }

    for model_name, proto in models.items():
        oof_score = np.full(len(x), np.nan, dtype=np.float64)
        oof_binary = np.full(len(x), -1, dtype=np.int64)
        oof_reason = np.full(len(x), -1, dtype=np.int64)
        folds = []

        for fold, (train_idx, test_idx) in enumerate(splits):
            binary_model = clone(proto)
            binary_model.fit(x[train_idx], binary[train_idx])
            train_score = predict_score(binary_model, x[train_idx])
            threshold = best_f1_threshold(binary[train_idx], train_score)
            test_score = predict_score(binary_model, x[test_idx])
            test_pred = (test_score >= threshold).astype(np.int64)
            oof_score[test_idx] = test_score
            oof_binary[test_idx] = test_pred

            reason_model = clone(proto)
            reason_model.fit(x[train_idx], reason[train_idx])
            reason_pred = reason_model.predict(x[test_idx]).astype(np.int64)
            oof_reason[test_idx] = reason_pred

            folds.append(
                {
                    "fold": int(fold),
                    "threshold": float(threshold),
                    "binary_auc": safe_auc(binary[test_idx], test_score),
                    "binary_ap": safe_ap(binary[test_idx], test_score),
                    "binary_balanced_accuracy": float(balanced_accuracy_score(binary[test_idx], test_pred)),
                    "binary_macro_f1": float(f1_score(binary[test_idx], test_pred, average="macro", zero_division=0)),
                    "reason_macro_f1": float(f1_score(reason[test_idx], reason_pred, average="macro", zero_division=0)),
                    "quality_spearman": spearman_simple(test_score, quality[test_idx]),
                    "quality_pearson": finite_corr(test_score, quality[test_idx]),
                }
            )

        valid = np.isfinite(oof_score)
        cm = confusion_matrix(reason[valid], oof_reason[valid], labels=[0, 1, 2, 3])
        dec = decile_summary(oof_score[valid], quality[valid], binary[valid], args.n_bins)
        overall = {
            "binary_auc": safe_auc(binary[valid], oof_score[valid]),
            "binary_ap": safe_ap(binary[valid], oof_score[valid]),
            "binary_balanced_accuracy": float(balanced_accuracy_score(binary[valid], oof_binary[valid])),
            "binary_macro_f1": float(f1_score(binary[valid], oof_binary[valid], average="macro", zero_division=0)),
            "reason_macro_f1": float(f1_score(reason[valid], oof_reason[valid], average="macro", zero_division=0)),
            "quality_spearman": spearman_simple(oof_score[valid], quality[valid]),
            "quality_pearson": finite_corr(oof_score[valid], quality[valid]),
            "reason_confusion_matrix": cm.tolist(),
            "decile": dec,
        }
        result["models"][model_name] = {"overall": overall, "folds": folds}
        write_fold_csv(folds, variant_dir / f"{model_name}_fold_metrics.csv")
        plot_deciles(dec["bins"], f"{name}/{model_name}", variant_dir / f"{model_name}_deciles.png")
        plot_confusion(cm, f"{name}/{model_name}", variant_dir / f"{model_name}_reason_confusion.png")
    result["best_model"] = best_model_name(result)
    return result


def write_fold_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_deciles(rows: List[Dict[str, Any]], title: str, path: Path) -> None:
    if not rows:
        return
    xs = [r["bin"] for r in rows]
    q = [r["quality_mean"] for r in rows]
    g = [r["good_rate"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(xs, q, marker="o", color="#2563eb")
    axes[0].set_title("Physical quality by score decile")
    axes[0].set_ylabel("quality")
    axes[1].plot(xs, g, marker="o", color="#16a34a")
    axes[1].set_title("Good-label rate by score decile")
    axes[1].set_ylabel("good rate")
    for ax in axes:
        ax.set_xlabel("score decile, low to high")
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [ID_TO_REASON[i] for i in range(4)]
    ax.set_xticks(range(4), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(4), labels=labels)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_model_name(variant_result: Mapping[str, Any]) -> str:
    def key(item: Tuple[str, Any]) -> Tuple[float, float, float, float]:
        overall = item[1]["overall"]
        dec = overall["decile"]
        return (
            float(overall["binary_auc"] or -1.0),
            float(overall["binary_balanced_accuracy"] or -1.0),
            float(overall["reason_macro_f1"] or -1.0),
            float(overall["quality_spearman"] or -1.0),
            float(dec.get("top_bottom_quality_gap") or -1.0),
        )

    return max(variant_result["models"].items(), key=key)[0]


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Board Force-Band TacQuality Scorer Evaluation",
        "",
        "This experiment evaluates candidate features for a board-wiping tactile quality scorer.",
        "All metrics use episode-level `GroupKFold`; train/test windows never share an episode.",
        "",
        "## Purpose",
        "",
        "- `marker_*` variants approximate what is deployable today from Foresight-predicted marker fields.",
        "- `force_oracle` variants estimate the upper bound if Foresight is extended with force/force-band prediction.",
        "- The target is not generic smoothness only: quality explicitly rewards the positive force band and penalizes too-light, too-heavy, and unstable contact.",
        "",
        "## Data",
        "",
    ]
    for label, root in result["inputs"]["datasets"].items():
        lines.append(f"- {label}: `{root}`")
    lines.extend(
        [
            "",
            f"- samples: `{result['n_samples']}`",
            f"- groups/episodes: `{result['n_groups']}`",
            f"- phase fraction: `{result['protocol']['phase_start_frac']}` to `{result['protocol']['phase_end_frac']}`",
            f"- future horizon for scored marker/force window: `{result['protocol']['horizon']}`",
            "",
            "## Physical Quality Target",
            "",
            f"- definition: `{result['quality_reference']['definition']}`",
            f"- positive force-magnitude center: `{fmt(result['quality_reference']['force_mag_center_positive_median'])}`",
            f"- force-band sigma: `{fmt(result['quality_reference']['force_mag_sigma'])}`",
            f"- positive force-delta q75: `{fmt(result['quality_reference']['force_delta_ref_positive_q75'])}`",
            "",
            "Quality by collection label:",
            "",
            "| label | n | quality mean | force mag mean | force delta mean | |Fz| mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label in sorted(result["label_counts"]):
        q = result["quality_reference"]["quality_by_label"][label]
        fm = result["quality_reference"]["force_mag_by_label"][label]
        fd = result["quality_reference"]["force_delta_by_label"][label]
        fz = result["quality_reference"]["fz_abs_by_label"][label]
        lines.append(
            f"| {label} | {result['label_counts'][label]} | {fmt(q.get('mean'))} | "
            f"{fmt(fm.get('mean'))} | {fmt(fd.get('mean'))} | {fmt(fz.get('mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Best Result By Feature Variant",
            "",
            "| variant | best model | dim | AUC | AP | bACC | binary F1 | reason F1 | Spearman(q) | q top-bottom gap |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in result["variant_order"]:
        payload = result[variant]
        best = payload["best_model"]
        overall = payload["models"][best]["overall"]
        dec = overall["decile"]
        lines.append(
            f"| {variant} | {best} | {payload['feature_dim']} | {fmt(overall['binary_auc'])} | "
            f"{fmt(overall['binary_ap'])} | {fmt(overall['binary_balanced_accuracy'])} | "
            f"{fmt(overall['binary_macro_f1'])} | {fmt(overall['reason_macro_f1'])} | "
            f"{fmt(overall['quality_spearman'])} | {fmt(dec.get('top_bottom_quality_gap'))} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(result["interpretation"])
    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "",
            "- This is offline, weak-label evaluation, not real robot success.",
            "- `force_oracle` uses measured future force and is not deployable unless Foresight predicts force or the runtime has a future-force proxy.",
            "- A DP guidance-ready scorer still needs a differentiable PyTorch implementation and a Foresight-gradient audit after force-band prediction is added.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_summary(result: Mapping[str, Any], path: Path) -> None:
    variants = result["variant_order"]
    aucs = []
    baccs = []
    reason = []
    qsp = []
    for v in variants:
        payload = result[v]
        best = payload["best_model"]
        overall = payload["models"][best]["overall"]
        aucs.append(overall["binary_auc"] or np.nan)
        baccs.append(overall["binary_balanced_accuracy"] or np.nan)
        reason.append(overall["reason_macro_f1"] or np.nan)
        qsp.append(overall["quality_spearman"] or np.nan)
    x = np.arange(len(variants))
    width = 0.21
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.bar(x - 1.5 * width, aucs, width, label="AUC")
    ax.bar(x - 0.5 * width, baccs, width, label="bACC")
    ax.bar(x + 0.5 * width, reason, width, label="reason F1")
    ax.bar(x + 1.5 * width, qsp, width, label="Spearman(q)")
    ax.set_xticks(x, variants, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Best held-out-episode scorer metrics by feature variant")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = sample_dataset(args)
    labels = np.asarray([s.label for s in samples])
    binary = np.asarray([s.binary for s in samples], dtype=np.int64)
    reason = np.asarray([s.reason for s in samples], dtype=np.int64)
    groups = np.asarray([s.group for s in samples], dtype=str)
    features = build_feature_tables(samples)
    quality, quality_ref = make_quality_target(samples, out_dir)

    np.savez_compressed(
        out_dir / "board_force_band_features.npz",
        labels=labels,
        binary=binary,
        reason=reason,
        quality=quality,
        groups=groups,
        episode=np.asarray([s.episode for s in samples], dtype=object),
        start=np.asarray([s.start for s in samples], dtype=np.int64),
        end=np.asarray([s.end for s in samples], dtype=np.int64),
        **features,
    )

    models = make_models(args.seed, args.n_jobs, args.include_mlp)
    result: Dict[str, Any] = {
        "purpose": "Episode-level evaluation for force-band-aware board TacQuality scorer candidates.",
        "inputs": {"datasets": BOARD_DATASETS},
        "protocol": {
            "split": "GroupKFold by episode.",
            "n_splits": int(args.n_splits),
            "n_bins": int(args.n_bins),
            "samples_per_episode": int(args.samples_per_episode),
            "window": int(args.window),
            "horizon": int(args.horizon),
            "action_chunk": int(args.action_chunk),
            "phase_start_frac": float(args.phase_start_frac),
            "phase_end_frac": float(args.phase_end_frac),
            "models": list(models.keys()),
        },
        "n_samples": int(len(samples)),
        "n_groups": int(len(np.unique(groups))),
        "label_counts": {label: int(np.sum(labels == label)) for label in sorted(set(labels))},
        "reason_to_id": REASON_TO_ID,
        "quality_reference": quality_ref,
        "variant_order": list(features.keys()),
    }
    for variant, x in features.items():
        print(f"[variant] {variant}: n={len(x)} dim={x.shape[1]}", flush=True)
        result[variant] = evaluate_variant(variant, x.astype(np.float32), binary, reason, quality, groups, models, args, out_dir)
        print(f"  best={result[variant]['best_model']}", flush=True)

    marker_best = result["marker_action"]["models"][result["marker_action"]["best_model"]]["overall"]
    force_best = result["force_oracle"]["models"][result["force_oracle"]["best_model"]]["overall"]
    combined_best = result["marker_action_force_oracle"]["models"][result["marker_action_force_oracle"]["best_model"]]["overall"]
    interpretation = [
        f"- Best deployable marker/action-only AUC is `{fmt(marker_best['binary_auc'])}` with bACC `{fmt(marker_best['binary_balanced_accuracy'])}`.",
        f"- Force-oracle AUC is `{fmt(force_best['binary_auc'])}` with bACC `{fmt(force_best['binary_balanced_accuracy'])}`.",
        f"- Marker/action + force-oracle AUC is `{fmt(combined_best['binary_auc'])}` with bACC `{fmt(combined_best['binary_balanced_accuracy'])}`.",
    ]
    if (force_best["binary_auc"] or 0.0) > (marker_best["binary_auc"] or 0.0) + 0.03:
        interpretation.append("- Force features provide a material gain; this supports adding a force/force-band head to Foresight before final DP guidance.")
    else:
        interpretation.append("- Marker/action features are already close to force oracle for binary good/bad; force prediction may still help reason labels and score calibration.")
    if (combined_best["reason_macro_f1"] or 0.0) > (marker_best["reason_macro_f1"] or 0.0) + 0.05:
        interpretation.append("- Multi-class reason recognition benefits from force information, especially for separating too-small, too-large, and oscillation modes.")
    result["interpretation"] = interpretation

    json_path = out_dir / "board_force_band_scorer_eval.json"
    md_path = out_dir / "board_force_band_scorer_eval.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    plot_summary(result, out_dir / "board_force_band_summary.png")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--joint_action_key", default="actions/joint_abs")
    parser.add_argument("--eef_action_key", default="actions/eef_abs")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--samples_per_episode", type=int, default=12)
    parser.add_argument("--max_episodes_per_class", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--phase_start_frac", type=float, default=0.25)
    parser.add_argument("--phase_end_frac", type=float, default=0.85)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--include_mlp", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
