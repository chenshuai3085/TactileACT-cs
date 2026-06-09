"""Evaluate unified tactile quality labels across insertion and board wiping.

The goal is not only classification accuracy.  The final PTG use case needs a
score whose gradient can guide DP denoising toward better tactile outcomes.

This script therefore builds one shared feature/label table for two tasks:
  1. socket insertion: labels from human annotations; bad samples only from
     bounce episodes around pre-bounce / bounce / recovery.
  2. board wiping: pseudo-labels from force magnitude and smoothness.

It evaluates T4/T3/binary taxonomies with episode-level splits, cross-task
generalization, and score ranking quality.  All artifacts are saved locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "/home/chenshuai/Project/TactileACT-cs"
sys.path.insert(0, ROOT)
from TFAC_V5.tactile_vae import TactileVAE


INSERTION_DIR = "/home/chenshuai/data/dataset/0414"
BOARD_DIR = "/home/chenshuai/data/dataset/260522_v8l_caheiban"
VAE_CKPT = "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt"
OUT_DIR = "/home/chenshuai/Project/output/unified_quality_taxonomy"
TEMPORAL_WINDOW = 8

T4_NAMES = {
    0: "weak_no_contact",
    1: "good_stable",
    2: "excessive_or_risk",
    3: "rough_or_impact",
}
T3_NAMES = {
    0: "weak_no_contact",
    1: "good_stable",
    2: "bad_contact",
}
BINARY_NAMES = {0: "bad", 1: "good"}

T4_SCORE = {
    0: 0.25,  # weak contact is not failure, but not useful good contact.
    1: 1.00,
    2: 0.05,
    3: 0.00,
}


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def simple_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-8 or np.std(rb) < 1e-8:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_tactile_vae(device: torch.device):
    model = TactileVAE(latent_dim=16, temporal_window=TEMPORAL_WINDOW)
    ckpt = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        raise RuntimeError("TactileVAE checkpoint has no norm_stats")
    return model, norm_stats


def normalize_marker(marker: np.ndarray, norm_stats: dict) -> np.ndarray:
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    return ((marker - mean) / std).astype(np.float32)


def encode_marker_sequence(
    model: TactileVAE,
    marker: np.ndarray,
    norm_stats: dict,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode every frame t>=7 into TactileVAE latent using the last 8 frames."""
    marker = normalize_marker(marker, norm_stats)
    n = marker.shape[0]
    if n < TEMPORAL_WINDOW:
        return np.zeros((0, 144), dtype=np.float32)

    windows = []
    for t in range(TEMPORAL_WINDOW - 1, n):
        windows.append(marker[t - TEMPORAL_WINDOW + 1 : t + 1])
    windows = np.stack(windows, axis=0)

    latents = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[start : start + batch_size]).float().to(device)
            _, mu_last = model.encode_single_frame(x)
            latents.append(mu_last.flatten(1).cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def feature_from_latent_sequence(latents: np.ndarray, i: int, history: int = 8) -> np.ndarray:
    """Return [z_cur, z_future, delta, norm_cur, norm_future, norm_delta]."""
    cur_idx = max(0, i - history)
    z_cur = latents[cur_idx]
    z_future = latents[i]
    delta = z_future - z_cur
    norms = np.array(
        [
            np.linalg.norm(z_cur),
            np.linalg.norm(z_future),
            np.linalg.norm(delta),
        ],
        dtype=np.float32,
    )
    return np.concatenate([z_cur, z_future, delta, norms], axis=0).astype(np.float32)


def insertion_t4_label(ep_type: str, raw_label: int, frame_idx: int, lifts, pre_window: int) -> int:
    if raw_label == 0:
        return 0
    if raw_label == 1:
        label = 1
    elif raw_label == 2:
        label = 2
    elif raw_label in (3, 4):
        label = 3
    else:
        label = 0

    if ep_type == "bounce":
        for lift_start, lift_end, _ in lifts:
            if lift_start - pre_window <= frame_idx < lift_start:
                return 2
            if lift_start <= frame_idx <= lift_end:
                return 3
    return label


def build_insertion_samples(model, norm_stats, device, pre_window=10, max_per_episode=None):
    ann_path = os.path.join(INSERTION_DIR, "annotations.pkl")
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)

    samples = []
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        ep_path = os.path.join(INSERTION_DIR, f"{ep_name}.hdf5")
        if not os.path.exists(ep_path):
            continue

        with h5py.File(ep_path, "r") as f:
            marker = f["observations/tac/left/marker_offset"][:]
        latents = encode_marker_sequence(model, marker, norm_stats, device)
        labels = np.asarray(info["labels"])
        n = min(len(latents), len(labels) - (TEMPORAL_WINDOW - 1))
        if n <= 0:
            continue

        indices = np.arange(n)
        if max_per_episode and n > max_per_episode:
            rng = np.random.default_rng(abs(hash(ep_name)) % (2**32))
            indices = np.sort(rng.choice(indices, max_per_episode, replace=False))

        ep_type = info.get("type", "")
        lifts = info.get("lifts", [])
        for i in indices:
            frame_idx = i + TEMPORAL_WINDOW - 1
            raw_label = int(labels[frame_idx])
            t4 = insertion_t4_label(ep_type, raw_label, frame_idx, lifts, pre_window)
            # Enforce the user's constraint explicitly: bad classes only from bounce episodes.
            if ep_type != "bounce" and t4 in (2, 3):
                t4 = 1 if raw_label == 1 else 0
            samples.append(
                {
                    "task": "insertion",
                    "episode": ep_name,
                    "sample_id": f"insertion/{ep_name}/{frame_idx}",
                    "feature": feature_from_latent_sequence(latents, i),
                    "t4": t4,
                    "quality_score": float(T4_SCORE[t4]),
                    "raw_label": raw_label,
                    "source_label": T4_NAMES[t4],
                    "frame_start": int(max(0, frame_idx - 8)),
                    "frame_end": int(frame_idx),
                }
            )
    return samples


def safe_diff_norm(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros(1, dtype=np.float32)
    return np.linalg.norm(np.diff(x, axis=0), axis=1)


def build_board_rows(window=32, stride=16):
    files = sorted(Path(BOARD_DIR).glob("success/*.hdf5"))
    rows = []
    for path in files:
        with h5py.File(path, "r") as f:
            ft = f["ft"][:] if "ft" in f else f["observations/tac/left/force6d"][:]
            left_marker = f["observations/tac/left/marker_offset"][:]
            right_marker = f["observations/tac/right/marker_offset"][:]
            action = f["actions/eef_abs"][:]

        n = min(len(ft), len(left_marker), len(right_marker), len(action))
        ep = path.stem
        for start in range(0, max(1, n - window + 1), stride):
            end = min(n, start + window)
            if end - start < max(8, window // 2):
                continue
            ff = ft[start:end]
            lm = left_marker[start:end]
            rm = right_marker[start:end]
            aa = action[start:end]
            force_mag = np.linalg.norm(ff[:, :3], axis=1)
            marker_mag = 0.5 * (
                np.linalg.norm(lm.reshape(len(lm), -1), axis=1)
                + np.linalg.norm(rm.reshape(len(rm), -1), axis=1)
            )
            force_delta = safe_diff_norm(ff[:, :3])
            action_delta = safe_diff_norm(aa)
            marker_delta = safe_diff_norm(marker_mag[:, None])
            rows.append(
                {
                    "path": str(path),
                    "episode": ep,
                    "start": int(start),
                    "end": int(end),
                    "force_mean": float(force_mag.mean()),
                    "force_p95": float(np.percentile(force_mag, 95)),
                    "force_delta_mean": float(force_delta.mean()),
                    "force_delta_p95": float(np.percentile(force_delta, 95)),
                    "action_delta_mean": float(action_delta.mean()),
                    "marker_delta_mean": float(marker_delta.mean()),
                }
            )
    return rows


def assign_board_labels(rows):
    force = np.array([r["force_mean"] for r in rows], dtype=np.float64)
    force_p95 = np.array([r["force_p95"] for r in rows], dtype=np.float64)
    fd = np.array([r["force_delta_mean"] for r in rows], dtype=np.float64)
    ad = np.array([r["action_delta_mean"] for r in rows], dtype=np.float64)
    md = np.array([r["marker_delta_mean"] for r in rows], dtype=np.float64)

    thresholds = {
        "force_low_q20": float(np.quantile(force, 0.20)),
        "force_high_q85": float(np.quantile(force, 0.85)),
        "force_peak_q90": float(np.quantile(force_p95, 0.90)),
        "force_delta_robust_z": 1.0,
        "action_delta_robust_z": 1.0,
        "marker_delta_robust_z": 1.0,
    }

    low_force = force < thresholds["force_low_q20"]
    high_force = (force > thresholds["force_high_q85"]) | (force_p95 > thresholds["force_peak_q90"])
    rough = (robust_z(fd) > 1.0) | (robust_z(ad) > 1.0) | (robust_z(md) > 1.0)

    force_mid = np.median(force[~low_force & ~high_force])
    force_half = max((thresholds["force_high_q85"] - thresholds["force_low_q20"]) / 2, 1e-8)
    fd_z = robust_z(fd)
    ad_z = robust_z(ad)
    md_z = robust_z(md)

    labels, scores, names = [], [], []
    for i in range(len(rows)):
        if low_force[i]:
            t4 = 0
            name = "too_light"
        elif high_force[i]:
            t4 = 2
            name = "too_heavy"
        elif rough[i]:
            t4 = 3
            name = "rough"
        else:
            t4 = 1
            name = "good"

        band_score = 1.0 - min(abs(force[i] - force_mid) / force_half, 1.0)
        rough_penalty = max(0.0, fd_z[i] - 1.0, ad_z[i] - 1.0, md_z[i] - 1.0)
        peak_penalty = max(0.0, (force_p95[i] - thresholds["force_peak_q90"]) / (thresholds["force_peak_q90"] + 1e-8))
        quality = np.clip(band_score - 0.35 * rough_penalty - 0.50 * peak_penalty, -1.0, 1.0)
        if t4 == 0:
            quality = min(float(quality), 0.25)
        elif t4 in (2, 3):
            quality = min(float(quality), 0.10)

        labels.append(t4)
        scores.append(float(quality))
        names.append(name)
    return np.array(labels, dtype=np.int64), np.array(scores, dtype=np.float32), names, thresholds


def build_board_samples(model, norm_stats, device, window=32, stride=16):
    rows = build_board_rows(window=window, stride=stride)
    t4_labels, scores, names, thresholds = assign_board_labels(rows)

    marker_cache = {}
    samples = []
    for row, t4, score, name in zip(rows, t4_labels, scores, names):
        path = row["path"]
        if path not in marker_cache:
            with h5py.File(path, "r") as f:
                marker = f["observations/tac/left/marker_offset"][:]
            marker_cache[path] = encode_marker_sequence(model, marker, norm_stats, device)
        latents = marker_cache[path]
        start_lat_idx = max(0, row["start"] - (TEMPORAL_WINDOW - 1))
        end_lat_idx = min(len(latents) - 1, row["end"] - 1 - (TEMPORAL_WINDOW - 1))
        if end_lat_idx < 0 or start_lat_idx >= len(latents):
            continue
        i = max(start_lat_idx, end_lat_idx)
        samples.append(
            {
                "task": "board",
                "episode": row["episode"],
                "sample_id": f"board/{row['episode']}/{row['start']}_{row['end']}",
                "feature": feature_from_latent_sequence(latents, i),
                "t4": int(t4),
                "quality_score": float(score),
                "raw_label": name,
                "source_label": name,
                "frame_start": row["start"],
                "frame_end": row["end"],
                "force_mean": row["force_mean"],
                "force_p95": row["force_p95"],
                "force_delta_mean": row["force_delta_mean"],
                "action_delta_mean": row["action_delta_mean"],
                "marker_delta_mean": row["marker_delta_mean"],
            }
        )
    return samples, thresholds


def save_samples_csv(samples, path):
    fieldnames = [
        "sample_id",
        "task",
        "episode",
        "frame_start",
        "frame_end",
        "t4",
        "t4_name",
        "t3",
        "t3_name",
        "binary_good_bad",
        "quality_score",
        "raw_label",
        "source_label",
        "force_mean",
        "force_p95",
        "force_delta_mean",
        "action_delta_mean",
        "marker_delta_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {k: s.get(k, "") for k in fieldnames}
            row["t4_name"] = T4_NAMES[s["t4"]]
            row["t3"] = t4_to_t3(s["t4"])
            row["t3_name"] = T3_NAMES[row["t3"]]
            row["binary_good_bad"] = t4_to_binary(s["t4"])
            writer.writerow(row)


def t4_to_t3(t4: int) -> int:
    if t4 == 0:
        return 0
    if t4 == 1:
        return 1
    return 2


def t4_to_binary(t4: int):
    if t4 == 1:
        return 1
    if t4 in (2, 3):
        return 0
    return -1  # neutral/weak excluded from binary good-vs-bad evaluation.


def make_arrays(samples):
    X = np.stack([s["feature"] for s in samples]).astype(np.float32)
    y_t4 = np.array([s["t4"] for s in samples], dtype=np.int64)
    y_t3 = np.array([t4_to_t3(s["t4"]) for s in samples], dtype=np.int64)
    y_binary = np.array([t4_to_binary(s["t4"]) for s in samples], dtype=np.int64)
    quality = np.array([s["quality_score"] for s in samples], dtype=np.float32)
    task = np.array([s["task"] for s in samples])
    groups = np.array([f"{s['task']}::{s['episode']}" for s in samples])
    sample_ids = np.array([s["sample_id"] for s in samples])
    return X, y_t4, y_t3, y_binary, quality, task, groups, sample_ids


def make_models(selected=None):
    all_models = {
        "LogReg": LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=24,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            batch_size=256,
            n_iter_no_change=20,
            random_state=42,
        ),
    }
    if not selected:
        return all_models
    selected_set = {name.strip() for name in selected.split(",") if name.strip()}
    return {name: model for name, model in all_models.items() if name in selected_set}


def balanced_train_indices(y, seed=42):
    rng = np.random.default_rng(seed)
    labels = [c for c in np.unique(y) if c >= 0]
    counts = [np.sum(y == c) for c in labels]
    if not counts:
        return np.array([], dtype=np.int64)
    n = min(counts)
    idx = []
    for c in labels:
        cls = np.flatnonzero(y == c)
        idx.extend(rng.choice(cls, n, replace=False).tolist())
    idx = np.array(idx, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def class_score_from_proba(proba, classes, taxonomy):
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    if taxonomy == "t4":
        weights = {0: -0.25, 1: 1.0, 2: -0.85, 3: -1.0}
    elif taxonomy == "t3":
        weights = {0: -0.25, 1: 1.0, 2: -1.0}
    else:
        weights = {0: -1.0, 1: 1.0}
    score = np.zeros(proba.shape[0], dtype=np.float64)
    for cls, weight in weights.items():
        if cls in class_to_col:
            score += weight * proba[:, class_to_col[cls]]
    return score


def evaluate_model(model, X_train, y_train, X_test, y_test, quality_test, taxonomy):
    train_keep = y_train >= 0
    test_keep = y_test >= 0
    X_train = X_train[train_keep]
    y_train = y_train[train_keep]
    X_test = X_test[test_keep]
    y_test = y_test[test_keep]
    quality_test = quality_test[test_keep]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    bal = balanced_train_indices(y_train)
    if len(bal) == 0:
        return None

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train[bal], y_train[bal])
    pred = pipe.predict(X_test)
    out = {
        "n_train": int(len(bal)),
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }

    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        score = class_score_from_proba(proba, pipe.named_steps["clf"].classes_, taxonomy)
        out["score_quality_spearman"] = simple_spearman(score, quality_test)
        if taxonomy in {"t4", "t3"}:
            good_bad_mask = (y_test == 1) | (y_test >= 2)
            y_gb = (y_test[good_bad_mask] == 1).astype(np.int64)
            if len(np.unique(y_gb)) == 2:
                out["good_vs_bad_auc"] = float(roc_auc_score(y_gb, score[good_bad_mask]))
        elif taxonomy == "binary" and len(np.unique(y_test)) == 2:
            class_to_col = {int(c): i for i, c in enumerate(pipe.named_steps["clf"].classes_)}
            if 1 in class_to_col:
                out["good_vs_bad_auc"] = float(roc_auc_score(y_test, proba[:, class_to_col[1]]))
    return out


def aggregate_metric(rows, key):
    vals = [r[key] for r in rows if r is not None and key in r and np.isfinite(r[key])]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


def group_cv_eval(X, y, quality, groups, models, taxonomy, n_splits=5):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return {}
    cv = GroupKFold(n_splits=min(n_splits, len(unique_groups)))
    out = {}
    for name, model in models.items():
        folds = []
        for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
            row = evaluate_model(model, X[tr], y[tr], X[te], y[te], quality[te], taxonomy)
            if row is not None:
                row["fold"] = fold
                folds.append(row)
        out[name] = {
            "balanced_accuracy": aggregate_metric(folds, "balanced_accuracy"),
            "macro_f1": aggregate_metric(folds, "macro_f1"),
            "good_vs_bad_auc": aggregate_metric(folds, "good_vs_bad_auc"),
            "score_quality_spearman": aggregate_metric(folds, "score_quality_spearman"),
            "folds": folds,
        }
    return out


def cross_task_eval(X, y, quality, task, models, taxonomy, train_task, test_task):
    tr = np.flatnonzero(task == train_task)
    te = np.flatnonzero(task == test_task)
    out = {}
    for name, model in models.items():
        row = evaluate_model(model, X[tr], y[tr], X[te], y[te], quality[te], taxonomy)
        if row is not None:
            out[name] = row
    return out


def taxonomy_summary(y, task):
    out = {}
    for task_name in ["insertion", "board", "all"]:
        mask = np.ones(len(y), dtype=bool) if task_name == "all" else task == task_name
        out[task_name] = {str(k): int(v) for k, v in Counter(y[mask].tolist()).items()}
    return out


def choose_best(results):
    candidates = []
    for taxonomy, block in results["evaluations"].items():
        for model_name, metrics in block["mixed_group_cv"].items():
            ba = metrics.get("balanced_accuracy") or {}
            f1 = metrics.get("macro_f1") or {}
            auc = metrics.get("good_vs_bad_auc") or {}
            sp = metrics.get("score_quality_spearman") or {}
            cross = block["cross_task"]
            cross_scores = []
            for direction in ["insertion_to_board", "board_to_insertion"]:
                if model_name in cross.get(direction, {}):
                    cross_scores.append(cross[direction][model_name].get("macro_f1", 0.0))
            cross_mean = float(np.mean(cross_scores)) if cross_scores else 0.0
            objective = (
                0.35 * ba.get("mean", 0.0)
                + 0.25 * f1.get("mean", 0.0)
                + 0.20 * auc.get("mean", 0.0)
                + 0.10 * max(sp.get("mean", 0.0), 0.0)
                + 0.10 * cross_mean
            )
            candidates.append(
                {
                    "taxonomy": taxonomy,
                    "model": model_name,
                    "objective": float(objective),
                    "mixed_balanced_accuracy": ba.get("mean"),
                    "mixed_macro_f1": f1.get("mean"),
                    "mixed_good_vs_bad_auc": auc.get("mean"),
                    "mixed_score_spearman": sp.get("mean"),
                    "cross_task_macro_f1_mean": cross_mean,
                }
            )
    candidates.sort(key=lambda x: x["objective"], reverse=True)
    return candidates


def plot_visualizations(X, y_t4, task, quality, out_dir):
    fig_dir = Path(out_dir) / "figures"
    ensure_dir(fig_dir)

    rng = np.random.default_rng(42)
    chosen = []
    for task_name in ["insertion", "board"]:
        for cls in sorted(np.unique(y_t4)):
            idx = np.flatnonzero((task == task_name) & (y_t4 == cls))
            if len(idx) == 0:
                continue
            n = min(500, len(idx))
            chosen.extend(rng.choice(idx, n, replace=False).tolist())
    chosen = np.array(chosen, dtype=np.int64)
    X_vis = StandardScaler().fit_transform(X[chosen])
    y_vis = y_t4[chosen]
    task_vis = task[chosen]

    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(X_vis)
    tsne = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", n_iter=1000, random_state=42)
    Z_tsne = tsne.fit_transform(X_vis)

    colors = {0: "#6baed6", 1: "#31a354", 2: "#fd8d3c", 3: "#de2d26"}
    markers = {"insertion": "o", "board": "^"}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, Z, title in [
        (axes[0], Z_pca, f"PCA T4 quality space (var={pca.explained_variance_ratio_.sum():.1%})"),
        (axes[1], Z_tsne, "t-SNE T4 quality space"),
    ]:
        for cls in sorted(colors):
            for task_name in ["insertion", "board"]:
                mask = (y_vis == cls) & (task_vis == task_name)
                if not mask.any():
                    continue
                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    s=14,
                    alpha=0.55,
                    c=colors[cls],
                    marker=markers[task_name],
                    label=f"{task_name}:{T4_NAMES[cls]}",
                    edgecolors="none",
                )
        ax.set_title(title)
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    fig.savefig(fig_dir / "unified_t4_pca_tsne.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for cls in sorted(colors):
        vals = quality[y_t4 == cls]
        if len(vals):
            ax.hist(vals, bins=35, density=True, alpha=0.55, color=colors[cls], label=T4_NAMES[cls])
    ax.set_title("Continuous quality score distribution by T4 class")
    ax.set_xlabel("quality score")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "unified_quality_score_distribution.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--cache", default=os.path.join(OUT_DIR, "unified_quality_features.npz"))
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--board-window", type=int, default=32)
    parser.add_argument("--board-stride", type=int, default=16)
    parser.add_argument("--pre-bounce-window", type=int, default=10)
    parser.add_argument("--models", default="LogReg,RandomForest,MLP")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cache_path = Path(args.cache)
    metadata = {}
    if cache_path.exists() and not args.rebuild_cache:
        data = np.load(cache_path, allow_pickle=True)
        X = data["X"]
        y_t4 = data["y_t4"]
        y_t3 = data["y_t3"]
        y_binary = data["y_binary"]
        quality = data["quality"]
        task = data["task"]
        groups = data["groups"]
        sample_ids = data["sample_ids"]
        with open(Path(args.out_dir) / "class_mapping.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded cache: {cache_path}")
    else:
        print(f"Building cache on {device}...")
        model, norm_stats = load_tactile_vae(device)
        insertion_samples = build_insertion_samples(
            model, norm_stats, device, pre_window=args.pre_bounce_window
        )
        board_samples, board_thresholds = build_board_samples(
            model, norm_stats, device, window=args.board_window, stride=args.board_stride
        )
        samples = insertion_samples + board_samples
        save_samples_csv(samples, Path(args.out_dir) / "samples.csv")
        X, y_t4, y_t3, y_binary, quality, task, groups, sample_ids = make_arrays(samples)
        np.savez_compressed(
            cache_path,
            X=X,
            y_t4=y_t4,
            y_t3=y_t3,
            y_binary=y_binary,
            quality=quality,
            task=task,
            groups=groups,
            sample_ids=sample_ids,
        )
        metadata = {
            "taxonomies": {
                "t4": T4_NAMES,
                "t3": T3_NAMES,
                "binary": BINARY_NAMES,
            },
            "insertion_mapping": {
                "label_0": "weak_no_contact",
                "label_1": "good_stable",
                "label_2_or_pre_lift_window": "excessive_or_risk",
                "label_3_4_or_lift_window": "rough_or_impact",
                "constraint": "bad classes are only kept from bounce episodes",
            },
            "board_mapping": {
                "too_light": "weak_no_contact",
                "good": "good_stable",
                "too_heavy": "excessive_or_risk",
                "rough": "rough_or_impact",
                "thresholds": board_thresholds,
            },
            "feature": "[z_cur(144), z_future(144), z_delta(144), ||z_cur||, ||z_future||, ||delta||]",
            "n_samples": int(len(X)),
        }
        with open(Path(args.out_dir) / "class_mapping.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved cache: {cache_path}")

    models = make_models(args.models)
    if not models:
        raise ValueError(f"No valid models selected from --models={args.models}")
    taxonomies = {
        "t4": (y_t4, T4_NAMES),
        "t3": (y_t3, T3_NAMES),
        "binary": (y_binary, BINARY_NAMES),
    }

    results = {
        "data": {
            "n_samples": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "tasks": {k: int(v) for k, v in Counter(task.tolist()).items()},
            "n_groups": int(len(np.unique(groups))),
            "t4_counts": taxonomy_summary(y_t4, task),
            "t3_counts": taxonomy_summary(y_t3, task),
            "binary_counts_excluding_neutral": taxonomy_summary(y_binary[y_binary >= 0], task[y_binary >= 0]),
        },
        "metadata": metadata,
        "evaluations": {},
    }

    for taxonomy, (y, names) in taxonomies.items():
        block = {}
        valid = y >= 0
        block["mixed_group_cv"] = group_cv_eval(
            X[valid], y[valid], quality[valid], groups[valid], models, taxonomy
        )
        block["in_task_group_cv"] = {}
        for task_name in ["insertion", "board"]:
            mask = valid & (task == task_name)
            block["in_task_group_cv"][task_name] = group_cv_eval(
                X[mask], y[mask], quality[mask], groups[mask], models, taxonomy
            )
        block["cross_task"] = {
            "insertion_to_board": cross_task_eval(X[valid], y[valid], quality[valid], task[valid], models, taxonomy, "insertion", "board"),
            "board_to_insertion": cross_task_eval(X[valid], y[valid], quality[valid], task[valid], models, taxonomy, "board", "insertion"),
        }
        results["evaluations"][taxonomy] = block

    results["best_candidates"] = choose_best(results)
    plot_visualizations(X, y_t4, task, quality, args.out_dir)

    summary_path = Path(args.out_dir) / "unified_quality_eval.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results["data"], ensure_ascii=False, indent=2))
    print("\nBest candidates:")
    for row in results["best_candidates"][:8]:
        print(
            f"  {row['taxonomy']:6s} {row['model']:12s} obj={row['objective']:.4f} "
            f"mixed_f1={row['mixed_macro_f1']} cross_f1={row['cross_task_macro_f1_mean']:.4f}"
        )
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
