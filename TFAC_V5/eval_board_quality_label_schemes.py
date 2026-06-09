"""Evaluate weak tactile-quality labels for the board-wiping task.

Goal:
  Build a scientifically useful quality target before training another scorer.
  Board wiping has no human good/bad annotation, so this script compares weak
  labels derived from force magnitude and smoothness:

    too_light / good_smooth / too_heavy / rough_force / rough_motion

Inputs deliberately exclude force sensors.  The models must predict those
quality labels from marker_offset and action features, which is the deployable
setting for DP guidance through a predicted tactile consequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/board_quality_label_schemes")

CLASS_NAMES_T4 = {
    0: "too_light",
    1: "good_smooth",
    2: "too_heavy",
    3: "rough",
}

CLASS_NAMES_T5 = {
    0: "too_light",
    1: "good_smooth",
    2: "too_heavy",
    3: "rough_force",
    4: "rough_motion",
}

FEATURE_BLOCKS = {
    "left_marker": ["left"],
    "right_marker": ["right"],
    "both_marker": ["left", "right", "lr_absdiff"],
    "both_marker_eef": ["left", "right", "lr_absdiff", "eef"],
    "both_marker_joint": ["left", "right", "lr_absdiff", "joint"],
    "both_marker_actions": ["left", "right", "lr_absdiff", "eef", "joint"],
}


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def nan_to_num(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def marker_proxy_features(marker_seq: np.ndarray) -> np.ndarray:
    marker_seq = np.asarray(marker_seq, dtype=np.float32)
    mag = np.linalg.norm(marker_seq, axis=-1)
    flat = mag.reshape(len(mag), -1)
    mean_mag_t = flat.mean(axis=1)
    max_mag_t = flat.max(axis=1)
    p90_mag_t = np.percentile(flat, 90, axis=1)

    thresh = np.maximum(1e-6, 0.25 * max_mag_t)
    area_t = (flat > thresh[:, None]).mean(axis=1)

    yy, xx = np.mgrid[0:9, 0:9]
    coords = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    weights = flat + 1e-6
    wsum = weights.sum(axis=1, keepdims=True)
    centroid = weights @ coords / wsum
    centered = coords[None, :, :] - centroid[:, None, :]
    spread = np.sqrt((weights[:, :, None] * centered**2).sum(axis=1) / wsum)

    if len(marker_seq) > 1:
        d_marker = np.linalg.norm(np.diff(marker_seq, axis=0), axis=-1).reshape(len(marker_seq) - 1, -1)
        d_centroid = np.linalg.norm(np.diff(centroid, axis=0), axis=1)
        d_mean_mag = np.abs(np.diff(mean_mag_t))
    else:
        d_marker = np.zeros((1, 81), dtype=np.float32)
        d_centroid = np.zeros(1, dtype=np.float32)
        d_mean_mag = np.zeros(1, dtype=np.float32)

    first = marker_seq[: max(1, len(marker_seq) // 2)]
    second = marker_seq[len(marker_seq) // 2 :]
    first_mean = np.linalg.norm(first, axis=-1).mean()
    second_mean = np.linalg.norm(second, axis=-1).mean()

    return nan_to_num(
        np.array(
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
                np.percentile(d_marker, 90),
                d_centroid.mean(),
                d_mean_mag.mean(),
                second_mean - first_mean,
                np.linalg.norm(marker_seq[-1] - marker_seq[0]),
            ],
            dtype=np.float32,
        )
    )


def action_proxy_features(action_seq: np.ndarray) -> np.ndarray:
    action_seq = np.asarray(action_seq, dtype=np.float32)
    if len(action_seq) < 2:
        delta = np.zeros((1, action_seq.shape[-1]), dtype=np.float32)
    else:
        delta = np.diff(action_seq, axis=0)
    speed = np.linalg.norm(delta, axis=-1)
    if len(delta) < 2:
        accel = np.zeros((1, action_seq.shape[-1]), dtype=np.float32)
    else:
        accel = np.diff(delta, axis=0)
    accel_norm = np.linalg.norm(accel, axis=-1)
    return nan_to_num(
        np.array(
            [
                np.linalg.norm(action_seq, axis=-1).mean(),
                np.linalg.norm(action_seq, axis=-1).std(),
                speed.mean(),
                speed.std(),
                np.percentile(speed, 90),
                accel_norm.mean(),
                np.percentile(accel_norm, 90),
                np.linalg.norm(action_seq[-1] - action_seq[0]),
                np.abs(delta).mean(),
                np.abs(delta).max(),
            ],
            dtype=np.float32,
        )
    )


def force_stats(force6: np.ndarray) -> dict[str, float]:
    force = np.linalg.norm(force6[:, :3], axis=1)
    if len(force) > 1:
        delta = np.abs(np.diff(force))
        jerk = np.abs(np.diff(delta)) if len(delta) > 1 else np.zeros(1, dtype=np.float32)
    else:
        delta = np.zeros(1, dtype=np.float32)
        jerk = np.zeros(1, dtype=np.float32)
    return {
        "mean": float(force.mean()),
        "std": float(force.std()),
        "p50": float(np.percentile(force, 50)),
        "p95": float(np.percentile(force, 95)),
        "max": float(force.max()),
        "delta_mean": float(delta.mean()),
        "delta_p90": float(np.percentile(delta, 90)),
        "jerk_mean": float(jerk.mean()),
    }


def marker_delta(marker: np.ndarray) -> float:
    if len(marker) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(marker, axis=0), axis=-1).mean())


def action_delta(action: np.ndarray) -> tuple[float, float]:
    if len(action) < 2:
        return 0.0, 0.0
    delta = np.diff(action, axis=0)
    speed = np.linalg.norm(delta, axis=-1)
    accel = np.diff(delta, axis=0) if len(delta) > 1 else np.zeros((1, action.shape[-1]), dtype=np.float32)
    return float(speed.mean()), float(np.linalg.norm(accel, axis=-1).mean())


def load_or_build_windows(args) -> tuple[np.lib.npyio.NpzFile, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / f"board_windows_w{args.window}_s{args.stride}.npz"
    meta_path = OUT_DIR / f"board_windows_w{args.window}_s{args.stride}_meta.json"
    sample_path = OUT_DIR / f"board_windows_w{args.window}_s{args.stride}.csv"
    if data_path.exists() and not args.force_rebuild:
        return np.load(data_path, allow_pickle=True), json.load(open(meta_path, encoding="utf-8"))

    paths = sorted((BOARD_DIR / "success").glob("*.hdf5"))
    rows = []
    features = {name: [] for name in ["left", "right", "lr_absdiff", "eef", "joint"]}
    for path in paths:
        with h5py.File(path, "r") as f:
            left = f["observations/tac/left/marker_offset"][:]
            right = f["observations/tac/right/marker_offset"][:]
            ft = f["ft"][:]
            left_force = f["observations/tac/left/force6d"][:]
            right_force = f["observations/tac/right/force6d"][:]
            eef = f["actions/eef_abs"][:]
            joint = f["actions/joint_abs"][:]
        n = min(len(left), len(right), len(ft), len(left_force), len(right_force), len(eef), len(joint))
        for start in range(0, max(1, n - args.window + 1), args.stride):
            end = min(n, start + args.window)
            if end - start < max(8, args.window // 2):
                continue
            lseq = left[start:end]
            rseq = right[start:end]
            eseq = eef[start:end]
            jseq = joint[start:end]
            ft_stats = force_stats(ft[start:end])
            lf_stats = force_stats(left_force[start:end])
            rf_stats = force_stats(right_force[start:end])
            eef_d, eef_a = action_delta(eseq)
            joint_d, joint_a = action_delta(jseq)
            row = {
                "sample_id": f"{path.stem}/{start}_{end}",
                "episode": path.stem,
                "start": start,
                "end": end,
                "left_marker_delta": marker_delta(lseq),
                "right_marker_delta": marker_delta(rseq),
                "eef_delta": eef_d,
                "eef_accel": eef_a,
                "joint_delta": joint_d,
                "joint_accel": joint_a,
            }
            for prefix, stats in [("ft", ft_stats), ("left_force", lf_stats), ("right_force", rf_stats)]:
                for k, v in stats.items():
                    row[f"{prefix}_{k}"] = v
            rows.append(row)
            features["left"].append(marker_proxy_features(lseq))
            features["right"].append(marker_proxy_features(rseq))
            features["lr_absdiff"].append(np.abs(features["left"][-1] - features["right"][-1]))
            features["eef"].append(action_proxy_features(eseq))
            features["joint"].append(action_proxy_features(jseq))

    arrays = {
        "sample_id": np.array([r["sample_id"] for r in rows]),
        "episode": np.array([r["episode"] for r in rows]),
        "start": np.array([r["start"] for r in rows], dtype=np.int32),
        "end": np.array([r["end"] for r in rows], dtype=np.int32),
    }
    numeric_keys = sorted(k for k in rows[0] if k not in {"sample_id", "episode"})
    for key in numeric_keys:
        arrays[key] = np.array([r[key] for r in rows], dtype=np.float32)
    for key, vals in features.items():
        arrays[f"X_{key}"] = np.stack(vals).astype(np.float32)
    np.savez_compressed(data_path, **arrays)

    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "episode"] + numeric_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    meta = {
        "board_dir": str(BOARD_DIR),
        "n_episodes": len(paths),
        "n_windows": len(rows),
        "window": args.window,
        "stride": args.stride,
        "feature_blocks": {k: list(v) for k, v in FEATURE_BLOCKS.items()},
        "note": "Force is used only to define weak labels. Model inputs are marker/action proxy features.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.load(data_path, allow_pickle=True), meta


def continuous_quality(data, force_source: str) -> tuple[np.ndarray, dict]:
    f = data[f"{force_source}_mean"].astype(np.float64)
    p95 = data[f"{force_source}_p95"].astype(np.float64)
    fd = data[f"{force_source}_delta_mean"].astype(np.float64)
    fj = data[f"{force_source}_jerk_mean"].astype(np.float64)
    md = np.maximum(data["left_marker_delta"], data["right_marker_delta"]).astype(np.float64)
    ad = data["eef_accel"].astype(np.float64)

    target = float(np.quantile(f, 0.55))
    sigma = float(max(np.quantile(f, 0.75) - np.quantile(f, 0.25), 1e-6))
    force_band = np.exp(-0.5 * ((f - target) / sigma) ** 2)
    high_peak = np.maximum(0.0, robust_z(p95) - 0.5)
    rough = (
        np.maximum(0.0, robust_z(fd))
        + 0.50 * np.maximum(0.0, robust_z(fj))
        + 0.35 * np.maximum(0.0, robust_z(md))
        + 0.25 * np.maximum(0.0, robust_z(ad))
        + 0.50 * high_peak
    )
    smooth = np.exp(-0.55 * rough)
    score = np.clip(force_band * smooth, 0.0, 1.0).astype(np.float32)
    params = {"target_force_q55": target, "force_iqr_sigma": sigma}
    return score, params


def assign_labels(data, force_source: str, scheme: str) -> tuple[np.ndarray, np.ndarray, dict]:
    f = data[f"{force_source}_mean"].astype(np.float64)
    p95 = data[f"{force_source}_p95"].astype(np.float64)
    fd = data[f"{force_source}_delta_mean"].astype(np.float64)
    md = np.maximum(data["left_marker_delta"], data["right_marker_delta"]).astype(np.float64)
    ad = data["eef_accel"].astype(np.float64)
    score, params = continuous_quality(data, force_source)

    thresholds = {
        **params,
        "force_q20": float(np.quantile(f, 0.20)),
        "force_q25": float(np.quantile(f, 0.25)),
        "force_q30": float(np.quantile(f, 0.30)),
        "force_q80": float(np.quantile(f, 0.80)),
        "force_q85": float(np.quantile(f, 0.85)),
        "force_p95_q90": float(np.quantile(p95, 0.90)),
        "rough_force_z1": 1.0,
        "rough_motion_z1": 1.0,
        "score_good_q65": float(np.quantile(score, 0.65)),
        "score_bad_q35": float(np.quantile(score, 0.35)),
    }

    if scheme == "t4_quantile":
        low = f < thresholds["force_q20"]
        high = (f > thresholds["force_q85"]) | (p95 > thresholds["force_p95_q90"])
        rough = (robust_z(fd) > 1.0) | (robust_z(md) > 1.0) | (robust_z(ad) > 1.0)
        y = np.full(len(f), 1, dtype=np.int64)
        y[rough] = 3
        y[high] = 2
        y[low] = 0
    elif scheme == "t5_reason":
        low = f < thresholds["force_q25"]
        high = (f > thresholds["force_q80"]) | (p95 > thresholds["force_p95_q90"])
        rough_force = robust_z(fd) > 1.0
        rough_motion = (robust_z(md) > 1.0) | (robust_z(ad) > 1.0)
        y = np.full(len(f), 1, dtype=np.int64)
        y[rough_motion] = 4
        y[rough_force] = 3
        y[high] = 2
        y[low] = 0
    elif scheme == "t5_scoreband":
        low = f < thresholds["force_q30"]
        high = (f > thresholds["force_q80"]) | (p95 > thresholds["force_p95_q90"])
        rough_force = (robust_z(fd) > 0.75) | (robust_z(data[f"{force_source}_jerk_mean"]) > 0.75)
        rough_motion = (robust_z(md) > 0.75) | (robust_z(ad) > 0.75)
        good = score >= thresholds["score_good_q65"]
        y = np.full(len(f), 1, dtype=np.int64)
        y[~good & rough_motion] = 4
        y[~good & rough_force] = 3
        y[high] = 2
        y[low] = 0
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    return y, score, thresholds


def build_input(data, feature_set: str) -> np.ndarray:
    blocks = []
    for name in FEATURE_BLOCKS[feature_set]:
        blocks.append(data[f"X_{name}"].astype(np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def balanced_indices(y: np.ndarray, groups: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        if len(idx) > max_per_class:
            idx = rng.choice(idx, max_per_class, replace=False)
        keep.extend(idx.tolist())
    return np.array(sorted(keep), dtype=np.int64)


def classifier_suite(seed: int, names: str):
    all_models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=260,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gbm": GradientBoostingClassifier(n_estimators=220, max_depth=3, learning_rate=0.045, random_state=seed),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    selected = [n.strip() for n in names.split(",") if n.strip()]
    return {name: all_models[name] for name in selected}


def regressor_suite(seed: int, names: str):
    all_models = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "rf": RandomForestRegressor(
            n_estimators=260, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=-1
        ),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        max_iter=500,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    selected = [n.strip() for n in names.split(",") if n.strip()]
    return {name: all_models[name] for name in selected}


def score_from_proba(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    weights = {0: -0.35, 1: 1.0, 2: -1.0, 3: -0.85, 4: -0.75}
    out = np.zeros(len(proba), dtype=np.float64)
    cls_to_col = {int(c): i for i, c in enumerate(classes)}
    for cls, weight in weights.items():
        if cls in cls_to_col:
            out += weight * proba[:, cls_to_col[cls]]
    return out


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    good = (y == 1).astype(np.int64)
    if len(np.unique(good)) != 2:
        return None
    return float(roc_auc_score(good, s))


def eval_classifier(X: np.ndarray, y: np.ndarray, quality: np.ndarray, groups: np.ndarray, model, folds: int):
    cv = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    rows = []
    for tr, te in cv.split(X, y, groups):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        row = {
            "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
            "macro_f1": float(f1_score(y[te], pred, average="macro")),
        }
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X[te])
            classes = model.classes_ if hasattr(model, "classes_") else model.named_steps["clf"].classes_
            s = score_from_proba(proba, classes)
            row["good_auc"] = safe_auc(y[te], s)
            row["score_corr"] = float(np.corrcoef(s, quality[te])[0, 1]) if np.std(s) > 1e-8 else 0.0
        rows.append(row)
    return aggregate_rows(rows)


def eval_regressor(X: np.ndarray, quality: np.ndarray, groups: np.ndarray, model, folds: int):
    cv = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    rows = []
    for tr, te in cv.split(X, quality, groups):
        model.fit(X[tr], quality[tr])
        pred = np.asarray(model.predict(X[te]), dtype=np.float64)
        rows.append(
            {
                "quality_corr": float(np.corrcoef(pred, quality[te])[0, 1]) if np.std(pred) > 1e-8 else 0.0,
                "quality_r2": float(r2_score(quality[te], pred)),
            }
        )
    return aggregate_rows(rows)


def aggregate_rows(rows: list[dict]) -> dict:
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def run_grid(data, args):
    groups_all = data["episode"]
    rows = []
    details = {}
    for force_source in args.force_sources.split(","):
        force_source = force_source.strip()
        for scheme in args.schemes.split(","):
            scheme = scheme.strip()
            y_all, q_all, thresholds = assign_labels(data, force_source, scheme)
            max_per_class = args.max_per_class
            keep = balanced_indices(y_all, groups_all, max_per_class=max_per_class, seed=args.seed)
            y = y_all[keep]
            q = q_all[keep]
            groups = groups_all[keep]
            class_counts = {str(k): int(v) for k, v in Counter(y.tolist()).items()}
            for feature_set in args.feature_sets.split(","):
                feature_set = feature_set.strip()
                X = build_input(data, feature_set)[keep]
                for model_name, model in classifier_suite(args.seed, args.class_models).items():
                    print(
                        f"class {force_source}/{scheme}/{feature_set}/{model_name}",
                        flush=True,
                    )
                    metrics = eval_classifier(X, y, q, groups, model, args.folds)
                    row = {
                        "kind": "classification",
                        "force_source": force_source,
                        "scheme": scheme,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n": int(len(X)),
                        "dim": int(X.shape[1]),
                        "class_counts": class_counts,
                        **{k: v["mean"] for k, v in metrics.items()},
                        **{f"{k}_std": v["std"] for k, v in metrics.items()},
                    }
                    rows.append(row)
                    details[f"{force_source}/{scheme}/{feature_set}/{model_name}"] = {
                        "thresholds": thresholds,
                        "class_counts": class_counts,
                        "metrics": metrics,
                    }
                for model_name, model in regressor_suite(args.seed, args.reg_models).items():
                    print(
                        f"reg {force_source}/{scheme}/{feature_set}/{model_name}",
                        flush=True,
                    )
                    metrics = eval_regressor(X, q, groups, model, args.folds)
                    row = {
                        "kind": "regression",
                        "force_source": force_source,
                        "scheme": scheme,
                        "feature_set": feature_set,
                        "model": model_name,
                        "n": int(len(X)),
                        "dim": int(X.shape[1]),
                        "class_counts": class_counts,
                        **{k: v["mean"] for k, v in metrics.items()},
                        **{f"{k}_std": v["std"] for k, v in metrics.items()},
                    }
                    rows.append(row)
                    details[f"{force_source}/{scheme}/{feature_set}/reg_{model_name}"] = {
                        "thresholds": thresholds,
                        "class_counts": class_counts,
                        "metrics": metrics,
                    }
    return rows, details


def row_sort_key(row: dict) -> tuple:
    if row["kind"] == "classification":
        return (
            row.get("good_auc", -math.inf),
            row.get("macro_f1", -math.inf),
            row.get("score_corr", -math.inf),
        )
    return (
        row.get("quality_corr", -math.inf),
        row.get("quality_r2", -math.inf),
        -math.inf,
    )


def plot_best_space(data, best_row: dict, args, result_dir: Path):
    force_source = best_row["force_source"]
    scheme = best_row["scheme"]
    feature_set = best_row["feature_set"]
    y_all, q_all, thresholds = assign_labels(data, force_source, scheme)
    groups_all = data["episode"]
    keep = balanced_indices(y_all, groups_all, max_per_class=args.max_per_class, seed=args.seed)
    X = build_input(data, feature_set)[keep]
    y = y_all[keep]
    q = q_all[keep]
    sample_ids = data["sample_id"][keep]

    rng = np.random.default_rng(args.seed)
    if len(X) > args.viz_max:
        viz_idx = np.sort(rng.choice(len(X), args.viz_max, replace=False))
    else:
        viz_idx = np.arange(len(X))
    Xs = StandardScaler().fit_transform(X[viz_idx])
    yv = y[viz_idx]
    qv = q[viz_idx]

    fig_dir = result_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    class_names = CLASS_NAMES_T5 if y.max() >= 4 else CLASS_NAMES_T4

    pca = PCA(n_components=2, random_state=args.seed).fit_transform(Xs)
    plot_embedding(pca, yv, qv, class_names, fig_dir / "best_pca_classes.png", "PCA class space")

    perplexity = min(35, max(5, len(Xs) // 30))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=args.seed).fit_transform(Xs)
    plot_embedding(tsne, yv, qv, class_names, fig_dir / "best_tsne_classes.png", "t-SNE class space")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cls in sorted(np.unique(y)):
        ax.hist(q[y == cls], bins=28, alpha=0.55, density=True, label=class_names[int(cls)])
    ax.set_xlabel("continuous weak quality")
    ax.set_ylabel("density")
    ax.set_title("Quality score distribution by weak class")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "best_quality_hist.png", dpi=180)
    plt.close(fig)

    with open(result_dir / "best_visualized_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "class", "class_name", "quality"])
        writer.writeheader()
        for sid, cls, score in zip(sample_ids, y, q):
            writer.writerow({"sample_id": sid, "class": int(cls), "class_name": class_names[int(cls)], "quality": float(score)})

    return thresholds


def plot_embedding(emb, y, q, class_names, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    cmap = plt.get_cmap("tab10")
    for cls in sorted(np.unique(y)):
        idx = y == cls
        ax.scatter(emb[idx, 0], emb[idx, 1], s=13, alpha=0.72, color=cmap(int(cls)), label=class_names[int(cls)])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=q, s=13, alpha=0.8, cmap="viridis")
    ax.set_title(title.replace("class", "quality"))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(sc, ax=ax, label="weak quality")
    fig.tight_layout()
    fig.savefig(out_path.with_name(out_path.stem.replace("_classes", "_quality") + ".png"), dpi=180)
    plt.close(fig)


def save_rows(rows: list[dict], result_dir: Path):
    keys = sorted({k for r in rows for k in r if k != "class_counts"})
    csv_path = result_dir / "board_quality_scheme_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys + ["class_counts_json"])
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k, "") for k in keys}
            out["class_counts_json"] = json.dumps(row.get("class_counts", {}), ensure_ascii=False)
            writer.writerow(out)
    return csv_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_per_class", type=int, default=900)
    parser.add_argument("--viz_max", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--force_sources", default="ft,right_force,left_force")
    parser.add_argument("--schemes", default="t4_quantile,t5_reason,t5_scoreband")
    parser.add_argument("--feature_sets", default="left_marker,right_marker,both_marker,both_marker_eef,both_marker_joint,both_marker_actions")
    parser.add_argument("--class_models", default="logreg,rf")
    parser.add_argument("--reg_models", default="ridge,rf")
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = OUT_DIR / f"w{args.window}_s{args.stride}"
    result_dir.mkdir(parents=True, exist_ok=True)
    data, meta = load_or_build_windows(args)
    rows, details = run_grid(data, args)
    rows_sorted = sorted(rows, key=row_sort_key, reverse=True)
    class_rows = [r for r in rows_sorted if r["kind"] == "classification"]
    reg_rows = [r for r in rows_sorted if r["kind"] == "regression"]
    best_class = class_rows[0]
    best_reg = reg_rows[0]
    best_thresholds = plot_best_space(data, best_class, args, result_dir)
    save_rows(rows_sorted, result_dir)

    result = {
        "meta": meta,
        "args": vars(args),
        "best_classification": best_class,
        "best_regression": best_reg,
        "best_thresholds": best_thresholds,
        "top_classification": class_rows[:15],
        "top_regression": reg_rows[:10],
        "details": details,
    }
    out_path = result_dir / "board_quality_scheme_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_classification": best_class, "best_regression": best_reg}, ensure_ascii=False, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
