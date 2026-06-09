"""Evaluate marker-derived physical proxy scorer for unified tactile quality.

Motivation:
The previous latent-only unified scorer did not transfer well between socket
insertion and board wiping.  For DP guidance, however, the scorer can use
features derived from predicted tactile marker fields, not only raw VAE latent.

This script tests a more interpretable scorer input:
  marker magnitude, contact area, centroid, spatial spread, temporal smoothness,
  and short-horizon changes.

These features can later be computed from Foresight-predicted marker_offset
or decoded tactile latent, so they are closer to deployable classifier guidance
than labels that directly use future force sensors.
"""

from __future__ import annotations

import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INSERTION_DIR = Path("/home/chenshuai/data/dataset/0414")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/marker_proxy_scorer")

T4_NAMES = {
    0: "weak_no_contact",
    1: "good_stable",
    2: "excessive_or_risk",
    3: "rough_or_impact",
}


def robust_z(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def marker_proxy_features(marker_seq):
    """Compute physical proxy features from marker_offset sequence.

    Args:
        marker_seq: (T, 9, 9, 2)
    Returns:
        feature vector with intensity, area, centroid, spread, and smoothness.
    """
    marker_seq = np.asarray(marker_seq, dtype=np.float32)
    mag = np.linalg.norm(marker_seq, axis=-1)  # (T, 9, 9)
    flat = mag.reshape(len(mag), -1)
    mean_mag_t = flat.mean(axis=1)
    max_mag_t = flat.max(axis=1)
    p90_mag_t = np.percentile(flat, 90, axis=1)

    # Adaptive contact area, robust to global scale differences.
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

    return np.array(
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


FEATURE_NAMES = [
    "mag_mean",
    "mag_std",
    "mag_last",
    "mag_max_mean",
    "mag_max_last",
    "mag_p90_mean",
    "area_mean",
    "area_last",
    "centroid_x",
    "centroid_y",
    "spread_x",
    "spread_y",
    "marker_delta_mean",
    "marker_delta_p90",
    "centroid_delta_mean",
    "mag_delta_mean",
    "mag_half_change",
    "marker_first_last_l2",
]


def insertion_t4(ep_type, raw_label, frame_idx, lifts, pre_window=10):
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


def build_insertion_samples(max_per_episode=220):
    annotations = pickle.load(open(INSERTION_DIR / "annotations.pkl", "rb"))
    rows = []
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        path = INSERTION_DIR / f"{ep_name}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            marker = f["observations/tac/left/marker_offset"][:]
        labels = np.asarray(info["labels"])
        n = min(len(marker), len(labels))
        indices = np.arange(7, n)
        if max_per_episode and len(indices) > max_per_episode:
            rng = np.random.default_rng(abs(hash(ep_name)) % (2**32))
            indices = np.sort(rng.choice(indices, max_per_episode, replace=False))
        ep_type = info.get("type")
        lifts = info.get("lifts", [])
        for frame_idx in indices:
            raw = int(labels[frame_idx])
            t4 = insertion_t4(ep_type, raw, frame_idx, lifts)
            if ep_type != "bounce" and t4 in (2, 3):
                t4 = 1 if raw == 1 else 0
            window = marker[max(0, frame_idx - 7) : frame_idx + 1]
            rows.append(
                {
                    "sample_id": f"insertion/{ep_name}/{frame_idx}",
                    "task": "insertion",
                    "episode": ep_name,
                    "t4": t4,
                    "binary": 1 if t4 == 1 else (0 if t4 in (2, 3) else -1),
                    "score": {0: 0.25, 1: 1.0, 2: 0.05, 3: 0.0}[t4],
                    "raw_label": raw,
                    "feature": marker_proxy_features(window),
                }
            )
    return rows


def board_windows(window=32, stride=16):
    rows = []
    for path in sorted((BOARD_DIR / "success").glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            marker = f["observations/tac/left/marker_offset"][:]
            ft = f["ft"][:]
            action = f["actions/eef_abs"][:]
        n = min(len(marker), len(ft), len(action))
        for start in range(0, max(1, n - window + 1), stride):
            end = min(n, start + window)
            if end - start < max(8, window // 2):
                continue
            ff = ft[start:end]
            aa = action[start:end]
            force = np.linalg.norm(ff[:, :3], axis=1)
            rows.append(
                {
                    "path": str(path),
                    "episode": path.stem,
                    "start": start,
                    "end": end,
                    "marker_seq": marker[start:end],
                    "force_mean": float(force.mean()),
                    "force_p95": float(np.percentile(force, 95)),
                    "force_delta": float(np.linalg.norm(np.diff(ff[:, :3], axis=0), axis=1).mean()),
                    "action_delta": float(np.linalg.norm(np.diff(aa, axis=0), axis=1).mean()),
                    "marker_delta_label": float(
                        np.linalg.norm(np.diff(marker[start:end], axis=0), axis=-1).mean()
                    ),
                }
            )
    return rows


def assign_board_t4(rows):
    force = np.array([r["force_mean"] for r in rows])
    force_p95 = np.array([r["force_p95"] for r in rows])
    fd = np.array([r["force_delta"] for r in rows])
    ad = np.array([r["action_delta"] for r in rows])
    md = np.array([r["marker_delta_label"] for r in rows])
    thresholds = {
        "force_low_q20": float(np.quantile(force, 0.20)),
        "force_high_q85": float(np.quantile(force, 0.85)),
        "force_peak_q90": float(np.quantile(force_p95, 0.90)),
        "smooth_robust_z": 1.0,
    }
    low = force < thresholds["force_low_q20"]
    high = (force > thresholds["force_high_q85"]) | (force_p95 > thresholds["force_peak_q90"])
    rough = (robust_z(fd) > 1.0) | (robust_z(ad) > 1.0) | (robust_z(md) > 1.0)
    labels = []
    for i in range(len(rows)):
        if low[i]:
            labels.append((0, "too_light"))
        elif high[i]:
            labels.append((2, "too_heavy"))
        elif rough[i]:
            labels.append((3, "rough"))
        else:
            labels.append((1, "good"))
    return labels, thresholds


def build_board_samples():
    rows = board_windows()
    labels, thresholds = assign_board_t4(rows)
    samples = []
    for row, (t4, raw) in zip(rows, labels):
        samples.append(
            {
                "sample_id": f"board/{row['episode']}/{row['start']}_{row['end']}",
                "task": "board",
                "episode": row["episode"],
                "t4": t4,
                "binary": 1 if t4 == 1 else (0 if t4 in (2, 3) else -1),
                "score": {0: 0.25, 1: 1.0, 2: 0.05, 3: 0.0}[t4],
                "raw_label": raw,
                "feature": marker_proxy_features(row["marker_seq"]),
                "force_mean": row["force_mean"],
                "force_p95": row["force_p95"],
                "force_delta": row["force_delta"],
                "action_delta": row["action_delta"],
            }
        )
    return samples, thresholds


def save_dataset(samples, thresholds):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.stack([s["feature"] for s in samples]).astype(np.float32)
    y_t4 = np.array([s["t4"] for s in samples], dtype=np.int64)
    y_bin = np.array([s["binary"] for s in samples], dtype=np.int64)
    score = np.array([s["score"] for s in samples], dtype=np.float32)
    task = np.array([s["task"] for s in samples])
    groups = np.array([f"{s['task']}::{s['episode']}" for s in samples])
    np.savez_compressed(
        OUT_DIR / "marker_proxy_features.npz",
        X=X,
        y_t4=y_t4,
        y_binary=y_bin,
        score=score,
        task=task,
        groups=groups,
    )
    with open(OUT_DIR / "marker_proxy_samples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "task", "episode", "t4", "t4_name", "binary", "score", "raw_label"],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "task": s["task"],
                    "episode": s["episode"],
                    "t4": s["t4"],
                    "t4_name": T4_NAMES[s["t4"]],
                    "binary": s["binary"],
                    "score": s["score"],
                    "raw_label": s["raw_label"],
                }
            )
    with open(OUT_DIR / "marker_proxy_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"feature_names": FEATURE_NAMES, "board_thresholds": thresholds, "t4_names": T4_NAMES},
            f,
            ensure_ascii=False,
            indent=2,
        )
    return X, y_t4, y_bin, score, task, groups


def sample_balanced(X, y, task, groups, score, max_per_task_class=1200):
    rng = np.random.default_rng(42)
    keep = []
    for task_name in np.unique(task):
        for cls in np.unique(y):
            if cls < 0:
                continue
            idx = np.flatnonzero((task == task_name) & (y == cls))
            if len(idx):
                keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    return X[keep], y[keep], task[keep], groups[keep], score[keep]


def model_suite():
    return {
        "SGD": SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2500, tol=1e-3, class_weight="balanced", random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(96, 48), max_iter=500, early_stopping=True, random_state=42),
        "GBM": GradientBoostingClassifier(n_estimators=180, max_depth=4, random_state=42),
        "RF": RandomForestClassifier(n_estimators=220, max_depth=16, class_weight="balanced", random_state=42, n_jobs=-1),
    }


def score_from_proba(proba, classes, taxonomy):
    weights = {
        "t4": {0: -0.25, 1: 1.0, 2: -0.85, 3: -1.0},
        "binary": {0: -1.0, 1: 1.0},
    }[taxonomy]
    cls_to_col = {int(c): i for i, c in enumerate(classes)}
    out = np.zeros(len(proba), dtype=np.float64)
    for cls, weight in weights.items():
        if cls in cls_to_col:
            out += weight * proba[:, cls_to_col[cls]]
    return out


def corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fit_eval(model, Xtr, ytr, Xte, yte, score_te, taxonomy):
    valid_tr = ytr >= 0
    valid_te = yte >= 0
    Xtr, ytr = Xtr[valid_tr], ytr[valid_tr]
    Xte, yte, score_te = Xte[valid_te], yte[valid_te], score_te[valid_te]
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    row = {
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
    }
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(Xte)
        s = score_from_proba(proba, pipe.named_steps["clf"].classes_, taxonomy)
        row["score_corr"] = corr(s, score_te)
        if taxonomy == "t4":
            mask = (yte == 1) | (yte >= 2)
            y_gb = (yte[mask] == 1).astype(np.int64)
            if len(np.unique(y_gb)) == 2:
                row["good_bad_auc"] = float(roc_auc_score(y_gb, s[mask]))
        elif taxonomy == "binary":
            cls_to_col = {int(c): i for i, c in enumerate(pipe.named_steps["clf"].classes_)}
            if 1 in cls_to_col and len(np.unique(yte)) == 2:
                row["good_bad_auc"] = float(roc_auc_score(yte, proba[:, cls_to_col[1]]))
    return row


def aggregate(rows):
    rows = [r for r in rows if r]
    out = {}
    for key in ["balanced_accuracy", "macro_f1", "good_bad_auc", "score_corr"]:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def group_cv(X, y, groups, score, taxonomy, models):
    out = {}
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for name, model in models.items():
        rows = []
        for tr, te in cv.split(X, y, groups):
            rows.append(fit_eval(model, X[tr], y[tr], X[te], y[te], score[te], taxonomy))
        out[name] = aggregate(rows)
        print(f"group {taxonomy} {name}: {out[name]}", flush=True)
    return out


def cross_task(X, y, task, score, taxonomy, models, train_task, test_task):
    out = {}
    tr = task == train_task
    te = task == test_task
    for name, model in models.items():
        out[name] = fit_eval(model, X[tr], y[tr], X[te], y[te], score[te], taxonomy)
        print(f"cross {taxonomy} {train_task}->{test_task} {name}: {out[name]}", flush=True)
    return out


def plot_results(results):
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    best = []
    for taxonomy, block in results["eval"].items():
        for model, row in block["mixed_group_cv"].items():
            best.append((taxonomy, model, row.get("macro_f1", {}).get("mean", 0), row.get("good_bad_auc", {}).get("mean", 0)))
    labels = [f"{t}-{m}" for t, m, _, _ in best]
    f1 = [x[2] for x in best]
    auc = [x[3] for x in best]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, f1, width=0.36, label="macro-F1")
    ax.bar(x + 0.18, auc, width=0.36, label="good/bad AUC")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title("Marker proxy scorer mixed group-CV")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "marker_proxy_metrics.png", dpi=180)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = OUT_DIR / "marker_proxy_features.npz"
    if feature_path.exists():
        data = np.load(feature_path, allow_pickle=True)
        X = data["X"]
        y_t4 = data["y_t4"]
        y_bin = data["y_binary"]
        score = data["score"]
        task = data["task"]
        groups = data["groups"]
    else:
        insertion = build_insertion_samples()
        board, thresholds = build_board_samples()
        X, y_t4, y_bin, score, task, groups = save_dataset(insertion + board, thresholds)

    results = {
        "data": {
            "n_samples": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "task_counts": {k: int(v) for k, v in Counter(task.tolist()).items()},
            "t4_counts": {str(k): int(v) for k, v in Counter(y_t4.tolist()).items()},
            "binary_counts": {str(k): int(v) for k, v in Counter(y_bin[y_bin >= 0].tolist()).items()},
        },
        "eval": {},
    }

    models = model_suite()
    for taxonomy, y in {"t4": y_t4, "binary": y_bin}.items():
        valid = y >= 0
        Xs, ys, ts, gs, ss = sample_balanced(X[valid], y[valid], task[valid], groups[valid], score[valid])
        block = {
            "sampled_counts": {
                "all": {str(k): int(v) for k, v in Counter(ys.tolist()).items()},
                "insertion": {str(k): int(v) for k, v in Counter(ys[ts == "insertion"].tolist()).items()},
                "board": {str(k): int(v) for k, v in Counter(ys[ts == "board"].tolist()).items()},
            },
            "mixed_group_cv": group_cv(Xs, ys, gs, ss, taxonomy, models),
            "in_task_group_cv": {},
            "cross_task": {},
        }
        for task_name in ["insertion", "board"]:
            m = ts == task_name
            block["in_task_group_cv"][task_name] = group_cv(Xs[m], ys[m], gs[m], ss[m], taxonomy, models)
        block["cross_task"]["insertion_to_board"] = cross_task(Xs, ys, ts, ss, taxonomy, models, "insertion", "board")
        block["cross_task"]["board_to_insertion"] = cross_task(Xs, ys, ts, ss, taxonomy, models, "board", "insertion")
        results["eval"][taxonomy] = block

    plot_results(results)
    with open(OUT_DIR / "marker_proxy_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUT_DIR / 'marker_proxy_eval.json'}")


if __name__ == "__main__":
    main()
