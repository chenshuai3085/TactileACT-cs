"""Compare scorer designs for DP tactile classifier guidance.

This is a scorer-design experiment, not a rollout gate.  It compares common
choices for the energy/scoring function that could guide DP denoising:

  - binary good/bad classifier;
  - multi-class reason classifier converted to a scalar score;
  - continuous quality regressor;
  - pairwise ranking scorer;
  - existing action-aware unified scorer results, if available.

All supervised metrics use episode-level GroupKFold.  The output is meant to
answer which scorer *form* is most suitable for differentiable guidance, not to
claim real robot improvement.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_board_quality_label_schemes import (
    assign_labels as assign_board_labels,
    build_input as build_board_input,
)


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_scorer_design_comparison")
INSERTION_CACHE = Path("/home/chenshuai/Project/output/ptg_quality_eval/insert_prebounce_episode_cache.npz")
BOARD_CACHE = Path("/home/chenshuai/Project/output/board_quality_label_schemes/board_windows_w32_s16.npz")
ACTION_AWARE_EVAL = Path("/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_eval.json")
BOARD_POS_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609")
BOARD_NEG_LIGHT_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def summarize(values: List[float]) -> Dict[str, float]:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(vals.mean()), "std": float(vals.std())}


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    val = spearmanr(a, b).correlation
    return float(0.0 if val is None or not np.isfinite(val) else val)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) != 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def aggregate_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for row in rows for k in row})
    return {key: summarize([row[key] for row in rows if key in row]) for key in keys}


def scalar_from_class_proba(proba: np.ndarray, classes: np.ndarray, weights: Dict[int, float]) -> np.ndarray:
    out = np.zeros(proba.shape[0], dtype=np.float64)
    class_to_col = {int(cls): i for i, cls in enumerate(classes)}
    for cls, weight in weights.items():
        if cls in class_to_col:
            out += weight * proba[:, class_to_col[cls]]
    return out


def pairwise_accuracy(score: np.ndarray, quality: np.ndarray, rng: np.random.Generator, n_pairs: int = 2000) -> float:
    n = len(score)
    if n < 2:
        return float("nan")
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = np.abs(quality[i] - quality[j]) > 1e-6
    if keep.sum() == 0:
        return float("nan")
    i, j = i[keep], j[keep]
    return float(np.mean((score[i] > score[j]) == (quality[i] > quality[j])))


def guidance_suitability(
    *,
    design: str,
    score_corr: float,
    good_auc: float,
    pair_acc: float,
    differentiable: bool,
    smooth_score: bool,
    reason_aware: bool,
) -> Dict[str, Any]:
    suitability = 0.0
    suitability += 0.30 * np.nan_to_num(score_corr, nan=0.0)
    suitability += 0.25 * np.nan_to_num(good_auc, nan=0.5)
    suitability += 0.20 * np.nan_to_num(pair_acc, nan=0.5)
    suitability += 0.15 if differentiable else 0.0
    suitability += 0.07 if smooth_score else 0.0
    suitability += 0.03 if reason_aware else 0.0
    caveats = []
    if design == "binary":
        caveats.append("good for risk/good separation, but probability/logit can saturate and gives weak reason labels")
    if design == "multiclass":
        caveats.append("reason-aware and interpretable, but scalar guidance depends on class-to-score weights")
    if design == "regression":
        caveats.append("direct continuous energy for guidance, but label quality matters")
    if design == "pairwise":
        caveats.append("strong ranking objective, but needs pointwise energy extraction for gradient guidance")
    return {
        "score": float(suitability),
        "differentiable_energy_ready": bool(differentiable and smooth_score),
        "reason_aware": bool(reason_aware),
        "caveats": caveats,
    }


def binary_model(seed: int):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
        ]
    )


def multiclass_model(seed: int):
    return RandomForestClassifier(
        n_estimators=260,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def regression_model():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "reg",
                MLPRegressor(
                    hidden_layer_sizes=(96, 48),
                    max_iter=350,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def pairwise_model(seed: int):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(96, 48),
                    max_iter=350,
                    early_stopping=True,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_pairwise_point_scorer(
    X: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    max_pairs: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    tr = train_idx
    n_pairs = min(max_pairs, max(1, len(tr) * 4))
    a = rng.choice(tr, size=n_pairs, replace=True)
    b = rng.choice(tr, size=n_pairs, replace=True)
    keep = np.abs(quality[a] - quality[b]) > 1e-6
    a, b = a[keep], b[keep]
    if len(a) < 100:
        return np.zeros(len(test_idx), dtype=np.float64), {"train_pair_acc": float("nan")}

    X_pair = X[a] - X[b]
    y_pair = (quality[a] > quality[b]).astype(np.int64)
    # Add reversed pairs so the learned decision boundary is antisymmetric.
    X_pair = np.concatenate([X_pair, -X_pair], axis=0)
    y_pair = np.concatenate([y_pair, 1 - y_pair], axis=0)

    clf = pairwise_model(seed)
    clf.fit(X_pair, y_pair)
    train_prob = clf.predict_proba(X_pair)[:, 1]
    train_acc = float(np.mean((train_prob >= 0.5) == y_pair))

    # Convert pairwise preference into a pointwise score by comparing each test
    # sample against a fixed set of training anchors.
    anchors = rng.choice(tr, size=min(256, len(tr)), replace=False)
    scores = []
    for start in range(0, len(test_idx), 256):
        idx = test_idx[start : start + 256]
        diffs = X[idx, None, :] - X[anchors][None, :, :]
        flat = diffs.reshape(-1, X.shape[1])
        prob = clf.predict_proba(flat)[:, 1].reshape(len(idx), len(anchors))
        scores.append(prob.mean(axis=1))
    return np.concatenate(scores), {"train_pair_acc": train_acc}


def evaluate_designs_for_dataset(
    *,
    task: str,
    X: np.ndarray,
    y_binary: np.ndarray,
    y_multi: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
    class_score_weights: Dict[int, float],
    seed: int,
    folds: int,
    max_pairs: int,
) -> Dict[str, Any]:
    cv = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    rng = np.random.default_rng(seed)
    rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in ["binary", "multiclass", "regression", "pairwise"]}

    for fold, (tr, te) in enumerate(cv.split(X, y_binary, groups)):
        binary = binary_model(seed + fold)
        binary.fit(X[tr], y_binary[tr])
        p_good = binary.predict_proba(X[te])[:, list(binary.named_steps["clf"].classes_).index(1)]
        pred = (p_good >= 0.5).astype(np.int64)
        rows["binary"].append(
            {
                "balanced_accuracy": float(balanced_accuracy_score(y_binary[te], pred)),
                "good_auc": safe_auc(y_binary[te], p_good),
                "score_corr": safe_spearman(p_good, quality[te]),
                "pairwise_accuracy": pairwise_accuracy(p_good, quality[te], rng),
            }
        )

        multi = multiclass_model(seed + fold)
        multi.fit(X[tr], y_multi[tr])
        pred_multi = multi.predict(X[te])
        proba = multi.predict_proba(X[te])
        score_multi = scalar_from_class_proba(proba, multi.classes_, class_score_weights)
        rows["multiclass"].append(
            {
                "balanced_accuracy": float(balanced_accuracy_score(y_multi[te], pred_multi)),
                "macro_f1": float(f1_score(y_multi[te], pred_multi, average="macro")),
                "good_auc": safe_auc(y_binary[te], score_multi),
                "score_corr": safe_spearman(score_multi, quality[te]),
                "pairwise_accuracy": pairwise_accuracy(score_multi, quality[te], rng),
            }
        )

        reg = regression_model()
        reg.fit(X[tr], quality[tr])
        q_pred = np.asarray(reg.predict(X[te]), dtype=np.float64)
        rows["regression"].append(
            {
                "quality_rmse": float(math.sqrt(mean_squared_error(quality[te], q_pred))),
                "quality_r2": float(r2_score(quality[te], q_pred)),
                "good_auc": safe_auc(y_binary[te], q_pred),
                "score_corr": safe_spearman(q_pred, quality[te]),
                "pairwise_accuracy": pairwise_accuracy(q_pred, quality[te], rng),
            }
        )

        pair_score, pair_info = fit_pairwise_point_scorer(
            X, quality, groups, tr, te, seed + fold, max_pairs=max_pairs
        )
        rows["pairwise"].append(
            {
                "train_pair_acc": pair_info["train_pair_acc"],
                "good_auc": safe_auc(y_binary[te], pair_score),
                "score_corr": safe_spearman(pair_score, quality[te]),
                "pairwise_accuracy": pairwise_accuracy(pair_score, quality[te], rng),
            }
        )

    designs = {}
    for name, fold_rows in rows.items():
        metrics = aggregate_rows(fold_rows)
        mean = lambda key, default=float("nan"): metrics.get(key, {}).get("mean", default)
        designs[name] = {
            "metrics": metrics,
            "guidance_suitability": guidance_suitability(
                design=name,
                score_corr=mean("score_corr"),
                good_auc=mean("good_auc"),
                pair_acc=mean("pairwise_accuracy"),
                differentiable=name in {"binary", "multiclass", "regression", "pairwise"},
                smooth_score=name in {"regression", "pairwise"},
                reason_aware=name == "multiclass",
            ),
        }

    ranked = sorted(
        [(name, item["guidance_suitability"]["score"]) for name, item in designs.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "task": task,
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "n_groups": int(len(np.unique(groups))),
        "class_counts_binary": {str(k): int(v) for k, v in zip(*np.unique(y_binary, return_counts=True))},
        "class_counts_multi": {str(k): int(v) for k, v in zip(*np.unique(y_multi, return_counts=True))},
        "designs": designs,
        "ranking_by_guidance_suitability": ranked,
    }


def load_insertion() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(INSERTION_CACHE, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_bad = data["y"].astype(np.int64)
    # Cache convention: 0=insert/good, 1=pre-bounce/bad.  This script uses the
    # common scorer convention 1=good for binary AUC and score correlation.
    y_binary = (1 - y_bad).astype(np.int64)
    # Two classes are all the current insertion cache contains: insert and pre-bounce.
    # We keep multiclass identical here and explicitly report that insertion still
    # needs richer labels for reason-aware multi-class scoring.
    y_multi = y_bad.copy()
    quality = y_binary.astype(np.float32)
    groups = data["groups"]
    return X, y_binary, y_multi, quality, groups


def load_board(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if args.board_label_source == "manual_dirs":
        return load_board_manual_dirs(args)
    data = np.load(BOARD_CACHE, allow_pickle=True)
    y_multi, quality, thresholds = assign_board_labels(data, args.board_force_source, args.board_scheme)
    X = build_board_input(data, args.board_feature_set).astype(np.float32)
    y_binary = (y_multi == 1).astype(np.int64)
    groups = data["episode"]
    return X, y_binary, y_multi, quality.astype(np.float32), groups, thresholds


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


def force_audit_features(force6: np.ndarray) -> Dict[str, float]:
    f = np.linalg.norm(force6[:, :3], axis=1)
    delta = np.abs(np.diff(f)) if len(f) > 1 else np.zeros(1, dtype=np.float32)
    jerk = np.abs(np.diff(delta)) if len(delta) > 1 else np.zeros(1, dtype=np.float32)
    return {
        "force_mean": float(f.mean()),
        "force_p95": float(np.percentile(f, 95)),
        "force_delta_mean": float(delta.mean()),
        "force_jerk_mean": float(jerk.mean()),
    }


def contact_segments(force6: np.ndarray, min_len: int, q: float = 0.60) -> List[Tuple[int, int]]:
    """Estimate board-contact/wiping segments from force magnitude.

    The manual board labels differ mainly during contact with the board.  This
    keeps approach/retreat frames out of scorer training so the model does not
    learn trajectory setup artifacts.
    """
    f = np.linalg.norm(force6[:, :3], axis=1)
    if len(f) == 0:
        return []
    threshold = max(float(np.quantile(f, q)), float(np.median(f) + 0.25 * np.std(f)))
    mask = f >= threshold
    segs = []
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if not segs:
        # Fall back to the middle part of the episode.  This is still better
        # than using approach/retreat windows when force thresholding is noisy.
        margin = max(0, len(f) // 5)
        if len(f) - 2 * margin >= min_len:
            segs = [(margin, len(f) - margin)]
    return segs


def load_board_manual_dirs(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    rows = []
    features = []
    episode_contact_summary = []
    label_dirs = [
        ("good_smooth", 1, 1.0, Path(args.board_positive_dir)),
        ("too_light_unclean", 0, 0.0, Path(args.board_negative_light_dir)),
    ]
    for label_name, binary, quality, root in label_dirs:
        for path in sorted(root.glob("*.hdf5")):
            with h5py.File(path, "r") as f:
                left = f["observations/tac/left/marker_offset"][:]
                right = f["observations/tac/right/marker_offset"][:]
                eef = f["actions/eef_abs"][:]
                joint = f["actions/joint_abs"][:]
                ft = f["ft"][:]
            n = min(len(left), len(right), len(eef), len(joint), len(ft))
            segs = [(0, n)]
            if args.board_contact_only:
                segs = contact_segments(ft[:n], min_len=max(args.board_window, 8), q=args.board_contact_quantile)
            n_windows_before = len(rows)
            for seg_start, seg_end in segs:
                if seg_end - seg_start < max(8, args.board_window // 2):
                    continue
                for start in range(seg_start, max(seg_start + 1, seg_end - args.board_window + 1), args.board_stride):
                    end = min(seg_end, start + args.board_window)
                    if end - start < max(8, args.board_window // 2):
                        continue
                    lseq = left[start:end]
                    rseq = right[start:end]
                    feat = np.concatenate(
                        [
                            marker_proxy_features(lseq),
                            marker_proxy_features(rseq),
                            np.abs(marker_proxy_features(lseq) - marker_proxy_features(rseq)),
                            action_proxy_features(eef[start:end]),
                            action_proxy_features(joint[start:end]),
                        ]
                    )
                    features.append(feat)
                    audit = force_audit_features(ft[start:end])
                    rows.append(
                        {
                            "episode": f"{root.name}/{path.stem}",
                            "sample_id": f"{root.name}/{path.stem}/{start}_{end}",
                            "label_name": label_name,
                            "binary": binary,
                            "multi": 1 if binary == 1 else 0,
                            "quality": quality,
                            "contact_only": bool(args.board_contact_only),
                            **audit,
                        }
                    )
            episode_contact_summary.append(
                {
                    "episode": f"{root.name}/{path.stem}",
                    "label_name": label_name,
                    "n_steps": int(n),
                    "segments": [[int(a), int(b)] for a, b in segs],
                    "n_windows": int(len(rows) - n_windows_before),
                }
            )
    if not rows:
        raise RuntimeError("No board manual-dir HDF5 windows found")
    X = np.stack(features).astype(np.float32)
    y_binary = np.array([r["binary"] for r in rows], dtype=np.int64)
    y_multi = np.array([r["multi"] for r in rows], dtype=np.int64)
    quality = np.array([r["quality"] for r in rows], dtype=np.float32)
    groups = np.array([r["episode"] for r in rows])
    thresholds = {
        "label_source": "manual_dirs",
        "positive_dir": str(args.board_positive_dir),
        "negative_light_dir": str(args.board_negative_light_dir),
        "n_positive_episodes": len(list(Path(args.board_positive_dir).glob("*.hdf5"))),
        "n_negative_light_episodes": len(list(Path(args.board_negative_light_dir).glob("*.hdf5"))),
        "n_windows": len(rows),
        "contact_only": bool(args.board_contact_only),
        "contact_quantile": float(args.board_contact_quantile),
        "episode_contact_summary_head": episode_contact_summary[:20],
        "label_meaning": {
            "good_smooth": "positive samples: force magnitude suitable and changes smooth",
            "too_light_unclean": "negative samples: force too small / wipe not clean; heavy and unstable negatives not included yet",
        },
        "current_multiclass_status": (
            "Only two manual classes are available now: good_smooth and too_light_unclean. "
            "The multiclass design slot is a reason-aware classifier scaffold; it becomes a true multi-class "
            "reason classifier after adding too-heavy and unstable-force negative directories."
        ),
        "force_audit": {
            "positive_force_mean": float(np.mean([r["force_mean"] for r in rows if r["binary"] == 1])),
            "negative_force_mean": float(np.mean([r["force_mean"] for r in rows if r["binary"] == 0])),
        },
    }
    return X, y_binary, y_multi, quality, groups, thresholds


def load_action_aware_summary() -> Dict[str, Any]:
    if not ACTION_AWARE_EVAL.exists():
        return {"available": False}
    data = json.load(open(ACTION_AWARE_EVAL, encoding="utf-8"))
    return {
        "available": True,
        "source": str(ACTION_AWARE_EVAL),
        "mixed_group_cv": {
            "binary_auc": data.get("mixed_group_cv", {}).get("binary_auc", {}).get("mean"),
            "binary_balanced_accuracy": data.get("mixed_group_cv", {}).get("binary_balanced_accuracy", {}).get("mean"),
            "score_corr": data.get("mixed_group_cv", {}).get("score_corr", {}).get("mean"),
            "t4_macro_f1": data.get("mixed_group_cv", {}).get("t4_macro_f1", {}).get("mean"),
        },
        "cross_task": data.get("cross_task", {}),
        "interpretation": (
            "Existing action-aware scorer is the strongest unified scorer evidence, "
            "but its zero-shot cross-task hard classification remains weak; use "
            "task-conditioned calibration for guidance."
        ),
    }


def make_recommendation(insertion: Dict[str, Any], board: Dict[str, Any], action_aware: Dict[str, Any]) -> Dict[str, Any]:
    best_insertion = insertion["ranking_by_guidance_suitability"][0]
    best_board = board["ranking_by_guidance_suitability"][0]
    return {
        "recommended_main_design": "continuous_energy_with_aux_reason_heads",
        "why": [
            "Regression/pairwise-style continuous scores are smoother energy functions for DP gradients than hard classes.",
            "Multiclass reason heads remain useful as auxiliary losses because they separate too-light, too-heavy, rough, and collision-risk modes.",
            "Binary good/bad should be retained as a calibrated risk margin, not as the only score.",
            "Action-aware/task-conditioned inputs are important because insertion and board quality do not transfer well under a single hard threshold.",
        ],
        "implementation_sketch": {
            "network": "shared encoder over predicted tactile/action features with three heads",
            "heads": [
                "quality_regression_head: scalar q in [0,1] for gradient energy",
                "reason_classification_head: task-specific bad-mode class probabilities",
                "binary_margin_head: calibrated P(good) or logit margin",
            ],
            "guidance_energy": "E = w_q * q + w_margin * logit_good - w_risk * expected_bad_reason_cost",
            "dp_use": "differentiate E through Foresight(action -> predicted tactile/action features) and apply trust-region line search",
        },
        "current_best_by_task": {
            "insertion": {"design": best_insertion[0], "suitability": best_insertion[1]},
            "board": {"design": best_board[0], "suitability": best_board[1]},
        },
        "action_aware_note": action_aware.get("interpretation"),
        "not_recommended_as_only_design": [
            "pure binary classifier only",
            "pure multiclass classifier without continuous scalar calibration",
            "pure pairwise scorer without a stable pointwise energy extraction",
        ],
    }


def plot_summary(result: Dict[str, Any], out_dir: Path) -> None:
    labels = ["binary", "multiclass", "regression", "pairwise"]
    metrics = ["good_auc", "score_corr", "pairwise_accuracy"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, task in zip(axes, ["insertion", "board"]):
        x = np.arange(len(labels))
        width = 0.24
        for i, metric in enumerate(metrics):
            vals = [
                result["tasks"][task]["designs"][label]["metrics"].get(metric, {}).get("mean", np.nan)
                for label in labels
            ]
            ax.bar(x + (i - 1) * width, vals, width, label=metric)
        ax.set_title(task)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("episode-level GroupKFold mean")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "scorer_design_metric_comparison.png", dpi=180)
    plt.close(fig)


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Scorer Design Comparison for DP Guidance",
        "",
        f"- git_commit: `{result['git_commit']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- primary_protocol: {result['primary_protocol']}",
        "",
        "## Recommendation",
        "",
        f"- recommended_main_design: `{result['recommendation']['recommended_main_design']}`",
        "",
    ]
    for item in result["recommendation"]["why"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Metrics", ""])
    for task, block in result["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- n: `{block['n']}`",
                f"- n_groups: `{block['n_groups']}`",
                "",
                "| design | suitability | good_auc | score_corr | pairwise_acc | bal_acc / r2 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for design, item in block["designs"].items():
            m = item["metrics"]
            suit = item["guidance_suitability"]["score"]
            acc = m.get("balanced_accuracy", m.get("quality_r2", {})).get("mean", float("nan"))
            lines.append(
                "| {d} | {s:.4f} | {auc:.4f} | {corr:.4f} | {pair:.4f} | {acc:.4f} |".format(
                    d=design,
                    s=suit,
                    auc=m.get("good_auc", {}).get("mean", float("nan")),
                    corr=m.get("score_corr", {}).get("mean", float("nan")),
                    pair=m.get("pairwise_accuracy", {}).get("mean", float("nan")),
                    acc=acc,
                )
            )
        lines.extend(["", f"- ranking: `{block['ranking_by_guidance_suitability']}`", ""])
        if task == "board":
            label_def = block.get("label_definition", {})
            thresholds = label_def.get("thresholds", {})
            lines.extend(
                [
                    "#### Board Label Scope",
                    "",
                    f"- label_source: `{label_def.get('source')}`",
                    f"- positive_dir: `{thresholds.get('positive_dir')}`",
                    f"- negative_light_dir: `{thresholds.get('negative_light_dir')}`",
                    f"- contact_only: `{thresholds.get('contact_only')}`",
                    f"- n_positive_episodes: `{thresholds.get('n_positive_episodes')}`",
                    f"- n_negative_light_episodes: `{thresholds.get('n_negative_light_episodes')}`",
                    f"- current_multiclass_status: {thresholds.get('current_multiclass_status')}",
                    f"- caveat: {label_def.get('negative_coverage_caveat')}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Implementation Sketch",
            "",
            "```text",
            json.dumps(result["recommendation"]["implementation_sketch"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Caveat",
            "",
            "This compares scorer forms on cached offline data. It does not prove real DP rollout improvement.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_pairs", type=int, default=8000)
    parser.add_argument("--board_label_source", choices=["manual_dirs", "weak_force"], default="manual_dirs")
    parser.add_argument("--board_positive_dir", type=Path, default=BOARD_POS_DIR)
    parser.add_argument("--board_negative_light_dir", type=Path, default=BOARD_NEG_LIGHT_DIR)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.set_defaults(board_contact_only=True)
    parser.add_argument("--board_contact_only", dest="board_contact_only", action="store_true")
    parser.add_argument("--no_board_contact_only", dest="board_contact_only", action="store_false")
    parser.add_argument("--board_contact_quantile", type=float, default=0.60)
    parser.add_argument("--board_force_source", default="left_force")
    parser.add_argument("--board_scheme", default="t5_scoreband")
    parser.add_argument("--board_feature_set", default="both_marker_actions")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ins_X, ins_y, ins_multi, ins_q, ins_groups = load_insertion()
    board_X, board_y, board_multi, board_q, board_groups, board_thresholds = load_board(args)

    insertion = evaluate_designs_for_dataset(
        task="insertion",
        X=ins_X,
        y_binary=ins_y,
        y_multi=ins_multi,
        quality=ins_q,
        groups=ins_groups,
        class_score_weights={0: 1.0, 1: -1.0},
        seed=args.seed,
        folds=args.folds,
        max_pairs=args.max_pairs,
    )
    insertion["label_note"] = (
        "Current cache is insert vs pre-bounce only; richer insertion reason classes need a separate cache."
    )
    board = evaluate_designs_for_dataset(
        task="board",
        X=board_X,
        y_binary=board_y,
        y_multi=board_multi,
        quality=board_q,
        groups=board_groups,
        class_score_weights={0: -0.35, 1: 1.0, 2: -1.0, 3: -0.85, 4: -0.75},
        seed=args.seed + 100,
        folds=args.folds,
        max_pairs=args.max_pairs,
    )
    board["label_definition"] = {
        "source": args.board_label_source,
        "scheme": args.board_scheme,
        "force_source": args.board_force_source,
        "feature_set": args.board_feature_set,
        "thresholds": board_thresholds,
        "contact_window_note": (
            "Manual board positives/negatives are evaluated only on estimated board-contact/wiping windows, "
            "because their semantic difference is during wiping contact, not approach or retreat."
        )
        if args.board_label_source == "manual_dirs"
        else None,
        "negative_coverage_caveat": (
            "Current manual negative board data only covers too-light / unclean wiping. "
            "Future too-heavy and force-unstable negatives should be added before claiming full bad-contact coverage."
        )
        if args.board_label_source == "manual_dirs"
        else None,
    }
    action_aware = load_action_aware_summary()

    result = {
        "purpose": "Compare scorer forms for DP tactile classifier guidance.",
        "scientific_evidence": "offline_scorer_design_evidence_not_real_rollout",
        "git_commit": git_commit(),
        "primary_protocol": "episode-level GroupKFold; no frame-level random splits",
        "inputs": {
            "insertion_cache": str(INSERTION_CACHE),
            "board_cache": str(BOARD_CACHE),
            "board_positive_dir": str(args.board_positive_dir),
            "board_negative_light_dir": str(args.board_negative_light_dir),
            "action_aware_eval": str(ACTION_AWARE_EVAL),
        },
        "tasks": {
            "insertion": insertion,
            "board": board,
        },
        "existing_action_aware_unified_scorer": action_aware,
        "recommendation": make_recommendation(insertion, board, action_aware),
    }

    json_path = args.output_dir / "scorer_design_comparison.json"
    md_path = args.output_dir / "scorer_design_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    plot_summary(result, args.output_dir)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "insertion_best": insertion["ranking_by_guidance_suitability"][0],
                "board_best": board["ranking_by_guidance_suitability"][0],
                "recommended_main_design": result["recommendation"]["recommended_main_design"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
