#!/usr/bin/env python3
"""Evaluate board TacQuality features after adding 260617 as a positive set.

The old board scorer was trained with 260609/260610 labeled regimes.  The
260617-only board data is intended as a new positive policy/data distribution,
but it is shifted in force magnitude.  This script asks whether a scorer can
jointly recognize old positive, new positive, too-small, too-large, and
oscillatory contact under episode-level splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.eval_board_force_band_scorer import (  # noqa: E402
    action_proxy_np,
    force_proxy_np,
    marker_proxy_np,
    safe_ap,
    safe_auc,
    finite_corr,
    spearman_simple,
    summarize,
    window_ending,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_board_force_band_with_260617_positive")
DATASETS = {
    "positive_old": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "positive_260617": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617",
    "too_small": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high",
    "too_large": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low",
    "oscillate": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate",
}
REASON_TO_ID = {
    "too_small": 0,
    "positive_old": 1,
    "positive_260617": 2,
    "too_large": 3,
    "oscillate": 4,
}
ID_TO_REASON = {v: k for k, v in REASON_TO_ID.items()}


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


def chunk_from(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(len(arr), start + length)
    chunk = arr[start:end]
    if len(chunk) == 0:
        idx = max(0, min(start, len(arr) - 1))
        chunk = arr[idx : idx + 1]
    if len(chunk) < length:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], length - len(chunk), axis=0)], axis=0)
    return chunk.astype(np.float32)


def collect_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    rows: List[Dict[str, Any]] = []
    for label, root in DATASETS.items():
        for path in sorted(Path(root).glob("episode_*.hdf5")):
            try:
                with h5py.File(path, "r") as f:
                    left = f[f"observations/tac/{args.tac_side}/marker_offset"][()].astype(np.float32)
                    other = "right" if args.tac_side == "left" else "left"
                    right_key = f"observations/tac/{other}/marker_offset"
                    right = f[right_key][()].astype(np.float32) if right_key in f else left
                    force = f[f"observations/tac/{args.tac_side}/force6d"][()].astype(np.float32)
                    joint = f[args.joint_action_key][()].astype(np.float32)
                    eef = f[args.eef_action_key][()].astype(np.float32) if args.eef_action_key in f else joint[:, :6]
            except (OSError, KeyError):
                continue

            length = min(len(left), len(right), len(force), len(joint), len(eef))
            min_start = max(args.window - 1, int(length * args.phase_start_frac))
            max_start = min(int(length * args.phase_end_frac), length - args.horizon - 1, length - args.action_chunk - 1)
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
                    {
                        "label": label,
                        "binary": 1 if label.startswith("positive") else 0,
                        "reason": REASON_TO_ID[label],
                        "group": f"{label}:{path.stem}",
                        "episode": str(path),
                        "start": int(start),
                        "end": int(end),
                        "left_marker": window_ending(left, end, args.window),
                        "right_marker": window_ending(right, end, args.window),
                        "force": window_ending(force, end, args.window),
                        "joint_action": chunk_from(joint, int(start), args.action_chunk),
                        "eef_action": chunk_from(eef, int(start), args.action_chunk),
                    }
                )
    if not rows:
        raise RuntimeError("No samples collected.")
    return rows


def build_features(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    left = np.stack([marker_proxy_np(r["left_marker"]) for r in rows])
    right = np.stack([marker_proxy_np(r["right_marker"]) for r in rows])
    marker_both = np.concatenate([left, right, np.abs(left - right)], axis=1)
    joint = np.stack([action_proxy_np(r["joint_action"], 7) for r in rows])
    eef = np.stack([action_proxy_np(r["eef_action"], 6) for r in rows])
    force = np.stack([force_proxy_np(r["force"]) for r in rows])
    return {
        "marker_left": left,
        "marker_both": marker_both,
        "left_marker_action": np.concatenate([left, joint, eef], axis=1),
        "marker_action": np.concatenate([marker_both, joint, eef], axis=1),
        "force_oracle": force,
        "marker_action_force_oracle": np.concatenate([marker_both, joint, eef, force], axis=1),
    }


def make_quality(rows: List[Dict[str, Any]], out_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    labels = np.asarray([r["label"] for r in rows], dtype=str)
    force = np.stack([force_proxy_np(r["force"]) for r in rows])
    marker_delta = np.asarray([marker_proxy_np(r["left_marker"])[12] for r in rows], dtype=np.float64)
    force_mag = force[:, 0].astype(np.float64)
    force_delta = force[:, 7].astype(np.float64)
    fz_abs = force[:, 3].astype(np.float64)
    pos = np.char.startswith(labels, "positive")
    pos_force = force_mag[pos]
    pos_delta = force_delta[pos]
    if len(pos_force) == 0:
        raise RuntimeError("No positive samples.")
    center = float(np.median(pos_force))
    mad = float(np.median(np.abs(pos_force - center)))
    sigma = max(1.4826 * mad, float(np.std(pos_force)), 0.75)
    delta_ref = max(float(np.quantile(pos_delta, 0.75)), 0.05)
    marker_ref = max(float(np.quantile(marker_delta[pos], 0.75)), 1e-4)
    band = np.exp(-0.5 * ((force_mag - center) / sigma) ** 2)
    smooth = np.exp(-force_delta / delta_ref)
    marker_smooth = np.exp(-marker_delta / marker_ref)
    quality = np.clip(0.62 * band + 0.25 * smooth + 0.13 * marker_smooth, 0.0, 1.0).astype(np.float32)
    ref = {
        "definition": "0.62*positive_union_force_band + 0.25*force_smooth + 0.13*marker_smooth",
        "positive_labels": ["positive_old", "positive_260617"],
        "force_mag_center_positive_union_median": center,
        "force_mag_sigma": sigma,
        "force_delta_ref_positive_union_q75": delta_ref,
        "marker_delta_ref_positive_union_q75": marker_ref,
        "force_mag_by_label": {label: summarize(force_mag[labels == label]) for label in sorted(set(labels.tolist()))},
        "force_delta_by_label": {label: summarize(force_delta[labels == label]) for label in sorted(set(labels.tolist()))},
        "fz_abs_by_label": {label: summarize(fz_abs[labels == label]) for label in sorted(set(labels.tolist()))},
        "quality_by_label": {label: summarize(quality[labels == label]) for label in sorted(set(labels.tolist()))},
    }
    (out_dir / "quality_target_reference.json").write_text(json.dumps(ref, indent=2, ensure_ascii=False), encoding="utf-8")
    return quality, ref


def save_runtime_compatible_features(
    out_dir: Path,
    labels: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
    rows: List[Dict[str, Any]],
    features: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Save a 4-reason feature file compatible with ForceBand runtime."""

    compat_reason = np.empty_like(reason)
    mapping = {
        "too_small": 0,
        "positive_old": 1,
        "positive_260617": 1,
        "too_large": 2,
        "oscillate": 3,
    }
    for label, reason_id in mapping.items():
        compat_reason[labels == label] = reason_id

    # Keep binary goodness explicit, because the union force-band target can
    # rate smooth too-small contact highly.  This is a guidance target, not a
    # pure force-band physics oracle.
    guidance_quality = np.clip(0.70 * binary.astype(np.float32) + 0.30 * quality.astype(np.float32), 0.0, 1.0)

    path = out_dir / "board_force_band_with_260617_features_compat4.npz"
    np.savez_compressed(
        path,
        labels=labels,
        binary=binary,
        reason=compat_reason.astype(np.int64),
        reason5=reason.astype(np.int64),
        quality=guidance_quality.astype(np.float32),
        quality_physical_union=quality.astype(np.float32),
        groups=groups,
        episode=np.asarray([r["episode"] for r in rows], dtype=object),
        start=np.asarray([r["start"] for r in rows], dtype=np.int64),
        end=np.asarray([r["end"] for r in rows], dtype=np.int64),
        **features,
    )
    return {
        "path": str(path),
        "reason_mapping": mapping,
        "quality_definition": "0.70*binary_good + 0.30*positive_union_physical_quality",
        "quality_by_label": {label: summarize(guidance_quality[labels == label]) for label in sorted(set(labels.tolist()))},
    }


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
        raise ValueError("Need at least two groups.")
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


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [ID_TO_REASON[i] for i in range(len(ID_TO_REASON))]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_deciles(rows: List[Dict[str, Any]], title: str, path: Path) -> None:
    if not rows:
        return
    x = [r["bin"] for r in rows]
    q = [r["quality_mean"] for r in rows]
    g = [r["good_rate"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(x, q, marker="o")
    axes[0].set_title("Physical quality by score decile")
    axes[1].plot(x, g, marker="o", color="#16a34a")
    axes[1].set_title("Good rate by score decile")
    for ax in axes:
        ax.set_xlabel("score decile")
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_fold_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def best_model_name(result: Mapping[str, Any]) -> str:
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

    return max(result["models"].items(), key=key)[0]


def evaluate_variant(
    name: str,
    x: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    labels: np.ndarray,
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

    result: Dict[str, Any] = {"n": int(len(x)), "feature_dim": int(x.shape[1]), "models": {}}
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
                    "positive_260617_recall": float(np.mean(test_pred[labels[test_idx] == "positive_260617"] == 1)) if np.any(labels[test_idx] == "positive_260617") else None,
                    "positive_old_recall": float(np.mean(test_pred[labels[test_idx] == "positive_old"] == 1)) if np.any(labels[test_idx] == "positive_old") else None,
                }
            )

        valid = np.isfinite(oof_score)
        cm = confusion_matrix(reason[valid], oof_reason[valid], labels=list(range(len(ID_TO_REASON))))
        dec = decile_summary(oof_score[valid], quality[valid], binary[valid], args.n_bins)
        per_label_binary_recall = {
            label: float(np.mean(oof_binary[(labels == label) & valid] == binary[(labels == label) & valid]))
            for label in sorted(set(labels.tolist()))
            if np.any((labels == label) & valid)
        }
        overall = {
            "binary_auc": safe_auc(binary[valid], oof_score[valid]),
            "binary_ap": safe_ap(binary[valid], oof_score[valid]),
            "binary_balanced_accuracy": float(balanced_accuracy_score(binary[valid], oof_binary[valid])),
            "binary_macro_f1": float(f1_score(binary[valid], oof_binary[valid], average="macro", zero_division=0)),
            "reason_macro_f1": float(f1_score(reason[valid], oof_reason[valid], average="macro", zero_division=0)),
            "quality_spearman": spearman_simple(oof_score[valid], quality[valid]),
            "quality_pearson": finite_corr(oof_score[valid], quality[valid]),
            "positive_260617_recall": float(np.mean(oof_binary[(labels == "positive_260617") & valid] == 1)),
            "positive_old_recall": float(np.mean(oof_binary[(labels == "positive_old") & valid] == 1)),
            "per_label_binary_recall": per_label_binary_recall,
            "reason_confusion_matrix": cm.tolist(),
            "decile": dec,
        }
        result["models"][model_name] = {"overall": overall, "folds": folds}
        write_fold_csv(folds, variant_dir / f"{model_name}_fold_metrics.csv")
        plot_confusion(cm, f"{name}/{model_name}", variant_dir / f"{model_name}_reason_confusion.png")
        plot_deciles(dec["bins"], f"{name}/{model_name}", variant_dir / f"{model_name}_deciles.png")
    result["best_model"] = best_model_name(result)
    return result


def plot_summary(result: Mapping[str, Any], path: Path) -> None:
    variants = result["variant_order"]
    aucs, baccs, reason, qsp, new_recall = [], [], [], [], []
    for v in variants:
        payload = result[v]
        best = payload["best_model"]
        overall = payload["models"][best]["overall"]
        aucs.append(overall["binary_auc"] or np.nan)
        baccs.append(overall["binary_balanced_accuracy"] or np.nan)
        reason.append(overall["reason_macro_f1"] or np.nan)
        qsp.append(overall["quality_spearman"] or np.nan)
        new_recall.append(overall["positive_260617_recall"] or np.nan)
    x = np.arange(len(variants))
    width = 0.16
    fig, ax = plt.subplots(figsize=(13, 5), dpi=150)
    ax.bar(x - 2 * width, aucs, width, label="AUC")
    ax.bar(x - width, baccs, width, label="bACC")
    ax.bar(x, reason, width, label="reason F1")
    ax.bar(x + width, qsp, width, label="Spearman(q)")
    ax.bar(x + 2 * width, new_recall, width, label="260617 recall")
    ax.set_xticks(x, variants, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_title("Board scorer with 260617 positive: held-out episode metrics")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Board ForceBand Scorer With 260617 Positive",
        "",
        "目的：把 260617-only 数据作为新的 positive 分布加入，检查 scorer 是否能同时识别旧 positive、新 positive 和旧负样本。",
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
            f"- split: `{result['protocol']['split']}`",
            f"- runtime-compatible features: `{result['runtime_compatible_features']['path']}`",
            "",
            "## Physical Quality Target",
            "",
            f"- definition: `{result['quality_reference']['definition']}`",
            f"- positive-union force center: `{fmt(result['quality_reference']['force_mag_center_positive_union_median'])}`",
            f"- force sigma: `{fmt(result['quality_reference']['force_mag_sigma'])}`",
            f"- runtime quality target: `{result['runtime_compatible_features']['quality_definition']}`",
            "",
            "| label | n | quality mean | force_mag mean | force_delta mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in sorted(result["label_counts"]):
        q = result["quality_reference"]["quality_by_label"][label]
        fm = result["quality_reference"]["force_mag_by_label"][label]
        fd = result["quality_reference"]["force_delta_by_label"][label]
        lines.append(
            f"| {label} | {result['label_counts'][label]} | {fmt(q.get('mean'))} | "
            f"{fmt(fm.get('mean'))} | {fmt(fd.get('mean'))} |"
        )
    lines.extend(
        [
            "",
            "## Best Result By Feature Variant",
            "",
            "| variant | best model | dim | AUC | bACC | binary F1 | reason F1 | Spearman(q) | old pos recall | 260617 pos recall |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in result["variant_order"]:
        payload = result[variant]
        best = payload["best_model"]
        overall = payload["models"][best]["overall"]
        lines.append(
            f"| {variant} | {best} | {payload['feature_dim']} | {fmt(overall['binary_auc'])} | "
            f"{fmt(overall['binary_balanced_accuracy'])} | {fmt(overall['binary_macro_f1'])} | "
            f"{fmt(overall['reason_macro_f1'])} | {fmt(overall['quality_spearman'])} | "
            f"{fmt(overall['positive_old_recall'])} | {fmt(overall['positive_260617_recall'])} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(result["interpretation"])
    lines.extend(
        [
            "",
            "## Evidence Boundaries",
            "",
            "- This is offline episode-level evaluation, not real robot rollout evidence.",
            "- Adding 260617 as positive tests scorer calibration; it does not prove guided DP improves force curves.",
            "- A deployable PyTorch checkpoint should be trained from this feature set only after selecting the best score target/mode.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args)
    labels = np.asarray([r["label"] for r in rows], dtype=str)
    binary = np.asarray([r["binary"] for r in rows], dtype=np.int64)
    reason = np.asarray([r["reason"] for r in rows], dtype=np.int64)
    groups = np.asarray([r["group"] for r in rows], dtype=str)
    features = build_features(rows)
    quality, quality_ref = make_quality(rows, out_dir)

    np.savez_compressed(
        out_dir / "board_force_band_with_260617_features.npz",
        labels=labels,
        binary=binary,
        reason=reason,
        quality=quality,
        groups=groups,
        episode=np.asarray([r["episode"] for r in rows], dtype=object),
        start=np.asarray([r["start"] for r in rows], dtype=np.int64),
        end=np.asarray([r["end"] for r in rows], dtype=np.int64),
        **features,
    )
    runtime_compatible_features = save_runtime_compatible_features(
        out_dir,
        labels,
        binary,
        reason,
        quality,
        groups,
        rows,
        features,
    )

    models = make_models(args.seed, args.n_jobs, args.include_mlp)
    result: Dict[str, Any] = {
        "purpose": "Episode-level evaluation for board TacQuality scorer after adding 260617 as positive.",
        "inputs": {"datasets": DATASETS},
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
        "n_samples": int(len(rows)),
        "n_groups": int(len(np.unique(groups))),
        "label_counts": {label: int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))},
        "reason_to_id": REASON_TO_ID,
        "quality_reference": quality_ref,
        "runtime_compatible_features": runtime_compatible_features,
        "variant_order": list(features.keys()),
    }
    for variant, x in features.items():
        print(f"[variant] {variant}: n={len(x)} dim={x.shape[1]}", flush=True)
        result[variant] = evaluate_variant(
            variant,
            x.astype(np.float32),
            binary,
            reason,
            quality,
            labels,
            groups,
            models,
            args,
            out_dir,
        )
        print(f"  best={result[variant]['best_model']}", flush=True)

    marker_best = result["marker_action"]["models"][result["marker_action"]["best_model"]]["overall"]
    force_best = result["force_oracle"]["models"][result["force_oracle"]["best_model"]]["overall"]
    result["interpretation"] = [
        f"- Best marker/action deployable AUC is `{fmt(marker_best['binary_auc'])}` with bACC `{fmt(marker_best['binary_balanced_accuracy'])}`.",
        f"- Best marker/action 260617 positive recall is `{fmt(marker_best['positive_260617_recall'])}`.",
        f"- Force-oracle AUC is `{fmt(force_best['binary_auc'])}` with bACC `{fmt(force_best['binary_balanced_accuracy'])}`.",
    ]
    if (marker_best["positive_260617_recall"] or 0.0) >= 0.90:
        result["interpretation"].append("- Adding 260617 as positive makes held-out 260617 windows recognizable as good under marker/action features.")
    else:
        result["interpretation"].append("- Marker/action features still struggle to recognize 260617 as positive; scorer calibration needs more 260617 labels or force-aware targets.")
    if (marker_best["quality_spearman"] or 0.0) >= 0.50:
        result["interpretation"].append("- The deployable score has a useful but still imperfect physical-quality ordering signal.")
    else:
        result["interpretation"].append("- Continuous quality ordering remains weak; do not use this as a strong force-quality optimizer without rollout validation.")

    json_path = out_dir / "board_force_band_with_260617_eval.json"
    md_path = out_dir / "board_force_band_with_260617_eval.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    plot_summary(result, out_dir / "board_force_band_with_260617_summary.png")
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
