"""Scientifically evaluate tactile latent classifiers for PTG scoring.

This script separates two questions:
1. Frame-level shuffled CV: can frames from the same distribution be separated?
2. Episode-level generalization: can the model classify frames from unseen episodes?

The second result is the one that should be trusted for PTG scorer selection.
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/home/chenshuai/Project/TactileACT-cs")
from TFAC_V5.tactile_vae import TactileVAE


DATA_DIR = "/home/chenshuai/data/dataset/0414"
VAE_CKPT = "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt"
OUTPUT_DIR = "/home/chenshuai/Project/output/ptg_quality_eval"
TEMPORAL_WINDOW = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = TactileVAE(latent_dim=16, temporal_window=TEMPORAL_WINDOW)
    ckpt = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)
    return model, ckpt.get("norm_stats")


def normalize_marker(marker_offset, norm_stats):
    mean = np.array(norm_stats["mean"], dtype=np.float32)
    std = np.array(norm_stats["std"], dtype=np.float32)
    return ((marker_offset - mean) / std).astype(np.float32)


def encode_episode(model, episode_path, norm_stats):
    with h5py.File(episode_path, "r") as f:
        marker = f["observations/tac/left/marker_offset"][:]
    marker = normalize_marker(marker, norm_stats)

    latents = []
    with torch.no_grad():
        for t in range(TEMPORAL_WINDOW - 1, marker.shape[0]):
            window = marker[t - TEMPORAL_WINDOW + 1 : t + 1]
            x = torch.from_numpy(window).float().unsqueeze(0).to(DEVICE)
            _, mu_last = model.encode_single_frame(x)
            latents.append(mu_last.flatten(1).cpu().numpy())
    return np.concatenate(latents, axis=0)


def extract_insert_prebounce(latents, info):
    labels = np.asarray(info["labels"])
    offset = TEMPORAL_WINDOW - 1
    ep_type = info.get("type", "")

    if ep_type == "success":
        mask = labels[offset : offset + len(latents)] == 1
        return {"insert": latents[mask[: len(latents)]]}

    if ep_type == "bounce":
        out = []
        for lift_start, _, _ in info.get("lifts", []):
            start = max(0, lift_start - offset - 10)
            end = max(0, lift_start - offset)
            if end > start:
                out.append(latents[start:end])
        if out:
            return {"pre_bounce": np.concatenate(out, axis=0)}
    return {}


def build_dataset(cache_path):
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"], data["groups"], data["group_names"]

    os.makedirs(cache_path.parent, exist_ok=True)
    with open(os.path.join(DATA_DIR, "annotations.pkl"), "rb") as f:
        annotations = pickle.load(f)

    model, norm_stats = load_model()
    X_parts, y_parts, group_parts, group_name_parts = [], [], [], []
    group_id = 0

    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        ep_path = os.path.join(DATA_DIR, f"{ep_name}.hdf5")
        if not os.path.exists(ep_path):
            continue

        latents = encode_episode(model, ep_path, norm_stats)
        items = extract_insert_prebounce(latents, info)
        for label_name, arr in items.items():
            if len(arr) == 0:
                continue
            y_val = 0 if label_name == "insert" else 1
            X_parts.append(arr.astype(np.float32))
            y_parts.append(np.full(len(arr), y_val, dtype=np.int64))
            group_parts.append(np.full(len(arr), group_id, dtype=np.int64))
            group_name_parts.extend([ep_name] * len(arr))
        group_id += 1

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)
    group_names = np.asarray(group_name_parts)
    np.savez_compressed(cache_path, X=X, y=y, groups=groups, group_names=group_names)
    return X, y, groups, group_names


def make_models():
    return {
        "LogReg_C10": LogisticRegression(max_iter=3000, C=10.0, class_weight="balanced"),
        "LDA": LinearDiscriminantAnalysis(),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=24,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "MLP_128_64": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            batch_size=256,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        ),
    }


def balance_training_indices(y, rng):
    cls0 = np.flatnonzero(y == 0)
    cls1 = np.flatnonzero(y == 1)
    n = min(len(cls0), len(cls1))
    keep = np.concatenate([rng.choice(cls0, n, replace=False), rng.choice(cls1, n, replace=False)])
    rng.shuffle(keep)
    return keep


def scores_for_model(clf, X_train, y_train, X_test, y_test):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    row = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_prebounce": float(f1_score(y_test, pred, pos_label=1)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "classification_report": classification_report(
            y_test, pred, target_names=["insert", "pre_bounce"], output_dict=True, zero_division=0
        ),
    }
    if hasattr(pipe, "predict_proba"):
        prob = pipe.predict_proba(X_test)[:, 1]
        row["roc_auc"] = float(roc_auc_score(y_test, prob))
    elif hasattr(pipe, "decision_function"):
        score = pipe.decision_function(X_test)
        row["roc_auc"] = float(roc_auc_score(y_test, score))
    return row


def aggregate(rows):
    keys = ["accuracy", "balanced_accuracy", "f1_prebounce", "roc_auc"]
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    out["folds"] = rows
    return out


def evaluate_frame_cv(X, y, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rng = np.random.default_rng(42)
    results = defaultdict(list)
    for train_idx, test_idx in cv.split(X, y):
        keep = balance_training_indices(y[train_idx], rng)
        train_bal = train_idx[keep]
        for name, model in models.items():
            results[name].append(scores_for_model(model, X[train_bal], y[train_bal], X[test_idx], y[test_idx]))
    return {name: aggregate(rows) for name, rows in results.items()}


def evaluate_group_cv(X, y, groups, models):
    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    cv = GroupKFold(n_splits=n_splits)
    rng = np.random.default_rng(123)
    results = defaultdict(list)
    for train_idx, test_idx in cv.split(X, y, groups):
        keep = balance_training_indices(y[train_idx], rng)
        train_bal = train_idx[keep]
        for name, model in models.items():
            results[name].append(scores_for_model(model, X[train_bal], y[train_bal], X[test_idx], y[test_idx]))
    return {name: aggregate(rows) for name, rows in results.items()}


def evaluate_holdout(X, y, groups, models):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    rng = np.random.default_rng(2026)
    keep = balance_training_indices(y[train_idx], rng)
    train_bal = train_idx[keep]
    return {
        name: scores_for_model(model, X[train_bal], y[train_bal], X[test_idx], y[test_idx])
        for name, model in models.items()
    }


def rank_by(results, key="balanced_accuracy"):
    rows = []
    for name, res in results.items():
        if key in res:
            rows.append((name, res[key]["mean"], res[key]["std"]))
    return sorted(rows, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.path.join(OUTPUT_DIR, "insert_prebounce_episode_cache.npz"))
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR, "tactile_quality_model_eval.json"))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    X, y, groups, group_names = build_dataset(args.cache)
    models = make_models()

    frame_cv = evaluate_frame_cv(X, y, models)
    group_cv = evaluate_group_cv(X, y, groups, models)
    holdout = evaluate_holdout(X, y, groups, models)

    result = {
        "task": "insert(0) vs pre_bounce(1)",
        "data": {
            "X_shape": list(X.shape),
            "n_insert": int((y == 0).sum()),
            "n_pre_bounce": int((y == 1).sum()),
            "n_groups": int(len(np.unique(groups))),
            "n_group_names": int(len(set(group_names.tolist()))),
        },
        "protocols": {
            "frame_cv": "5-fold StratifiedKFold over frames; optimistic because neighboring frames from same episode can cross folds.",
            "group_cv": "5-fold GroupKFold by episode; primary generalization estimate.",
            "group_holdout": "20% unseen episodes; final sanity check.",
            "training_balance": "Each train fold is class-balanced by undersampling insert frames; test folds keep natural class proportions.",
        },
        "frame_cv": frame_cv,
        "group_cv": group_cv,
        "group_holdout": holdout,
        "group_cv_ranking_by_balanced_accuracy": rank_by(group_cv, "balanced_accuracy"),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["data"], ensure_ascii=False, indent=2))
    print("\nGroup-CV ranking by balanced accuracy:")
    for i, (name, mean, std) in enumerate(result["group_cv_ranking_by_balanced_accuracy"], start=1):
        print(f"{i:2d}. {name:14s} {mean:.4f} +/- {std:.4f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
