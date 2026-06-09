"""Weakly label and evaluate board-wiping tactile/action quality.

There are no human quality labels in 260522_v8l_caheiban. This script therefore
does not report supervised ground-truth accuracy. It builds interpretable
pseudo-labels from two criteria requested by the user:
1. force magnitude should be neither too small nor too large;
2. force/action/tactile changes should be smooth.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = "/home/chenshuai/data/dataset/260522_v8l_caheiban"
OUTPUT_DIR = "/home/chenshuai/Project/output/caheiban_quality_eval"


def robust_norm(x):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / (1.4826 * mad)


def safe_diff_norm(x):
    if len(x) < 2:
        return np.zeros(1, dtype=np.float32)
    return np.linalg.norm(np.diff(x, axis=0), axis=1)


def window_features(path, window=32, stride=16):
    rows = []
    with h5py.File(path, "r") as f:
        left_f = f["observations/tac/left/force6d"][:]
        right_f = f["observations/tac/right/force6d"][:]
        left_m = f["observations/tac/left/marker_offset"][:]
        right_m = f["observations/tac/right/marker_offset"][:]
        action = f["actions/eef_abs"][:]
        ft = f["ft"][:] if "ft" in f else left_f + right_f

    n = min(len(left_f), len(right_f), len(left_m), len(action), len(ft))
    ep = Path(path).stem
    for start in range(0, max(1, n - window + 1), stride):
        end = min(n, start + window)
        if end - start < max(8, window // 2):
            continue

        lf = left_f[start:end]
        rf = right_f[start:end]
        ff = ft[start:end]
        lm = left_m[start:end]
        rm = right_m[start:end]
        aa = action[start:end]

        # Use translational force magnitude. This avoids torque scale mixing.
        force_mag = np.linalg.norm(ff[:, :3], axis=1)
        left_force_mag = np.linalg.norm(lf[:, :3], axis=1)
        right_force_mag = np.linalg.norm(rf[:, :3], axis=1)
        marker_mag = 0.5 * (
            np.linalg.norm(lm.reshape(len(lm), -1), axis=1)
            + np.linalg.norm(rm.reshape(len(rm), -1), axis=1)
        )

        force_delta = safe_diff_norm(ff[:, :3])
        action_delta = safe_diff_norm(aa)
        marker_delta = safe_diff_norm(marker_mag[:, None])

        rows.append(
            {
                "episode": ep,
                "start": int(start),
                "end": int(end),
                "force_mean": float(force_mag.mean()),
                "force_std": float(force_mag.std()),
                "force_p95": float(np.percentile(force_mag, 95)),
                "left_force_mean": float(left_force_mag.mean()),
                "right_force_mean": float(right_force_mag.mean()),
                "marker_mean": float(marker_mag.mean()),
                "marker_std": float(marker_mag.std()),
                "force_delta_mean": float(force_delta.mean()),
                "force_delta_p95": float(np.percentile(force_delta, 95)),
                "action_delta_mean": float(action_delta.mean()),
                "marker_delta_mean": float(marker_delta.mean()),
            }
        )
    return rows


def assign_pseudo_labels(rows):
    force = np.array([r["force_mean"] for r in rows])
    force_p95 = np.array([r["force_p95"] for r in rows])
    smooth = np.array([r["force_delta_mean"] for r in rows])
    action_smooth = np.array([r["action_delta_mean"] for r in rows])
    marker_smooth = np.array([r["marker_delta_mean"] for r in rows])
    force_z = robust_norm(force)
    smooth_z = robust_norm(smooth)
    action_z = robust_norm(action_smooth)
    marker_z = robust_norm(marker_smooth)

    # Dataset-relative contact quality. These thresholds are inspectable and
    # should be replaced by human labels once several episodes are annotated.
    low_force = force < np.quantile(force, 0.20)
    high_force = (force > np.quantile(force, 0.85)) | (force_p95 > np.quantile(force_p95, 0.90))
    rough = (
        robust_norm(smooth) > 1.0
    ) | (robust_norm(action_smooth) > 1.0) | (robust_norm(marker_smooth) > 1.0)

    labels = []
    scores = []
    for lf, hf, rg, fm, sm, am, mm in zip(
        low_force, high_force, rough, force, smooth, action_smooth, marker_smooth
    ):
        if lf:
            labels.append(0)  # too light / insufficient contact
        elif hf:
            labels.append(1)  # too heavy
        elif rg:
            labels.append(2)  # not smooth
        else:
            labels.append(3)  # good

        idx = len(scores)
        force_center = abs(force_z[idx])
        roughness = max(abs(smooth_z[idx]), abs(action_z[idx]), abs(marker_z[idx]))
        scores.append(float(-force_center - 0.5 * roughness))

    return np.array(labels, dtype=np.int64), np.array(scores, dtype=np.float32)


def rows_to_matrix(rows):
    feature_names = [
        "force_mean",
        "force_std",
        "force_p95",
        "left_force_mean",
        "right_force_mean",
        "marker_mean",
        "marker_std",
        "force_delta_mean",
        "force_delta_p95",
        "action_delta_mean",
        "marker_delta_mean",
    ]
    X = np.array([[r[k] for k in feature_names] for r in rows], dtype=np.float32)
    groups = np.array([r["episode"] for r in rows])
    return X, groups, feature_names


def evaluate_pseudo_label_models(X, y, groups):
    models = {
        "MLP_64_32": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    out = {}
    for name, clf in models.items():
        fold_scores = []
        fold_reports = []
        for train_idx, test_idx in cv.split(X, y, groups):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X[train_idx], y[train_idx])
            pred = pipe.predict(X[test_idx])
            fold_scores.append(float(balanced_accuracy_score(y[test_idx], pred)))
            fold_reports.append(
                {
                    "confusion_matrix": confusion_matrix(y[test_idx], pred).tolist(),
                    "report": classification_report(y[test_idx], pred, output_dict=True, zero_division=0),
                }
            )
        out[name] = {
            "balanced_accuracy_mean": float(np.mean(fold_scores)),
            "balanced_accuracy_std": float(np.std(fold_scores)),
            "folds": fold_reports,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--out-dir", default=OUTPUT_DIR)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(Path(args.data_dir).glob("success/*.hdf5"))
    rows = []
    for p in files:
        rows.extend(window_features(str(p), window=args.window, stride=args.stride))

    labels, quality_scores = assign_pseudo_labels(rows)
    for r, label, score in zip(rows, labels, quality_scores):
        r["pseudo_label"] = int(label)
        r["pseudo_label_name"] = ["too_light", "too_heavy", "rough", "good"][int(label)]
        r["quality_score"] = float(score)

    X, groups, feature_names = rows_to_matrix(rows)
    eval_result = evaluate_pseudo_label_models(X, labels, groups)

    csv_path = os.path.join(args.out_dir, "caheiban_window_quality_labels.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts = {name: int((labels == i).sum()) for i, name in enumerate(["too_light", "too_heavy", "rough", "good"])}
    result = {
        "data_dir": args.data_dir,
        "n_files": len(files),
        "n_windows": len(rows),
        "window": args.window,
        "stride": args.stride,
        "feature_names": feature_names,
        "pseudo_label_definition": {
            "too_light": "force_mean below dataset 20th percentile",
            "too_heavy": "force_mean above 85th percentile or force_p95 above 90th percentile",
            "rough": "not too_light/heavy, but force/action/marker change robust z-score > 1.0",
            "good": "force in acceptable dataset-relative band and changes are smooth",
        },
        "pseudo_label_counts": counts,
        "model_eval_against_pseudo_labels": eval_result,
        "csv": csv_path,
    }
    json_path = os.path.join(args.out_dir, "caheiban_quality_eval.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: result[k] for k in ["n_files", "n_windows", "pseudo_label_counts"]}, indent=2))
    print("Model eval against pseudo-labels:")
    for name, item in eval_result.items():
        print(f"  {name}: {item['balanced_accuracy_mean']:.4f} +/- {item['balanced_accuracy_std']:.4f}")
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
