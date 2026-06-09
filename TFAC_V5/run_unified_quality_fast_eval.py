"""Fast evaluator for cached unified quality taxonomy data."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("/home/chenshuai/Project/output/unified_quality_taxonomy")
CACHE = OUT_DIR / "unified_quality_features.npz"
OUT_JSON = OUT_DIR / "unified_quality_eval_fast.json"


def balance_indices(y, seed=42):
    rng = np.random.default_rng(seed)
    labels = [c for c in np.unique(y) if c >= 0]
    n = min(int((y == c).sum()) for c in labels)
    idx = []
    for c in labels:
        idx.extend(rng.choice(np.flatnonzero(y == c), n, replace=False).tolist())
    idx = np.array(idx, dtype=np.int64)
    rng.shuffle(idx)
    return idx


def spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-8 or np.std(rb) < 1e-8:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def score_from_proba(proba, classes, taxonomy):
    weights = {
        "t4": {0: -0.25, 1: 1.0, 2: -0.85, 3: -1.0},
        "t3": {0: -0.25, 1: 1.0, 2: -1.0},
        "binary": {0: -1.0, 1: 1.0},
    }[taxonomy]
    class_to_col = {int(c): i for i, c in enumerate(classes)}
    score = np.zeros(len(proba), dtype=np.float64)
    for cls, weight in weights.items():
        if cls in class_to_col:
            score += weight * proba[:, class_to_col[cls]]
    return score


def eval_split(model, X_train, y_train, X_test, y_test, q_test, taxonomy):
    train_keep = y_train >= 0
    test_keep = y_test >= 0
    X_train, y_train = X_train[train_keep], y_train[train_keep]
    X_test, y_test, q_test = X_test[test_keep], y_test[test_keep], q_test[test_keep]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    train_idx = balance_indices(y_train)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    pipe.fit(X_train[train_idx], y_train[train_idx])
    pred = pipe.predict(X_test)
    row = {
        "n_train_balanced": int(len(train_idx)),
        "n_test": int(len(y_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)
        score = score_from_proba(proba, pipe.named_steps["clf"].classes_, taxonomy)
        sp = spearman(score, q_test)
        if sp is not None:
            row["score_quality_spearman"] = sp
        if taxonomy in {"t4", "t3"}:
            mask = (y_test == 1) | (y_test >= 2)
            y_gb = (y_test[mask] == 1).astype(np.int64)
            if len(np.unique(y_gb)) == 2:
                row["good_vs_bad_auc"] = float(roc_auc_score(y_gb, score[mask]))
        elif taxonomy == "binary" and len(np.unique(y_test)) == 2:
            cls_to_col = {int(c): i for i, c in enumerate(pipe.named_steps["clf"].classes_)}
            if 1 in cls_to_col:
                row["good_vs_bad_auc"] = float(roc_auc_score(y_test, proba[:, cls_to_col[1]]))
    return row


def aggregate(rows):
    out = {}
    for key in ["balanced_accuracy", "macro_f1", "good_vs_bad_auc", "score_quality_spearman"]:
        vals = [r[key] for r in rows if r is not None and key in r and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def group_cv(X, y, quality, groups, models, taxonomy):
    out = {}
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for name, model in models.items():
        rows = []
        for train_idx, test_idx in cv.split(X, y, groups):
            rows.append(
                eval_split(model, X[train_idx], y[train_idx], X[test_idx], y[test_idx], quality[test_idx], taxonomy)
            )
        out[name] = aggregate(rows)
        print(f"group_cv {taxonomy} {name}: {out[name]}", flush=True)
    return out


def cross_task(X, y, quality, task, models, taxonomy, train_task, test_task):
    out = {}
    train_idx = np.flatnonzero(task == train_task)
    test_idx = np.flatnonzero(task == test_task)
    for name, model in models.items():
        row = eval_split(model, X[train_idx], y[train_idx], X[test_idx], y[test_idx], quality[test_idx], taxonomy)
        out[name] = row
        print(f"cross {taxonomy} {train_task}->{test_task} {name}: {row}", flush=True)
    return out


def objective(row):
    if not row:
        return 0.0
    return (
        0.30 * row.get("mixed_macro_f1", 0.0)
        + 0.25 * row.get("mixed_balanced_accuracy", 0.0)
        + 0.25 * row.get("mixed_good_vs_bad_auc", 0.0)
        + 0.10 * max(row.get("mixed_score_spearman", 0.0), 0.0)
        + 0.10 * row.get("cross_macro_f1", 0.0)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-rf", action="store_true")
    parser.add_argument("--max-per-task-class", type=int, default=1200)
    args = parser.parse_args()

    data = np.load(CACHE, allow_pickle=True)
    X = data["X"]
    y_by_tax = {
        "t4": data["y_t4"],
        "t3": data["y_t3"],
        "binary": data["y_binary"],
    }
    quality = data["quality"]
    task = data["task"]
    groups = data["groups"]
    if args.max_per_task_class and args.max_per_task_class > 0:
        base_y = data["y_t4"]
        rng = np.random.default_rng(42)
        keep = []
        for task_name in np.unique(task):
            for cls in np.unique(base_y):
                idx = np.flatnonzero((task == task_name) & (base_y == cls))
                if len(idx) == 0:
                    continue
                n = min(args.max_per_task_class, len(idx))
                keep.extend(rng.choice(idx, n, replace=False).tolist())
        keep = np.array(sorted(keep), dtype=np.int64)
        X = X[keep]
        y_by_tax = {name: y[keep] for name, y in y_by_tax.items()}
        quality = quality[keep]
        task = task[keep]
        groups = groups[keep]
    models = {
        "LogReg": LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced"),
    }
    if args.include_rf:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    results = {
        "data": {
            "n_samples": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "tasks": {k: int(v) for k, v in Counter(task.tolist()).items()},
        },
        "eval": {},
        "best_candidates": [],
    }

    for taxonomy, y in y_by_tax.items():
        valid = y >= 0
        block = {
            "counts": {
                "all": {str(k): int(v) for k, v in Counter(y[valid].tolist()).items()},
                "insertion": {str(k): int(v) for k, v in Counter(y[(task == "insertion") & valid].tolist()).items()},
                "board": {str(k): int(v) for k, v in Counter(y[(task == "board") & valid].tolist()).items()},
            },
            "mixed_group_cv": group_cv(X[valid], y[valid], quality[valid], groups[valid], models, taxonomy),
            "in_task_group_cv": {},
            "cross_task": {},
        }
        for task_name in ["insertion", "board"]:
            mask = (task == task_name) & valid
            block["in_task_group_cv"][task_name] = group_cv(
                X[mask], y[mask], quality[mask], groups[mask], models, taxonomy
            )
        block["cross_task"]["insertion_to_board"] = cross_task(
            X[valid], y[valid], quality[valid], task[valid], models, taxonomy, "insertion", "board"
        )
        block["cross_task"]["board_to_insertion"] = cross_task(
            X[valid], y[valid], quality[valid], task[valid], models, taxonomy, "board", "insertion"
        )
        results["eval"][taxonomy] = block

        for model_name in models:
            mixed = block["mixed_group_cv"].get(model_name, {})
            cross_vals = []
            for direction in ["insertion_to_board", "board_to_insertion"]:
                row = block["cross_task"][direction].get(model_name)
                if row:
                    cross_vals.append(row.get("macro_f1", 0.0))
            cand = {
                "taxonomy": taxonomy,
                "model": model_name,
                "mixed_balanced_accuracy": (mixed.get("balanced_accuracy") or {}).get("mean", 0.0),
                "mixed_macro_f1": (mixed.get("macro_f1") or {}).get("mean", 0.0),
                "mixed_good_vs_bad_auc": (mixed.get("good_vs_bad_auc") or {}).get("mean", 0.0),
                "mixed_score_spearman": (mixed.get("score_quality_spearman") or {}).get("mean", 0.0),
                "cross_macro_f1": float(np.mean(cross_vals)) if cross_vals else 0.0,
            }
            cand["objective"] = objective(cand)
            results["best_candidates"].append(cand)

    results["best_candidates"].sort(key=lambda r: r["objective"], reverse=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"saved {OUT_JSON}", flush=True)
    print(json.dumps(results["best_candidates"][:8], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
