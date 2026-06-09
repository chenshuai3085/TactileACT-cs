"""Very fast probe for unified quality taxonomy selection."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("/home/chenshuai/Project/output/unified_quality_taxonomy")
CACHE = OUT_DIR / "unified_quality_features.npz"
OUT_JSON = OUT_DIR / "quick_taxonomy_probe.json"


def sample_balanced(X, y, task, groups, quality, max_per_task_class=800):
    rng = np.random.default_rng(42)
    keep = []
    for task_name in np.unique(task):
        for cls in np.unique(y):
            if cls < 0:
                continue
            idx = np.flatnonzero((task == task_name) & (y == cls))
            if len(idx) == 0:
                continue
            keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    keep = np.array(sorted(keep), dtype=np.int64)
    return X[keep], y[keep], task[keep], groups[keep], quality[keep]


def fit_eval(Xtr, ytr, Xte, yte, qte, taxonomy):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    max_iter=2000,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    row = {
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
    }
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(Xte)
        classes = clf.named_steps["clf"].classes_
        score = quality_score_from_proba(proba, classes, taxonomy)
        if taxonomy in {"t4", "t3"}:
            mask = (yte == 1) | (yte >= 2)
            y_gb = (yte[mask] == 1).astype(np.int64)
            if len(np.unique(y_gb)) == 2:
                row["good_vs_bad_auc"] = float(roc_auc_score(y_gb, score[mask]))
        elif taxonomy == "binary" and len(np.unique(yte)) == 2:
            cls_to_col = {int(c): i for i, c in enumerate(classes)}
            row["good_vs_bad_auc"] = float(roc_auc_score(yte, proba[:, cls_to_col[1]]))
        row["score_quality_corr"] = corr(score, qte)
    return row


def quality_score_from_proba(proba, classes, taxonomy):
    weights = {
        "t4": {0: -0.25, 1: 1.0, 2: -0.85, 3: -1.0},
        "t3": {0: -0.25, 1: 1.0, 2: -1.0},
        "binary": {0: -1.0, 1: 1.0},
    }[taxonomy]
    cls_to_col = {int(c): i for i, c in enumerate(classes)}
    score = np.zeros(len(proba), dtype=np.float64)
    for cls, weight in weights.items():
        if cls in cls_to_col:
            score += weight * proba[:, cls_to_col[cls]]
    return score


def corr(a, b):
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def aggregate(rows):
    rows = [r for r in rows if r]
    out = {}
    for key in ["balanced_accuracy", "macro_f1", "good_vs_bad_auc", "score_quality_corr"]:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def group_cv(X, y, groups, quality, taxonomy):
    rows = []
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for tr, te in cv.split(X, y, groups):
        rows.append(fit_eval(X[tr], y[tr], X[te], y[te], quality[te], taxonomy))
    return aggregate(rows)


def cross_task(X, y, task, quality, taxonomy, train_task, test_task):
    tr = task == train_task
    te = task == test_task
    return fit_eval(X[tr], y[tr], X[te], y[te], quality[te], taxonomy)


def main():
    d = np.load(CACHE, allow_pickle=True)
    X_all = d["X"]
    task_all = d["task"]
    groups_all = d["groups"]
    quality_all = d["quality"]
    y_map = {"t4": d["y_t4"], "t3": d["y_t3"], "binary": d["y_binary"]}

    result = {"taxonomies": {}, "best": []}
    for taxonomy, y_all in y_map.items():
        valid = y_all >= 0
        X, y, task, groups, quality = sample_balanced(
            X_all[valid], y_all[valid], task_all[valid], groups_all[valid], quality_all[valid]
        )
        block = {
            "counts": {
                "all": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
                "insertion": {str(k): int(v) for k, v in Counter(y[task == "insertion"].tolist()).items()},
                "board": {str(k): int(v) for k, v in Counter(y[task == "board"].tolist()).items()},
            },
            "mixed_group_cv": group_cv(X, y, groups, quality, taxonomy),
            "insertion_group_cv": group_cv(X[task == "insertion"], y[task == "insertion"], groups[task == "insertion"], quality[task == "insertion"], taxonomy),
            "board_group_cv": group_cv(X[task == "board"], y[task == "board"], groups[task == "board"], quality[task == "board"], taxonomy),
            "cross_insertion_to_board": cross_task(X, y, task, quality, taxonomy, "insertion", "board"),
            "cross_board_to_insertion": cross_task(X, y, task, quality, taxonomy, "board", "insertion"),
        }
        result["taxonomies"][taxonomy] = block
        mixed = block["mixed_group_cv"]
        cross_f1 = np.mean(
            [
                (block["cross_insertion_to_board"] or {}).get("macro_f1", 0.0),
                (block["cross_board_to_insertion"] or {}).get("macro_f1", 0.0),
            ]
        )
        objective = (
            0.35 * (mixed.get("macro_f1") or {}).get("mean", 0.0)
            + 0.25 * (mixed.get("balanced_accuracy") or {}).get("mean", 0.0)
            + 0.25 * (mixed.get("good_vs_bad_auc") or {}).get("mean", 0.0)
            + 0.15 * cross_f1
        )
        result["best"].append(
            {
                "taxonomy": taxonomy,
                "objective": float(objective),
                "mixed_macro_f1": (mixed.get("macro_f1") or {}).get("mean", 0.0),
                "mixed_balanced_accuracy": (mixed.get("balanced_accuracy") or {}).get("mean", 0.0),
                "mixed_good_vs_bad_auc": (mixed.get("good_vs_bad_auc") or {}).get("mean", 0.0),
                "cross_macro_f1": float(cross_f1),
            }
        )
    result["best"].sort(key=lambda r: r["objective"], reverse=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result["best"], ensure_ascii=False, indent=2))
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
