"""Audit frame-level split leakage for TacQuality classification evidence.

High frame-random accuracy is not enough for a DP guidance scorer: adjacent
frames from the same episode can be nearly identical, so random splits may
overestimate generalization.  This audit compares frame-level StratifiedShuffle
Split against episode-level GroupKFold on the same cached features.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_split_leakage_audit")
CACHES = {
    "unified_quality_taxonomy": Path("/home/chenshuai/Project/output/unified_quality_taxonomy/unified_quality_features.npz"),
    "ptg_proxy_scorer_v2": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz"),
}


def rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-8 or np.std(rb) < 1e-8:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def balance_train_indices(y: np.ndarray, seed: int, max_per_class: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx: List[int] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) == 0:
            continue
        n = len(cls_idx) if max_per_class is None else min(max_per_class, len(cls_idx))
        idx.extend(rng.choice(cls_idx, n, replace=False).tolist())
    idx_arr = np.array(idx, dtype=np.int64)
    rng.shuffle(idx_arr)
    return idx_arr


def subsample_by_task_class(
    y: np.ndarray,
    task: np.ndarray,
    max_per_task_class: int,
    seed: int,
) -> np.ndarray:
    if max_per_task_class <= 0:
        return np.arange(len(y), dtype=np.int64)
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for task_name in np.unique(task):
        for cls in np.unique(y):
            idx = np.flatnonzero((task == task_name) & (y == cls))
            if len(idx) == 0:
                continue
            keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


def make_model(name: str, seed: int):
    if name == "logreg":
        return LogisticRegression(max_iter=3000, C=3.0, class_weight="balanced", random_state=seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"unknown model: {name}")


def score_from_proba(proba: np.ndarray, classes: Iterable[int], label_name: str) -> np.ndarray | None:
    classes = [int(c) for c in classes]
    class_to_col = {c: i for i, c in enumerate(classes)}
    if label_name in {"binary", "y_binary"}:
        if 1 not in class_to_col:
            return None
        return proba[:, class_to_col[1]]
    if label_name in {"reason", "y_t4"}:
        weights = {0: -0.25, 1: 1.0, 2: -0.85, 3: -1.0, 4: -0.55}
    elif label_name == "y_t3":
        weights = {0: -0.25, 1: 1.0, 2: -1.0}
    else:
        return None
    score = np.zeros(len(proba), dtype=np.float64)
    for cls, weight in weights.items():
        if cls in class_to_col:
            score += weight * proba[:, class_to_col[cls]]
    return score


def eval_one_split(
    X: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    model_name: str,
    label_name: str,
    seed: int,
    max_train_per_class: int,
) -> Dict[str, Any] | None:
    y_train = y[train_idx]
    y_test = y[test_idx]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    local_train = balance_train_indices(y_train, seed, max_train_per_class)
    train_balanced = train_idx[local_train]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", make_model(model_name, seed))])
    pipe.fit(X[train_balanced], y[train_balanced])
    pred = pipe.predict(X[test_idx])
    row: Dict[str, Any] = {
        "n_train": int(len(train_idx)),
        "n_train_balanced": int(len(train_balanced)),
        "n_test": int(len(test_idx)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X[test_idx])
        score = score_from_proba(proba, pipe.named_steps["clf"].classes_, label_name)
        if score is not None:
            corr = rank_corr(score, quality[test_idx])
            if corr is not None:
                row["score_quality_spearman"] = corr
            if label_name in {"binary", "y_binary"} and len(np.unique(y_test)) == 2:
                row["good_vs_bad_auc"] = float(roc_auc_score(y_test, score))
            elif 1 in np.unique(y_test) and np.any(y_test >= 2):
                mask = (y_test == 1) | (y_test >= 2)
                y_good = (y_test[mask] == 1).astype(np.int64)
                if len(np.unique(y_good)) == 2:
                    row["good_vs_bad_auc"] = float(roc_auc_score(y_good, score[mask]))
    return row


def aggregate(rows: List[Dict[str, Any] | None]) -> Dict[str, Any]:
    valid = [r for r in rows if r is not None]
    out: Dict[str, Any] = {"n_splits": len(valid)}
    for key in ["balanced_accuracy", "macro_f1", "good_vs_bad_auc", "score_quality_spearman"]:
        vals = [r[key] for r in valid if key in r and np.isfinite(r[key])]
        if vals:
            out[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return out


def random_cv(
    X: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
    *,
    model_name: str,
    label_name: str,
    n_splits: int,
    seed: int,
    max_train_per_class: int,
) -> Dict[str, Any]:
    splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=seed)
    rows = []
    for split_id, (tr, te) in enumerate(splitter.split(X, y)):
        rows.append(
            eval_one_split(
                X,
                y,
                quality,
                tr,
                te,
                model_name=model_name,
                label_name=label_name,
                seed=seed + split_id,
                max_train_per_class=max_train_per_class,
            )
        )
    return aggregate(rows)


def group_cv(
    X: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    label_name: str,
    n_splits: int,
    seed: int,
    max_train_per_class: int,
) -> Dict[str, Any]:
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    rows = []
    for split_id, (tr, te) in enumerate(splitter.split(X, y, groups)):
        rows.append(
            eval_one_split(
                X,
                y,
                quality,
                tr,
                te,
                model_name=model_name,
                label_name=label_name,
                seed=seed + split_id,
                max_train_per_class=max_train_per_class,
            )
        )
    return aggregate(rows)


def load_cache(name: str, path: Path) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if name == "unified_quality_taxonomy":
        labels = {
            "y_t4": data["y_t4"].astype(np.int64),
            "y_t3": data["y_t3"].astype(np.int64),
            "y_binary": data["y_binary"].astype(np.int64),
        }
    else:
        labels = {
            "reason": data["reason"].astype(np.int64),
            "binary": data["binary"].astype(np.int64),
        }
    return {
        "X": data["X"].astype(np.float32),
        "quality": data["quality"].astype(np.float32),
        "task": data["task"],
        "groups": data["groups"],
        "labels": labels,
    }


def leakage_delta(random_result: Dict[str, Any], group_result: Dict[str, Any], key: str) -> float | None:
    if key not in random_result or key not in group_result:
        return None
    return float(random_result[key]["mean"] - group_result[key]["mean"])


def audit_cache(name: str, path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    payload = load_cache(name, path)
    X = payload["X"]
    quality = payload["quality"]
    task = payload["task"]
    groups = payload["groups"]
    labels = payload["labels"]
    result: Dict[str, Any] = {
        "cache_path": str(path),
        "n_raw": int(len(X)),
        "feature_dim": int(X.shape[1]),
        "task_counts_raw": {str(k): int(v) for k, v in Counter(task.tolist()).items()},
        "labels": {},
    }
    for label_name, y_full in labels.items():
        valid = y_full >= 0
        if valid.sum() < 10 or len(np.unique(y_full[valid])) < 2:
            continue
        keep_local = subsample_by_task_class(y_full[valid], task[valid], args.max_per_task_class, args.seed)
        X_use = X[valid][keep_local]
        y = y_full[valid][keep_local]
        q = quality[valid][keep_local]
        task_use = task[valid][keep_local]
        groups_use = groups[valid][keep_local]
        block: Dict[str, Any] = {
            "n": int(len(y)),
            "n_groups": int(len(np.unique(groups_use))),
            "class_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
            "task_counts": {str(k): int(v) for k, v in Counter(task_use.tolist()).items()},
            "models": {},
        }
        for model_name in args.models:
            frame_random = random_cv(
                X_use,
                y,
                q,
                model_name=model_name,
                label_name=label_name,
                n_splits=args.splits,
                seed=args.seed,
                max_train_per_class=args.max_train_per_class,
            )
            episode_group = group_cv(
                X_use,
                y,
                q,
                groups_use,
                model_name=model_name,
                label_name=label_name,
                n_splits=args.splits,
                seed=args.seed,
                max_train_per_class=args.max_train_per_class,
            )
            deltas = {
                key: leakage_delta(frame_random, episode_group, key)
                for key in ["balanced_accuracy", "macro_f1", "good_vs_bad_auc", "score_quality_spearman"]
            }
            block["models"][model_name] = {
                "frame_random_split": frame_random,
                "episode_group_kfold": episode_group,
                "frame_minus_group_delta": {k: v for k, v in deltas.items() if v is not None},
            }
        result["labels"][label_name] = block
    return result


def best_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for cache_name, cache_result in results["caches"].items():
        for label_name, label_result in cache_result["labels"].items():
            for model_name, model_result in label_result["models"].items():
                group = model_result["episode_group_kfold"]
                row = {
                    "cache": cache_name,
                    "label": label_name,
                    "model": model_name,
                    "group_balanced_accuracy": group.get("balanced_accuracy", {}).get("mean"),
                    "group_macro_f1": group.get("macro_f1", {}).get("mean"),
                    "group_good_vs_bad_auc": group.get("good_vs_bad_auc", {}).get("mean"),
                    "group_score_quality_spearman": group.get("score_quality_spearman", {}).get("mean"),
                    "random_minus_group_bal_acc": model_result["frame_minus_group_delta"].get("balanced_accuracy"),
                    "random_minus_group_macro_f1": model_result["frame_minus_group_delta"].get("macro_f1"),
                }
                objective = 0.0
                for key, weight in [
                    ("group_good_vs_bad_auc", 0.35),
                    ("group_macro_f1", 0.25),
                    ("group_balanced_accuracy", 0.25),
                    ("group_score_quality_spearman", 0.15),
                ]:
                    val = row[key]
                    objective += weight * (0.0 if val is None else max(float(val), 0.0))
                leak_penalty = abs(row["random_minus_group_bal_acc"] or 0.0)
                row["guidance_evidence_score"] = float(objective - 0.15 * leak_penalty)
                rows.append(row)
    rows.sort(key=lambda r: r["guidance_evidence_score"], reverse=True)
    return rows


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Split Leakage Audit",
        "",
        "This audit compares frame-level random split against episode-level GroupKFold.",
        "For scientific claims and DP guidance design, episode-level GroupKFold is the authoritative offline split.",
        "",
        "## Best Candidates",
        "",
        "| rank | cache | label | model | group bal acc | group macro F1 | group AUC | quality spearman | random-group bal acc |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(result["best_candidates"][:12], 1):
        lines.append(
            "| {rank} | {cache} | {label} | {model} | {gba} | {gmf} | {auc} | {sp} | {delta} |".format(
                rank=i,
                cache=row["cache"],
                label=row["label"],
                model=row["model"],
                gba=fmt(row["group_balanced_accuracy"]),
                gmf=fmt(row["group_macro_f1"]),
                auc=fmt(row["group_good_vs_bad_auc"]),
                sp=fmt(row["group_score_quality_spearman"]),
                delta=fmt(row["random_minus_group_bal_acc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A large positive random-group delta means frame-random split overestimates generalization.",
            "- A small delta with high GroupKFold metrics is stronger evidence that the scorer generalizes to unseen episodes.",
            "- This is still offline evidence. Production completion still requires baseline-vs-guided rollout gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(x: Any) -> str:
    if x is None:
        return "-"
    return f"{float(x):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_per_task_class", type=int, default=1200)
    parser.add_argument("--max_train_per_class", type=int, default=2500)
    parser.add_argument("--models", nargs="+", default=["logreg"], choices=["logreg", "rf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "purpose": "Detect frame-level split leakage risk in TacQuality classifier/scorer evaluation.",
        "authoritative_split": "episode_group_kfold",
        "args": vars(args),
        "caches": {},
    }
    for name, path in CACHES.items():
        if path.exists():
            print(f"auditing {name}: {path}", flush=True)
            result["caches"][name] = audit_cache(name, path, args)
    result["best_candidates"] = best_rows(result)
    json_path = out_dir / "tac_quality_split_leakage_audit.json"
    md_path = out_dir / "tac_quality_split_leakage_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "best": result["best_candidates"][:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
