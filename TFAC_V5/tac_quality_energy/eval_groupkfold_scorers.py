#!/usr/bin/env python3
"""Episode-level GroupKFold evaluation for TacQuality scorer features.

This script answers a stricter question than frame/window random CV:

    Does a tactile-quality classifier/scorer generalize to held-out episodes?

It uses saved feature caches that already contain ``groups``.  No GPU is
required and no DP checkpoint is touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_groupkfold_eval")
DEFAULT_INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
DEFAULT_PTG_FEATURES = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_simple(x: np.ndarray, y: np.ndarray) -> float:
    return finite_corr(rankdata_simple(np.asarray(x)), rankdata_simple(np.asarray(y)))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, score))


def best_balanced_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    # Use F1 on train folds only to pick a classification threshold.
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def decile_summary(score: np.ndarray, quality: np.ndarray, binary: np.ndarray, n_bins: int) -> Dict[str, Any]:
    order = np.argsort(score, kind="mergesort")
    rows: List[Dict[str, Any]] = []
    for i, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": int(i),
                "n": int(len(idx)),
                "score_mean": float(np.mean(score[idx])),
                "quality_mean": float(np.mean(quality[idx])),
                "good_rate": float(np.mean(binary[idx][binary[idx] >= 0] == 1)) if np.any(binary[idx] >= 0) else None,
            }
        )
    q = np.asarray([r["quality_mean"] for r in rows], dtype=np.float64)
    g = np.asarray([np.nan if r["good_rate"] is None else r["good_rate"] for r in rows], dtype=np.float64)
    g_valid = g[np.isfinite(g)]
    return {
        "bins": rows,
        "quality_positive_step_rate": float(np.mean(np.diff(q) >= -1e-8)) if len(q) > 1 else None,
        "good_rate_positive_step_rate": float(np.mean(np.diff(g_valid) >= -1e-8)) if len(g_valid) > 1 else None,
        "top_bottom_quality_gap": float(q[-1] - q[0]) if len(q) else None,
        "top_bottom_good_rate_gap": float(g_valid[-1] - g_valid[0]) if len(g_valid) else None,
    }


def nanmean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def nanstd(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return 0.0 if vals else None
    return float(np.std(vals, ddof=1))


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def make_models(seed: int, n_jobs: int, include_mlp: bool) -> Dict[str, Any]:
    models: Dict[str, Any] = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2500,
                        C=1.0,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "rf": RandomForestClassifier(
            n_estimators=220,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.06,
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
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=350,
                        early_stopping=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return models


def predict_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return proba[:, 0]
        return proba[:, -1]
    score = model.decision_function(x)
    score = np.asarray(score)
    if score.ndim > 1:
        score = score[:, -1]
    return score.astype(np.float64)


@dataclass
class Section:
    name: str
    x: np.ndarray
    binary: np.ndarray
    reason: np.ndarray
    quality: np.ndarray
    groups: np.ndarray


def add_task_onehot(x: np.ndarray, task_id: np.ndarray) -> np.ndarray:
    task_id = np.asarray(task_id, dtype=np.int64)
    onehot = np.zeros((len(task_id), 2), dtype=np.float32)
    valid = (task_id >= 0) & (task_id < 2)
    onehot[np.arange(len(task_id))[valid], task_id[valid]] = 1.0
    return np.concatenate([x.astype(np.float32), onehot], axis=1)


def load_sections(args: argparse.Namespace) -> List[Section]:
    sections: List[Section] = []

    insertion = np.load(args.insertion_features, allow_pickle=True)
    insertion_x = np.concatenate(
        [
            insertion["marker_proxy"].astype(np.float32),
            insertion["action_proxy"].astype(np.float32),
        ],
        axis=1,
    )
    sections.append(
        Section(
            name="insertion_proxy",
            x=insertion_x,
            binary=insertion["binary"].astype(np.int64),
            reason=insertion["reason"].astype(np.int64),
            quality=insertion["quality"].astype(np.float32),
            groups=insertion["groups"].astype(str),
        )
    )

    ptg = np.load(args.ptg_features, allow_pickle=True)
    task = ptg["task"].astype(str)
    task_id = ptg["task_id"].astype(np.int64)
    ptg_x = ptg["X"].astype(np.float32)
    for name, mask, use_task_id in [
        ("ptg_board", task == "board", False),
        ("ptg_insertion", task == "insertion", False),
        ("ptg_mixed", np.ones(len(task), dtype=bool), True),
    ]:
        x = ptg_x[mask]
        if use_task_id:
            x = add_task_onehot(x, task_id[mask])
        sections.append(
            Section(
                name=name,
                x=x,
                binary=ptg["binary"][mask].astype(np.int64),
                reason=ptg["reason"][mask].astype(np.int64),
                quality=ptg["quality"][mask].astype(np.float32),
                groups=ptg["groups"][mask].astype(str),
            )
        )
    return sections


def fold_splits(groups: np.ndarray, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    splits = min(n_splits, len(unique_groups))
    if splits < 2:
        raise ValueError("Need at least two episode groups for GroupKFold.")
    return list(GroupKFold(n_splits=splits).split(np.zeros(len(groups)), groups=groups))


def evaluate_section(section: Section, models: Dict[str, Any], args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    splits = fold_splits(section.groups, args.n_splits)
    section_dir = out_dir / section.name
    section_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        split_manifest.append(
            {
                "fold": int(fold),
                "train_groups": sorted(np.unique(section.groups[train_idx]).tolist()),
                "test_groups": sorted(np.unique(section.groups[test_idx]).tolist()),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    (section_dir / "group_splits.json").write_text(json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    result: Dict[str, Any] = {
        "n": int(len(section.x)),
        "n_groups": int(len(np.unique(section.groups))),
        "feature_dim": int(section.x.shape[1]),
        "binary_counts": {str(k): int(v) for k, v in zip(*np.unique(section.binary, return_counts=True))},
        "reason_counts": {str(k): int(v) for k, v in zip(*np.unique(section.reason, return_counts=True))},
        "models": {},
    }

    for model_name, model_proto in models.items():
        fold_rows: List[Dict[str, Any]] = []
        oof_score = np.full(len(section.x), np.nan, dtype=np.float64)
        oof_binary_pred = np.full(len(section.x), -1, dtype=np.int64)
        oof_reason_pred = np.full(len(section.x), -1, dtype=np.int64)

        for fold, (train_idx, test_idx) in enumerate(splits):
            x_train, x_test = section.x[train_idx], section.x[test_idx]
            train_binary_mask = section.binary[train_idx] >= 0
            test_binary_mask = section.binary[test_idx] >= 0
            y_train = section.binary[train_idx][train_binary_mask]
            y_test = section.binary[test_idx][test_binary_mask]
            q_test = section.quality[test_idx]

            binary_model = clone(model_proto)
            binary_model.fit(x_train[train_binary_mask], y_train)
            train_score = predict_score(binary_model, x_train[train_binary_mask])
            threshold = best_balanced_threshold(y_train, train_score)
            test_score = predict_score(binary_model, x_test)
            test_pred = (test_score >= threshold).astype(np.int64)
            oof_score[test_idx] = test_score
            binary_test_idx = test_idx[test_binary_mask]
            oof_binary_pred[binary_test_idx] = test_pred[test_binary_mask]

            reason_model = clone(model_proto)
            reason_model.fit(x_train, section.reason[train_idx])
            reason_pred = reason_model.predict(x_test).astype(np.int64)
            oof_reason_pred[test_idx] = reason_pred

            fold_rows.append(
                {
                    "fold": int(fold),
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "n_binary_train": int(train_binary_mask.sum()),
                    "n_binary_test": int(test_binary_mask.sum()),
                    "n_train_groups": int(len(np.unique(section.groups[train_idx]))),
                    "n_test_groups": int(len(np.unique(section.groups[test_idx]))),
                    "threshold": float(threshold),
                    "binary_auc": safe_auc(y_test, test_score[test_binary_mask]),
                    "binary_ap": safe_ap(y_test, test_score[test_binary_mask]),
                    "binary_balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred[test_binary_mask])) if len(y_test) else None,
                    "binary_macro_f1": float(f1_score(y_test, test_pred[test_binary_mask], average="macro", zero_division=0)) if len(y_test) else None,
                    "reason_macro_f1": float(f1_score(section.reason[test_idx], reason_pred, average="macro", zero_division=0)),
                    "quality_spearman": spearman_simple(test_score, q_test),
                    "quality_pearson": finite_corr(test_score, q_test),
                }
            )

        valid = np.isfinite(oof_score)
        valid_binary = valid & (section.binary >= 0) & (oof_binary_pred >= 0)
        deciles = decile_summary(oof_score[valid], section.quality[valid], section.binary[valid], args.n_bins)
        overall = {
            "binary_auc": safe_auc(section.binary[valid_binary], oof_score[valid_binary]),
            "binary_ap": safe_ap(section.binary[valid_binary], oof_score[valid_binary]),
            "binary_balanced_accuracy": float(balanced_accuracy_score(section.binary[valid_binary], oof_binary_pred[valid_binary])) if np.any(valid_binary) else None,
            "binary_macro_f1": float(f1_score(section.binary[valid_binary], oof_binary_pred[valid_binary], average="macro", zero_division=0)) if np.any(valid_binary) else None,
            "reason_macro_f1": float(f1_score(section.reason[valid], oof_reason_pred[valid], average="macro", zero_division=0)),
            "quality_spearman": spearman_simple(oof_score[valid], section.quality[valid]),
            "quality_pearson": finite_corr(oof_score[valid], section.quality[valid]),
            "fold_mean": {
                key: nanmean(row.get(key) for row in fold_rows)
                for key in [
                    "binary_auc",
                    "binary_ap",
                    "binary_balanced_accuracy",
                    "binary_macro_f1",
                    "reason_macro_f1",
                    "quality_spearman",
                    "quality_pearson",
                ]
            },
            "fold_std": {
                key: nanstd(row.get(key) for row in fold_rows)
                for key in [
                    "binary_auc",
                    "binary_ap",
                    "binary_balanced_accuracy",
                    "binary_macro_f1",
                    "reason_macro_f1",
                    "quality_spearman",
                    "quality_pearson",
                ]
            },
            "decile": deciles,
        }
        result["models"][model_name] = {"overall": overall, "folds": fold_rows}
        write_fold_csv(fold_rows, section_dir / f"{model_name}_fold_metrics.csv")
        plot_deciles(deciles["bins"], f"{section.name} / {model_name}", section_dir / f"{model_name}_deciles.png")

    return result


def write_fold_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_deciles(rows: List[Dict[str, Any]], title: str, path: Path) -> None:
    if not rows:
        return
    xs = [r["bin"] for r in rows]
    q = [r["quality_mean"] for r in rows]
    g = [r["good_rate"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    axes[0].plot(xs, q, marker="o", color="#2563eb")
    axes[0].set_title("Target quality by score decile")
    axes[0].set_ylabel("quality")
    axes[1].plot(xs, g, marker="o", color="#16a34a")
    axes[1].set_title("Good rate by score decile")
    axes[1].set_ylabel("good rate")
    for ax in axes:
        ax.set_xlabel("score decile, low to high")
        ax.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_model_name(section_result: Dict[str, Any]) -> str:
    def key(item: Tuple[str, Any]) -> Tuple[float, float, float, float]:
        overall = item[1]["overall"]
        dec = overall["decile"]
        auc = -1.0 if overall["binary_auc"] is None else float(overall["binary_auc"])
        return (
            float(overall["quality_spearman"]),
            float(dec.get("quality_positive_step_rate") or 0.0),
            float(dec.get("top_bottom_quality_gap") or 0.0),
            auc,
        )

    return max(section_result["models"].items(), key=key)[0]


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality GroupKFold Scorer Evaluation",
        "",
        "This evaluation uses episode-level `GroupKFold`; train and test windows never share the same episode group.",
        "It evaluates held-out-episode generalization for scorer features, not real robot rollout success.",
        "",
        "## Inputs",
        "",
        f"- insertion features: `{result['inputs']['insertion_features']}`",
        f"- PTG features: `{result['inputs']['ptg_features']}`",
        "",
        "## Best Models By Section",
        "",
        "| section | best model | AUC | AP | bACC | binary F1 | reason F1 | Spearman(q) | q decile step | q top-bottom gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for section_name in result["section_order"]:
        section = result[section_name]
        best = section["best_model"]
        overall = section["models"][best]["overall"]
        dec = overall["decile"]
        lines.append(
            f"| {section_name} | {best} | {fmt(overall['binary_auc'])} | {fmt(overall['binary_ap'])} | "
            f"{fmt(overall['binary_balanced_accuracy'])} | {fmt(overall['binary_macro_f1'])} | "
            f"{fmt(overall['reason_macro_f1'])} | {fmt(overall['quality_spearman'])} | "
            f"{fmt(dec.get('quality_positive_step_rate'))} | {fmt(dec.get('top_bottom_quality_gap'))} |"
        )
    lines.extend(["", "## All Models", ""])
    for section_name in result["section_order"]:
        section = result[section_name]
        lines.extend(
            [
                f"### {section_name}",
                "",
                f"- samples: `{section['n']}`",
                f"- groups: `{section['n_groups']}`",
                f"- feature dim: `{section['feature_dim']}`",
                "",
                "| model | AUC | AP | bACC | binary F1 | reason F1 | Spearman(q) | q decile step | q top-bottom gap |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for model_name, payload in section["models"].items():
            overall = payload["overall"]
            dec = overall["decile"]
            lines.append(
                f"| {model_name} | {fmt(overall['binary_auc'])} | {fmt(overall['binary_ap'])} | "
                f"{fmt(overall['binary_balanced_accuracy'])} | {fmt(overall['binary_macro_f1'])} | "
                f"{fmt(overall['reason_macro_f1'])} | {fmt(overall['quality_spearman'])} | "
                f"{fmt(dec.get('quality_positive_step_rate'))} | {fmt(dec.get('top_bottom_quality_gap'))} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Evidence Boundaries",
            "",
            "- This is stricter than random frame/window split, but still uses weak/proxy labels.",
            "- It does not prove real robot improvement; real rollout force/marker contact-phase metrics are still required.",
            "- A scorer suitable for DP guidance must also pass gradient and Foresight-score-to-real-quality audits.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = make_models(args.seed, args.n_jobs, args.include_mlp)
    sections = load_sections(args)
    result: Dict[str, Any] = {
        "purpose": "Episode-level GroupKFold evaluation for TacQuality scorer features.",
        "protocol": {
            "n_splits": int(args.n_splits),
            "n_bins": int(args.n_bins),
            "models": list(models.keys()),
            "split": "GroupKFold by episode groups from feature caches.",
        },
        "inputs": {
            "insertion_features": str(args.insertion_features),
            "ptg_features": str(args.ptg_features),
        },
        "section_order": [s.name for s in sections],
    }
    for section in sections:
        print(f"[section] {section.name}: n={len(section.x)} groups={len(np.unique(section.groups))} dim={section.x.shape[1]}", flush=True)
        section_result = evaluate_section(section, models, args, out_dir)
        section_result["best_model"] = best_model_name(section_result)
        result[section.name] = section_result
        print(f"  best={section_result['best_model']}", flush=True)

    json_path = out_dir / "groupkfold_scorer_eval.json"
    md_path = out_dir / "groupkfold_scorer_eval.md"
    json_path.write_text(json.dumps(jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2, ensure_ascii=False))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--insertion_features", type=Path, default=DEFAULT_INSERTION_FEATURES)
    parser.add_argument("--ptg_features", type=Path, default=DEFAULT_PTG_FEATURES)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--include_mlp", action="store_true", help="Also evaluate a small MLP. Slower on CPU.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
