"""Summarize tactile scorer experiments and generate comparison figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path("/home/chenshuai/Project/output/tactile_scorer_comparison")
EXPERIMENTS = {
    "marker_proxy": Path("/home/chenshuai/Project/output/marker_proxy_multitask_scorer/marker_proxy_multitask_eval.json"),
    "marker_field": Path("/home/chenshuai/Project/output/marker_field_scorer/marker_field_scorer_eval.json"),
    "action_aware": Path("/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_eval.json"),
}
METRICS = [
    "binary_balanced_accuracy",
    "binary_auc",
    "t4_macro_f1",
    "score_corr",
]


def load_results():
    rows = {}
    for name, path in EXPERIMENTS.items():
        with open(path, encoding="utf-8") as f:
            rows[name] = json.load(f)
    return rows


def metric_mean(block, key):
    value = block.get(key)
    if isinstance(value, dict):
        return value.get("mean")
    return value


def metric_std(block, key):
    value = block.get(key)
    if isinstance(value, dict):
        return value.get("std")
    return None


def make_summary(results):
    summary = {
        "experiments": {},
        "best_by_metric": {},
        "interpretation": [
            "action_aware is the best mixed-task scorer by binary AUC, balanced accuracy, T4 macro-F1, and score correlation.",
            "cross-task zero-shot classification remains weak; cross-task AUC is more meaningful than a fixed 0.5 threshold.",
            "For DP guidance, use the continuous score/log P(good) with task-conditioned calibration, not a universal hard class threshold.",
        ],
    }
    for name, row in results.items():
        mixed = row["mixed_group_cv"]
        summary["experiments"][name] = {
            key: {
                "mean": metric_mean(mixed, key),
                "std": metric_std(mixed, key),
            }
            for key in METRICS
        }
        summary["experiments"][name]["cross_task"] = {
            split: {
                "binary_balanced_accuracy": metrics.get("binary_balanced_accuracy"),
                "binary_auc": metrics.get("binary_auc"),
                "t4_macro_f1": metrics.get("t4_macro_f1"),
                "score_corr": metrics.get("score_corr"),
            }
            for split, metrics in row.get("cross_task", {}).items()
        }
    for key in METRICS:
        best = max(
            ((name, metric_mean(row["mixed_group_cv"], key)) for name, row in results.items()),
            key=lambda x: -np.inf if x[1] is None else x[1],
        )
        summary["best_by_metric"][key] = {"model": best[0], "value": best[1]}
    return summary


def plot_metric_bars(summary):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = list(summary["experiments"])
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(METRICS):
        vals = [summary["experiments"][name][metric]["mean"] for name in labels]
        ax.bar(x + (i - 1.5) * width, vals, width, label=metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("group-CV mean")
    ax.set_title("Tactile scorer comparison across socket + board")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scorer_metric_comparison.png", dpi=180)
    plt.close(fig)


def plot_cross_task_auc(summary):
    labels = list(summary["experiments"])
    splits = ["insertion_to_board", "board_to_insertion"]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, split in enumerate(splits):
        vals = [summary["experiments"][name]["cross_task"][split]["binary_auc"] for name in labels]
        ax.bar(x + (i - 0.5) * width, vals, width, label=split)
    ax.axhline(0.5, color="k", lw=1, ls="--", alpha=0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("zero-shot cross-task AUC")
    ax.set_title("Cross-task ranking transfer")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cross_task_auc.png", dpi=180)
    plt.close(fig)


def write_markdown(summary):
    lines = [
        "# 触觉质量评分器实验对比摘要",
        "",
        "日期：2026-06-09",
        "",
        "## 模型对比",
        "",
        "| 模型 | Binary Bal Acc | Binary AUC | T4 Macro-F1 | Score Corr |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in summary["experiments"].items():
        lines.append(
            "| {name} | {bacc:.4f} | {auc:.4f} | {t4:.4f} | {corr:.4f} |".format(
                name=name,
                bacc=metrics["binary_balanced_accuracy"]["mean"],
                auc=metrics["binary_auc"]["mean"],
                t4=metrics["t4_macro_f1"]["mean"],
                corr=metrics["score_corr"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            "1. `action_aware` 是当前最佳模型：同时使用触觉marker场、marker物理proxy、action时序/proxy和task id。",
            "2. 它的 mixed group-CV binary AUC 达到 0.9562，balanced accuracy 达到 0.8763，score相关达到 0.7372。",
            "3. 纯跨任务零样本分类仍不可靠，说明插座和擦黑板需要 task-conditioned calibration；但跨任务AUC显示排序分数仍有一部分迁移价值。",
            "4. 对DP guidance，推荐使用连续分数 `log P(good)` 或 score head，而不是固定0.5分类阈值。",
            "",
            "## 输出图",
            "",
            "- `scorer_metric_comparison.png`",
            "- `cross_task_auc.png`",
        ]
    )
    (OUT_DIR / "scorer_experiment_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    results = load_results()
    summary = make_summary(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "scorer_experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    plot_metric_bars(summary)
    plot_cross_task_auc(summary)
    write_markdown(summary)
    print(json.dumps(summary["best_by_metric"], ensure_ascii=False, indent=2))
    print(f"Saved {OUT_DIR}")


if __name__ == "__main__":
    main()
