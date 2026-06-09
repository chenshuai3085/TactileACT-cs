"""Visualize and audit the PTG tactile quality scorer space.

This script reads previously generated scorer datasets/results and creates an
auditable view of whether the quality taxonomy is meaningful for DP guidance:

  - multi-class reason space, colored by task and tactile-quality reason;
  - good/bad space with weak/no-contact samples masked separately;
  - continuous quality target distribution by task/reason;
  - summary comparing the current proxy scorer to marker-field and latent
    taxonomy baselines.

It does not rebuild labels or retrain models.  It is meant as a lightweight
diagnostic after scorer experiments.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")
DEFAULT_METADATA = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_metadata.json")
DEFAULT_PTG_EVAL = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json")
DEFAULT_MARKER_FIELD_EVAL = Path("/home/chenshuai/Project/output/marker_field_scorer/marker_field_scorer_eval.json")
DEFAULT_UNIFIED_EVAL = Path("/home/chenshuai/Project/output/unified_quality_taxonomy/unified_quality_eval_fast.json")
DEFAULT_EVIDENCE = Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json")
DEFAULT_OUT = Path("/home/chenshuai/Project/output/ptg_scorer_space")


TASK_COLORS = {"insertion": "#1f77b4", "board": "#d62728"}
BINARY_COLORS = {-1: "#9e9e9e", 0: "#c62828", 1: "#2e7d32"}
REASON_COLORS = {
    0: "#8d6e63",
    1: "#2e7d32",
    2: "#c62828",
    3: "#ef6c00",
    4: "#6a1b9a",
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d: Dict[str, Any], dotted: str, default=None):
    cur: Any = d
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def balanced_sample_indices(task: np.ndarray, reason: np.ndarray, max_per_task_reason: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for task_name in sorted(np.unique(task).tolist()):
        for cls in sorted(np.unique(reason).tolist()):
            idx = np.flatnonzero((task == task_name) & (reason == cls))
            if len(idx):
                keep.extend(rng.choice(idx, min(max_per_task_reason, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


def compute_embeddings(X: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    Xs = StandardScaler().fit_transform(X)
    pca_model = PCA(n_components=min(24, Xs.shape[1]), random_state=seed)
    Xp = pca_model.fit_transform(Xs)
    pca2 = Xp[:, :2]
    perplexity = max(5, min(35, (len(Xp) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=seed,
    )
    Xt = tsne.fit_transform(Xp)
    return pca2, Xt, float(pca_model.explained_variance_ratio_[:2].sum())


def scatter_discrete(ax, emb, labels, names, colors, title: str):
    for cls in sorted(np.unique(labels).tolist()):
        mask = labels == cls
        name = names.get(str(cls), names.get(int(cls), str(cls)))
        ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.70, c=colors.get(int(cls), "#333333"), label=name)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, markerscale=1.8, loc="best")


def scatter_task(ax, emb, task, title: str):
    for name in sorted(np.unique(task).tolist()):
        mask = task == name
        ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.70, c=TASK_COLORS.get(name, "#333333"), label=name)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=8, markerscale=1.8, loc="best")


def plot_space(pca, tsne, task, reason, binary, quality, reason_names, out_dir: Path, pca_var: float):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    scatter_discrete(axes[0, 0], pca, reason, reason_names, REASON_COLORS, f"PCA by reason (var={pca_var:.1%})")
    scatter_discrete(axes[0, 1], tsne, reason, reason_names, REASON_COLORS, "t-SNE by reason")
    scatter_task(axes[0, 2], tsne, task, "t-SNE by task")
    scatter_discrete(axes[1, 0], pca, binary, {-1: "neutral", 0: "bad", 1: "good"}, BINARY_COLORS, "PCA good/bad/neutral")
    im = axes[1, 1].scatter(tsne[:, 0], tsne[:, 1], s=10, c=quality, cmap="viridis", alpha=0.75)
    axes[1, 1].set_title("t-SNE by continuous quality target")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for task_name in sorted(np.unique(task).tolist()):
        mask = task == task_name
        axes[1, 2].scatter(tsne[mask, 0], tsne[mask, 1], s=10, alpha=0.55, c=TASK_COLORS.get(task_name), label=task_name)
    axes[1, 2].set_title("Task-conditioned overlap check")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    axes[1, 2].legend(frameon=False, fontsize=8, markerscale=1.8)
    fig.tight_layout()
    path = out_dir / "ptg_proxy_reason_task_space.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_task_facets(tsne, task, reason, quality, reason_names, out_dir: Path):
    task_values = sorted(np.unique(task).tolist())
    fig, axes = plt.subplots(len(task_values), 2, figsize=(12, 5 * len(task_values)))
    if len(task_values) == 1:
        axes = np.array([axes])
    for row, task_name in enumerate(task_values):
        mask = task == task_name
        scatter_discrete(
            axes[row, 0],
            tsne[mask],
            reason[mask],
            reason_names,
            REASON_COLORS,
            f"{task_name}: reason classes",
        )
        im = axes[row, 1].scatter(tsne[mask, 0], tsne[mask, 1], s=12, c=quality[mask], cmap="viridis", alpha=0.78)
        axes[row, 1].set_title(f"{task_name}: quality target")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        fig.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = out_dir / "ptg_proxy_task_facets.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def plot_quality_distributions(task, reason, binary, quality, reason_names, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for task_name in sorted(np.unique(task).tolist()):
        axes[0].hist(quality[task == task_name], bins=30, alpha=0.55, label=task_name)
    axes[0].set_title("Quality target by task")
    axes[0].set_xlabel("quality")
    axes[0].legend(frameon=False)

    for cls in sorted(np.unique(reason).tolist()):
        name = reason_names.get(str(cls), str(cls))
        axes[1].hist(quality[reason == cls], bins=25, alpha=0.50, label=name)
    axes[1].set_title("Quality target by reason")
    axes[1].set_xlabel("quality")
    axes[1].legend(frameon=False, fontsize=8)

    labels = [-1, 0, 1]
    values = [int(np.sum(binary == x)) for x in labels]
    axes[2].bar(["neutral", "bad", "good"], values, color=[BINARY_COLORS[x] for x in labels])
    axes[2].set_title("Binary label counts")
    axes[2].set_ylabel("count")
    fig.tight_layout()
    path = out_dir / "ptg_proxy_quality_distributions.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def summarize_metrics(ptg_eval, marker_field_eval, unified_eval, evidence) -> Dict[str, Any]:
    return {
        "ptg_proxy_v2": {
            "mixed_binary_auc": get(ptg_eval, "mixed_group_cv.binary_auc.mean"),
            "mixed_binary_balanced_accuracy": get(ptg_eval, "mixed_group_cv.binary_balanced_accuracy.mean"),
            "mixed_reason_macro_f1": get(ptg_eval, "mixed_group_cv.reason_macro_f1.mean"),
            "mixed_quality_corr": get(ptg_eval, "mixed_group_cv.quality_corr.mean"),
        },
        "marker_field_scorer": {
            "mixed_binary_auc": get(marker_field_eval, "mixed_group_cv.binary_auc.mean"),
            "mixed_binary_balanced_accuracy": get(marker_field_eval, "mixed_group_cv.binary_balanced_accuracy.mean"),
            "mixed_t4_macro_f1": get(marker_field_eval, "mixed_group_cv.t4_macro_f1.mean"),
            "mixed_score_corr": get(marker_field_eval, "mixed_group_cv.score_corr.mean"),
        },
        "unified_traditional_baseline": {
            "best_candidate": (unified_eval.get("best_candidates") or [None])[0],
        },
        "guidance_evidence": evidence.get("completion_assessment", {}),
        "selected_current_best": {
            "name": "TacQualityEnergy / PTGProxyScorerV2Runtime",
            "reason": (
                "Best current deployable scorer because it has the strongest episode-level GroupKFold "
                "classification, a continuous quality head, differentiable torch runtime, board guidance "
                "readiness, and surrogate full-chain action-gradient evidence."
            ),
        },
    }


def write_markdown(summary: Dict[str, Any], out_path: Path):
    p = summary["metrics"]["ptg_proxy_v2"]
    m = summary["metrics"]["marker_field_scorer"]
    best = summary["metrics"]["unified_traditional_baseline"]["best_candidate"]
    lines = [
        "# PTG Scorer Space Audit",
        "",
        "## Selected Current Best",
        "",
        "- TacQualityEnergy / PTGProxyScorerV2Runtime remains the current best scorer for gradient guidance.",
        "- It is not just binary classification: it has good/bad, failure-reason, and continuous quality heads.",
        "- The raw marker-field neural scorer is kept as an ablation, not the main scorer, because it underperforms the proxy scorer on episode-level GroupKFold.",
        "",
        "## Key Metrics",
        "",
        "| scorer | binary AUC | balanced acc | reason/T4 F1 | quality corr |",
        "|---|---:|---:|---:|---:|",
        f"| PTG proxy v2 | {p['mixed_binary_auc']:.4f} | {p['mixed_binary_balanced_accuracy']:.4f} | {p['mixed_reason_macro_f1']:.4f} | {p['mixed_quality_corr']:.4f} |",
        f"| marker-field NN | {m['mixed_binary_auc']:.4f} | {m['mixed_binary_balanced_accuracy']:.4f} | {m['mixed_t4_macro_f1']:.4f} | {m['mixed_score_corr']:.4f} |",
        "",
        "## Traditional Baseline",
        "",
        f"Best traditional baseline: `{best}`",
        "",
        "## Figures",
        "",
    ]
    for name, path in summary["figures"].items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Multi-class reason labels are necessary: board bad samples include too-light, too-heavy, rough-force, and rough-motion; insertion bad samples include risk/impact around bounce episodes.",
            "2. A task-conditioned scorer is necessary: cross-task zero-shot metrics are weak, while mixed task-conditioned GroupKFold is strong.",
            "3. For DP guidance, the selected score must be a continuous differentiable energy, not only a hard class label.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--ptg_eval", type=Path, default=DEFAULT_PTG_EVAL)
    parser.add_argument("--marker_field_eval", type=Path, default=DEFAULT_MARKER_FIELD_EVAL)
    parser.add_argument("--unified_eval", type=Path, default=DEFAULT_UNIFIED_EVAL)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max_per_task_reason", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.features, allow_pickle=True)
    meta = load_json(args.metadata)
    X = data["X"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
    task = data["task"]

    idx = balanced_sample_indices(task, reason, args.max_per_task_reason, args.seed)
    pca, tsne, pca_var = compute_embeddings(X[idx], args.seed)
    reason_names = meta.get("reason_names", {})

    figures = {
        "reason_task_space": plot_space(
            pca, tsne, task[idx], reason[idx], binary[idx], quality[idx], reason_names, args.out_dir, pca_var
        ),
        "task_facets": plot_task_facets(tsne, task[idx], reason[idx], quality[idx], reason_names, args.out_dir),
        "quality_distributions": plot_quality_distributions(task, reason, binary, quality, reason_names, args.out_dir),
    }

    metrics = summarize_metrics(
        load_json(args.ptg_eval),
        load_json(args.marker_field_eval),
        load_json(args.unified_eval),
        load_json(args.evidence),
    )
    summary = {
        "input": {
            "features": str(args.features),
            "metadata": str(args.metadata),
            "n_total": int(len(X)),
            "n_visualized": int(len(idx)),
            "task_counts": {str(k): int(v) for k, v in Counter(task.tolist()).items()},
            "reason_counts": {str(k): int(v) for k, v in Counter(reason.tolist()).items()},
            "binary_counts": {str(k): int(v) for k, v in Counter(binary.tolist()).items()},
            "pca2_explained_variance": pca_var,
        },
        "figures": figures,
        "metrics": metrics,
    }
    json_path = args.out_dir / "ptg_scorer_space_summary.json"
    md_path = args.out_dir / "ptg_scorer_space_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    for path in figures.values():
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
