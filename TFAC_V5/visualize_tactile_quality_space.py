"""Visualize classifier space for insert vs pre-bounce tactile latents."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CACHE = "/home/chenshuai/Project/output/ptg_quality_eval/insert_prebounce_episode_cache.npz"
OUT_DIR = "/home/chenshuai/Project/output/ptg_quality_eval/figures"


def balance_indices(y, seed=42):
    rng = np.random.default_rng(seed)
    cls0 = np.flatnonzero(y == 0)
    cls1 = np.flatnonzero(y == 1)
    n = min(len(cls0), len(cls1))
    idx = np.concatenate([rng.choice(cls0, n, replace=False), rng.choice(cls1, n, replace=False)])
    rng.shuffle(idx)
    return idx


def sample_for_vis(X, y, groups, group_names, max_per_class=1200, seed=42):
    rng = np.random.default_rng(seed)
    selected = []
    for label in [0, 1]:
        idx = np.flatnonzero(y == label)
        n = min(max_per_class, len(idx))
        selected.extend(rng.choice(idx, n, replace=False).tolist())
    selected = np.array(selected)
    rng.shuffle(selected)
    return X[selected], y[selected], groups[selected], group_names[selected], selected


def scatter_by_label(ax, Z, y, title, s=12, alpha=0.55):
    colors = {0: "#2ca25f", 1: "#de2d26"}
    names = {0: "Insert", 1: "Pre-bounce"}
    for label in [0, 1]:
        mask = y == label
        ax.scatter(Z[mask, 0], Z[mask, 1], s=s, alpha=alpha, c=colors[label], label=names[label], edgecolors="none")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")


def plot_projection_grid(X_vis, y_vis, out_dir):
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(StandardScaler().fit_transform(X_vis))

    lda = LinearDiscriminantAnalysis(n_components=1)
    Z_lda_1d = lda.fit_transform(StandardScaler().fit_transform(X_vis), y_vis).ravel()
    Z_lda = np.column_stack([Z_lda_1d, np.zeros_like(Z_lda_1d)])

    tsne = TSNE(n_components=2, perplexity=35, n_iter=1000, init="pca", learning_rate="auto", random_state=42)
    Z_tsne = tsne.fit_transform(StandardScaler().fit_transform(X_vis))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    scatter_by_label(
        axes[0],
        Z_pca,
        y_vis,
        f"PCA label space (var={pca.explained_variance_ratio_[:2].sum():.1%})",
    )
    scatter_by_label(axes[1], Z_tsne, y_vis, "t-SNE label space")
    scatter_by_label(axes[2], Z_lda, y_vis, "LDA discriminant axis")
    axes[2].set_ylabel("jitter-free axis")
    fig.tight_layout()
    path = os.path.join(out_dir, "quality_space_pca_tsne_lda.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def train_holdout_mlp(X, y, groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=2026)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    keep = balance_indices(y[train_idx], seed=2026)
    train_bal = train_idx[keep]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    batch_size=256,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X[train_bal], y[train_bal])
    prob = model.predict_proba(X[test_idx])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    metrics = {
        "accuracy": float((pred == y[test_idx]).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
        "f1_prebounce": float(f1_score(y[test_idx], pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y[test_idx], prob)),
        "confusion_matrix": confusion_matrix(y[test_idx], pred).tolist(),
    }
    return model, train_idx, test_idx, pred, prob, metrics


def plot_holdout_predictions(X, y, group_names, test_idx, pred, prob, out_dir):
    X_test = X[test_idx]
    y_test = y[test_idx]
    scaler = StandardScaler()
    Z = PCA(n_components=2, random_state=42).fit_transform(scaler.fit_transform(X_test))
    correct = pred == y_test

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    scatter_by_label(axes[0], Z, y_test, "Holdout true labels")
    axes[1].scatter(
        Z[correct, 0],
        Z[correct, 1],
        s=10,
        alpha=0.35,
        c="#737373",
        label="correct",
        edgecolors="none",
    )
    axes[1].scatter(
        Z[~correct, 0],
        Z[~correct, 1],
        s=28,
        alpha=0.9,
        c="#fb6a4a",
        label="wrong",
        edgecolors="black",
        linewidths=0.2,
    )
    axes[1].set_title("MLP holdout mistakes on PCA space")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    pred_path = os.path.join(out_dir, "holdout_mlp_predictions_pca.png")
    fig.savefig(pred_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(prob[y_test == 0], bins=45, alpha=0.65, density=True, color="#2ca25f", label="Insert")
    ax.hist(prob[y_test == 1], bins=45, alpha=0.65, density=True, color="#de2d26", label="Pre-bounce")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("MLP P(pre-bounce)")
    ax.set_ylabel("Density")
    ax.set_title("Holdout score distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    score_path = os.path.join(out_dir, "holdout_mlp_score_distribution.png")
    fig.savefig(score_path, dpi=180)
    plt.close(fig)

    episode_rows = []
    for ep in np.unique(group_names[test_idx]):
        mask = group_names[test_idx] == ep
        episode_rows.append(
            {
                "episode": str(ep),
                "n": int(mask.sum()),
                "wrong": int((~correct[mask]).sum()),
                "error_rate": float((~correct[mask]).mean()),
                "prebounce_ratio": float((y_test[mask] == 1).mean()),
            }
        )
    episode_rows.sort(key=lambda r: (-r["error_rate"], -r["n"]))
    top = episode_rows[:20]
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.bar([r["episode"] for r in top], [r["error_rate"] for r in top], color="#756bb1")
    ax.set_ylabel("Error rate")
    ax.set_title("Top holdout episodes by MLP error rate")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    episode_path = os.path.join(out_dir, "holdout_episode_error_rates.png")
    fig.savefig(episode_path, dpi=180)
    plt.close(fig)

    return pred_path, score_path, episode_path, episode_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-per-class", type=int, default=1200)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = np.load(args.cache, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    groups = data["groups"]
    group_names = data["group_names"]

    X_vis, y_vis, _, _, _ = sample_for_vis(X, y, groups, group_names, max_per_class=args.max_per_class)
    projection_path = plot_projection_grid(X_vis, y_vis, args.out_dir)
    _, _, test_idx, pred, prob, metrics = train_holdout_mlp(X, y, groups)
    pred_path, score_path, episode_path, episode_rows = plot_holdout_predictions(
        X, y, group_names, test_idx, pred, prob, args.out_dir
    )

    summary = {
        "cache": args.cache,
        "out_dir": args.out_dir,
        "n_total": int(len(X)),
        "n_insert": int((y == 0).sum()),
        "n_pre_bounce": int((y == 1).sum()),
        "n_groups": int(len(np.unique(groups))),
        "holdout_mlp_metrics": metrics,
        "figures": {
            "projection_grid": projection_path,
            "holdout_predictions": pred_path,
            "score_distribution": score_path,
            "episode_error_rates": episode_path,
        },
        "top_error_episodes": episode_rows[:20],
    }
    summary_path = os.path.join(args.out_dir, "quality_space_visualization_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["holdout_mlp_metrics"], indent=2))
    for name, path in summary["figures"].items():
        print(f"{name}: {path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
