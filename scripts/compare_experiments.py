"""
Compare multiple TFAC experiments on the same dataset.
Usage: python scripts/compare_experiments.py [exp_dirs...]
If no args given, auto-discover experiments in /home/chenshuai/data/xiaomi_act/
"""
import os
import sys
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DEFAULT_BASE = "/home/chenshuai/data/xiaomi_act"


def load_experiment(exp_dir):
    """Load an experiment's args, validation history, and training history."""
    exp = {"dir": exp_dir, "name": os.path.basename(exp_dir)}

    # Args
    args_path = os.path.join(exp_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path) as f:
            exp["args"] = json.load(f)
    else:
        exp["args"] = {}

    # Validation history
    val_path = os.path.join(exp_dir, "validation_history.pkl")
    if os.path.exists(val_path):
        with open(val_path, "rb") as f:
            exp["val_hist"] = pickle.load(f)
    else:
        exp["val_hist"] = None

    # Training history
    train_path = os.path.join(exp_dir, "train_history.pkl")
    if os.path.exists(train_path):
        with open(train_path, "rb") as f:
            exp["train_hist"] = pickle.load(f)
    else:
        exp["train_hist"] = None

    return exp


def extract_key_metrics(exp):
    """Extract key metrics from experiment."""
    if exp["val_hist"] is None:
        return None

    val = exp["val_hist"]
    n_epochs = len(val)

    # Find best A2 epoch
    best_idx = min(range(n_epochs),
                   key=lambda i: val[i]["l1_final"].item())
    best = val[best_idx]

    # A1 vs A2 at best epoch
    a1 = best["l1_draft"].item()
    a2 = best["l1_final"].item()
    gap_pct = (a1 - a2) / a1 * 100 if a1 > 0 else 0

    metrics = {
        "n_epochs": n_epochs,
        "best_epoch": best_idx,
        "best_l1_final": a2,
        "best_l1_draft": a1,
        "a2_improvement": gap_pct,
        "best_loss": best["loss"].item(),
        "foresight_tac": best.get("foresight_tac", torch.tensor(float("nan"))).item(),
        "contrastive": best.get("contrastive", torch.tensor(float("nan"))).item(),
        "kl": best.get("kl", torch.tensor(float("nan"))).item(),
    }

    # Final epoch metrics
    final = val[-1]
    metrics["final_l1_final"] = final["l1_final"].item()
    metrics["final_l1_draft"] = final["l1_draft"].item()

    # Gate weights at best epoch (if available)
    if "gate_fut" in best:
        metrics["gate_mem"] = best["gate_mem"].item()
        metrics["gate_a1"] = best["gate_a1"].item()
        metrics["gate_fut"] = best["gate_fut"].item()

    # Config info
    args = exp["args"]
    metrics["fusion_mode"] = args.get("fusion_mode", "?")
    metrics["a2_init"] = args.get("a2_init", "?")
    metrics["tactile_mode"] = args.get("tactile_mode", "?")
    metrics["dataset_dir"] = args.get("dataset_dir", "?")
    metrics["lambda_foresight"] = args.get("lambda_foresight", "?")

    return metrics


def plot_comparison(experiments, save_path=None):
    """Plot l1_final comparison across experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Val l1_final
    ax = axes[0, 0]
    for exp in experiments:
        if exp["val_hist"]:
            vals = [d["l1_final"].item() for d in exp["val_hist"]]
            ax.plot(vals, label=exp["name"], alpha=0.8)
    ax.set_title("Val l1_final (A2)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Plot 2: Val l1_draft
    ax = axes[0, 1]
    for exp in experiments:
        if exp["val_hist"]:
            vals = [d["l1_draft"].item() for d in exp["val_hist"]]
            ax.plot(vals, label=exp["name"], alpha=0.8)
    ax.set_title("Val l1_draft (A1)")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Plot 3: A2 improvement over A1 (percentage)
    ax = axes[1, 0]
    for exp in experiments:
        if exp["val_hist"]:
            gaps = []
            for d in exp["val_hist"]:
                a1 = d["l1_draft"].item()
                a2 = d["l1_final"].item()
                gap = (a1 - a2) / a1 * 100 if a1 > 0 else 0
                gaps.append(gap)
            ax.plot(gaps, label=exp["name"], alpha=0.8)
    ax.set_title("A2 improvement over A1 (%)")
    ax.set_xlabel("Epoch")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(fontsize=8)

    # Plot 4: Foresight tactile loss
    ax = axes[1, 1]
    for exp in experiments:
        if exp["val_hist"]:
            vals = [d.get("foresight_tac", torch.tensor(float("nan"))).item()
                    for d in exp["val_hist"]]
            if not all(np.isnan(vals)):
                ax.plot(vals, label=exp["name"], alpha=0.8)
    ax.set_title("Foresight Tactile Loss")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison plot to {save_path}")
    plt.close()


def main():
    # Auto-discover or use CLI args
    if len(sys.argv) > 1:
        exp_dirs = sys.argv[1:]
    else:
        # Auto-discover 0414 experiments
        base = DEFAULT_BASE
        exp_dirs = []
        for name in os.listdir(base):
            if "0414" in name:
                path = os.path.join(base, name)
                if os.path.isdir(path) and os.path.exists(os.path.join(path, "args.json")):
                    exp_dirs.append(path)
        exp_dirs.sort()

    if not exp_dirs:
        print("No experiments found.")
        return

    print(f"Found {len(exp_dirs)} experiments:")
    for d in exp_dirs:
        print(f"  {d}")

    # Load experiments
    experiments = [load_experiment(d) for d in exp_dirs]

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Name':<30} {'Fusion':<12} {'a2_init':<10} {'Best A2':<10} {'Best A1':<10} {'A2>A1%':<8} {'Epoch':<6} {'Fore.tac':<10}")
    print("-" * 100)

    for exp in experiments:
        m = extract_key_metrics(exp)
        if m is None:
            print(f"{exp['name']:<30} (no validation data)")
            continue
        print(f"{exp['name']:<30} {m['fusion_mode']:<12} {m['a2_init']:<10} "
              f"{m['best_l1_final']:<10.4f} {m['best_l1_draft']:<10.4f} "
              f"{m['a2_improvement']:<8.1f} {m['best_epoch']:<6} "
              f"{m['foresight_tac']:<10.4f}")
    print("=" * 100)

    # Plot comparison
    save_dir = os.path.join(DEFAULT_BASE, "comparison_plots")
    os.makedirs(save_dir, exist_ok=True)
    plot_comparison(experiments, os.path.join(save_dir, "0414_comparison.png"))


if __name__ == "__main__":
    main()
