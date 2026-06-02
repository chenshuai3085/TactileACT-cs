"""
Analyze foresight prediction logs from real robot deployment.

Usage:
    python for_show_xiaomi/analyze_foresight_logs.py \
        --log_dir /home/chenshuai/Project/output/foresight_logs/20260602_120000 \
        --episode 0

    # Analyze all episodes
    python for_show_xiaomi/analyze_foresight_logs.py \
        --log_dir /path/to/logs --all
"""
from __future__ import annotations
import argparse
import os
import pickle
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_episode(log_dir, ep_id):
    path = os.path.join(log_dir, f"episode_{ep_id:04d}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)


def list_episodes(log_dir):
    files = sorted([f for f in os.listdir(log_dir) if f.startswith("episode_") and f.endswith(".pkl")])
    return [int(f.split("_")[1].split(".")[0]) for f in files]


def analyze_episode(data, save_dir=None):
    """Analyze a single episode's foresight logs."""
    ep = data["episode"]
    steps = data["steps"]
    n_steps = len(steps)
    print(f"\n{'='*60}")
    print(f"Episode {ep}: {n_steps} reranking steps")
    print(f"{'='*60}")

    # Collect metrics across steps
    all_scores = []          # (n_steps, K)
    all_best_idx = []        # (n_steps,)
    all_marker_pred = []     # (n_steps, K, 9, 9, 2)
    all_marker_current = []  # (n_steps, 9, 9, 2)
    all_z_pred = []          # (n_steps, K, 144)
    all_z_current = []       # (n_steps, 144)
    all_candidates = []      # (n_steps, K, pred_horizon, action_dim)

    for s in steps:
        all_scores.append(s["scores"])
        all_best_idx.append(s["best_idx"])
        all_marker_pred.append(s["marker_pred"])
        all_marker_current.append(s["marker_current"])
        all_z_pred.append(s["z_pred"])
        all_z_current.append(s["z_current"])
        all_candidates.append(s["candidates"])

    scores_arr = np.stack(all_scores)           # (T, K)
    best_idx_arr = np.array(all_best_idx)       # (T,)
    marker_pred_arr = np.stack(all_marker_pred) # (T, K, 9, 9, 2)
    marker_cur_arr = np.stack(all_marker_current)  # (T, 9, 9, 2)
    z_pred_arr = np.stack(all_z_pred)           # (T, K, 144)
    z_cur_arr = np.stack(all_z_current)         # (T, 144)
    cand_arr = np.stack(all_candidates)         # (T, K, H, 7)

    K = scores_arr.shape[1]

    # --- 1. Score statistics ---
    print(f"\n[Score Stats]")
    print(f"  K = {K} candidates")
    print(f"  Mean best score:  {scores_arr[np.arange(n_steps), best_idx_arr].mean():.4f}")
    print(f"  Mean score spread: {(scores_arr.max(axis=1) - scores_arr.min(axis=1)).mean():.4f}")
    print(f"  Best index distribution: {np.bincount(best_idx_arr, minlength=K)}")

    # --- 2. Foresight prediction analysis ---
    # For the selected candidate, compare predicted vs actual next-step marker
    pred_errors = []
    for i in range(n_steps - 1):
        best_pred = marker_pred_arr[i, best_idx_arr[i]]  # (9, 9, 2)
        actual_next = marker_cur_arr[i + 1]                # (9, 9, 2)
        error = np.linalg.norm(best_pred - actual_next)
        pred_errors.append(error)

    if pred_errors:
        pred_errors = np.array(pred_errors)
        print(f"\n[Foresight Prediction Error (selected candidate → next actual)]")
        print(f"  Mean L2 error: {pred_errors.mean():.4f}")
        print(f"  Std L2 error:  {pred_errors.std():.4f}")
        print(f"  Median:        {np.median(pred_errors):.4f}")
        print(f"  Min / Max:     {pred_errors.min():.4f} / {pred_errors.max():.4f}")

    # --- 3. Per-candidate diversity ---
    # How different are the 16 foresight predictions?
    z_pred_norms = np.linalg.norm(z_pred_arr, axis=-1)  # (T, K)
    print(f"\n[Foresight Diversity (z_pred norms)]")
    print(f"  Mean per-step z_pred std: {z_pred_norms.std(axis=1).mean():.4f}")

    # Marker prediction diversity
    marker_pred_flat = marker_pred_arr.reshape(n_steps, K, -1)  # (T, K, 162)
    marker_std_per_step = np.std(marker_pred_flat, axis=1).mean(axis=-1)  # (T,)
    print(f"  Mean per-step marker_pred std: {marker_std_per_step.mean():.4f}")

    # --- 4. Latent space analysis ---
    # Cosine similarity between z_pred and z_current
    z_cur_norm = z_cur_arr / (np.linalg.norm(z_cur_arr, axis=-1, keepdims=True) + 1e-8)
    z_pred_norm = z_pred_arr / (np.linalg.norm(z_pred_arr, axis=-1, keepdims=True) + 1e-8)
    cos_sim = np.sum(z_cur_norm[:, None, :] * z_pred_norm, axis=-1)  # (T, K)
    print(f"\n[Latent Space]")
    print(f"  Mean cos_sim(z_pred, z_current): {cos_sim.mean():.4f}")
    print(f"  Best candidate cos_sim:          {cos_sim[np.arange(n_steps), best_idx_arr].mean():.4f}")

    # --- 5. Action diversity ---
    cand_flat = cand_arr.reshape(n_steps, K, -1)  # (T, K, H*7)
    action_std = np.std(cand_flat, axis=1).mean(axis=-1)  # (T,)
    print(f"\n[Action Diversity]")
    print(f"  Mean per-step action std: {action_std.mean():.4f}")

    # --- 6. Plots ---
    if save_dir and HAS_MPL:
        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (a) Score over time
        ax = axes[0, 0]
        ax.plot(scores_arr.max(axis=1), label='best', color='green')
        ax.plot(scores_arr.min(axis=1), label='worst', color='red')
        ax.fill_between(range(n_steps), scores_arr.min(axis=1), scores_arr.max(axis=1), alpha=0.2)
        ax.set_xlabel('Rerank step')
        ax.set_ylabel('Score')
        ax.set_title('Score range over time')
        ax.legend()

        # (b) Prediction error over time
        ax = axes[0, 1]
        if pred_errors:
            ax.plot(pred_errors, marker='o', markersize=3)
            ax.axhline(pred_errors.mean(), color='red', linestyle='--', label=f'mean={pred_errors.mean():.3f}')
            ax.set_xlabel('Rerank step')
            ax.set_ylabel('L2 error')
            ax.set_title('Foresight pred error (selected → next actual)')
            ax.legend()

        # (c) Best index distribution
        ax = axes[1, 0]
        ax.bar(range(K), np.bincount(best_idx_arr, minlength=K))
        ax.set_xlabel('Candidate index')
        ax.set_ylabel('Count selected')
        ax.set_title('Best candidate selection distribution')

        # (d) Foresight diversity
        ax = axes[1, 1]
        ax.plot(marker_std_per_step, marker='o', markersize=3)
        ax.set_xlabel('Rerank step')
        ax.set_ylabel('Marker pred std')
        ax.set_title('Foresight prediction diversity across K candidates')

        plt.suptitle(f'Episode {ep} Foresight Analysis', fontsize=14)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"episode_{ep:04d}_analysis.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\n  Plot saved → {save_path}")

        # --- Marker heatmap comparison (selected candidate) ---
        n_show = min(6, n_steps)
        indices = np.linspace(0, n_steps - 1, n_show, dtype=int)
        fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
        for j, idx in enumerate(indices):
            best = best_idx_arr[idx]
            pred_m = marker_pred_arr[idx, best]  # (9, 9, 2)
            cur_m = marker_cur_arr[idx]            # (9, 9, 2)
            # Next actual (if available)
            if idx + 1 < n_steps:
                next_m = marker_cur_arr[idx + 1]
            else:
                next_m = np.zeros_like(cur_m)

            for row, (m, title_prefix) in enumerate([
                (cur_m, "Current"),
                (pred_m, "Predicted (best)"),
                (next_m, "Next actual"),
            ]):
                ax = axes[row, j]
                mag = np.linalg.norm(m, axis=-1)
                im = ax.imshow(mag, cmap='hot', vmin=0, vmax=max(mag.max(), 0.5))
                ax.set_title(f"{title_prefix}\nstep {idx}", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(f'Episode {ep}: Marker Magnitude (current → predicted → actual)', fontsize=12)
        plt.tight_layout()
        save_path2 = os.path.join(save_dir, f"episode_{ep:04d}_marker_heatmap.png")
        plt.savefig(save_path2, dpi=150)
        plt.close()
        print(f"  Marker heatmap saved → {save_path2}")

    return {
        "episode": ep,
        "n_steps": n_steps,
        "mean_best_score": scores_arr[np.arange(n_steps), best_idx_arr].mean(),
        "mean_score_spread": (scores_arr.max(axis=1) - scores_arr.min(axis=1)).mean(),
        "mean_pred_error": np.mean(pred_errors) if pred_errors else None,
        "mean_cos_sim": cos_sim.mean(),
        "mean_marker_diversity": marker_std_per_step.mean(),
        "mean_action_diversity": action_std.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze foresight prediction logs")
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--all", action="store_true", help="Analyze all episodes")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for plots (default: log_dir/plots)")
    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.log_dir, "plots")

    episodes = list_episodes(args.log_dir)
    if not episodes:
        print(f"No episode logs found in {args.log_dir}")
        return

    if args.all:
        all_results = []
        for ep_id in episodes:
            data = load_episode(args.log_dir, ep_id)
            result = analyze_episode(data, save_dir=save_dir)
            all_results.append(result)

        # Summary across episodes
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(all_results)} episodes")
        print(f"{'='*60}")
        for key in ["mean_best_score", "mean_score_spread", "mean_pred_error",
                     "mean_cos_sim", "mean_marker_diversity", "mean_action_diversity"]:
            vals = [r[key] for r in all_results if r[key] is not None]
            if vals:
                print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    else:
        data = load_episode(args.log_dir, args.episode)
        analyze_episode(data, save_dir=save_dir)


if __name__ == "__main__":
    main()
