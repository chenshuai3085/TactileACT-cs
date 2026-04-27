"""
CQF + TacDream Reranking 可视化脚本。

生成三类图表:
  1. CQF Phase A→B→C 训练曲线 (spread/rank1/loss, 有/无 perturbation 对比)
  2. Reranking 分数分布图 (expert vs noisy candidates histogram + boxplot)
  3. 逐帧 K=16 score 排名可视化 (每帧候选打分, expert 标红)

Usage:
  # 只画训练曲线 (不需要GPU)
  python TFAC_V5/visualize_cqf_results.py --mode training

  # 画 reranking 分布 + 逐帧排名 (需要GPU加载模型)
  python TFAC_V5/visualize_cqf_results.py --mode reranking \
    --dp_ckpt /home/chenshuai/Project/output/dp_joint7/dp_best.pth \
    --dp_config /home/chenshuai/Project/output/dp_joint7/config.json \
    --cqf_ckpt /home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt \
    --foresight_ckpt /home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw/foresight_best.ckpt

  # 全部
  python TFAC_V5/visualize_cqf_results.py --mode all [同上参数]
"""

import argparse
import json
import os
import sys
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)


# ============================================================
# 1. CQF 训练曲线
# ============================================================
def plot_training_curves(output_dir):
    """从 epoch checkpoint 提取指标, 画 Phase A→B→C 训练曲线。"""

    phase_dirs = {
        'no_perturb': {
            'A': '/home/chenshuai/Project/output/cqf_latent_0407_phaseA',
            'B': '/home/chenshuai/Project/output/cqf_latent_0407_phaseB',
            'C': '/home/chenshuai/Project/output/cqf_latent_0407_phaseC',
        },
        'perturb': {
            'A': '/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseA',
            'B': '/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseB',
            'C': '/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC',
        },
    }

    phase_colors = {'A': '#2196F3', 'B': '#FF9800', 'C': '#4CAF50'}
    phase_labels = {'A': 'Phase A (GT only)', 'B': 'Phase B (GT→Foresight)',
                    'C': 'Phase C (Foresight only)'}

    def extract_metrics(base_dir):
        """Extract per-epoch metrics from checkpoints."""
        records = []
        for fn in sorted(os.listdir(base_dir)):
            if not fn.endswith('.pt'):
                continue
            if fn in ('cqf_latent_best.pt', 'cqf_latent_final.pt'):
                continue
            ckpt = torch.load(os.path.join(base_dir, fn),
                              map_location='cpu', weights_only=False)
            vm = ckpt.get('val_metrics', {})
            records.append({
                'epoch': ckpt.get('epoch', 0),
                'spread': vm.get('spread', vm.get('score_spread', 0)),
                'rank1': vm.get('rank1_acc', 0),
                'loss': vm.get('loss', 0),
                'accuracy': vm.get('accuracy', 0),
                'pos_mean': vm.get('pos_score_mean', 0),
                'neg_mean': vm.get('neg_score_mean', 0),
            })
        records.sort(key=lambda x: x['epoch'])
        return records

    for setting, dirs in phase_dirs.items():
        is_perturb = setting == 'perturb'
        title_suffix = '(with Perturbation Negatives)' if is_perturb else '(without Perturbation)'

        # Collect data across phases with global epoch counter
        all_epochs, all_spread, all_rank1, all_loss = [], [], [], []
        all_pos, all_neg = [], []
        phase_boundaries = []
        epoch_offset = 0

        for phase_name in ['A', 'B', 'C']:
            d = dirs[phase_name]
            records = extract_metrics(d)
            for r in records:
                all_epochs.append(epoch_offset + r['epoch'])
                all_spread.append(r['spread'])
                all_rank1.append(r['rank1'])
                all_loss.append(r['loss'])
                all_pos.append(r['pos_mean'])
                all_neg.append(r['neg_mean'])
            if records:
                phase_boundaries.append({
                    'start': epoch_offset,
                    'end': epoch_offset + records[-1]['epoch'],
                    'phase': phase_name,
                })
                epoch_offset += records[-1]['epoch']

        # === Figure 1: Spread + Rank1 ===
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f'CQF Training: Phase A → B → C {title_suffix}', fontsize=14)

        # Phase background shading
        for pb in phase_boundaries:
            color = phase_colors[pb['phase']]
            for ax in [ax1, ax2]:
                ax.axvspan(pb['start'], pb['end'], alpha=0.08, color=color)
                ax.axvline(pb['start'], color=color, ls='--', alpha=0.3, lw=1)

        ax1.plot(all_epochs, all_spread, 'o-', color='#1565C0', lw=2, ms=5)
        ax1.set_ylabel('Score Spread (pos - neg)')
        ax1.grid(True, alpha=0.3)
        for pb in phase_boundaries:
            ax1.text((pb['start'] + pb['end']) / 2, ax1.get_ylim()[1] * 0.95,
                     phase_labels[pb['phase']], ha='center', fontsize=9,
                     color=phase_colors[pb['phase']], fontweight='bold')

        ax2.plot(all_epochs, [r * 100 for r in all_rank1], 's-',
                 color='#C62828', lw=2, ms=5)
        ax2.set_ylabel('Rank-1 Accuracy (%)')
        ax2.set_xlabel('Epoch (cumulative)')
        ax2.set_ylim([min(95, min(r * 100 for r in all_rank1) - 2), 101])
        ax2.grid(True, alpha=0.3)

        path1 = os.path.join(output_dir, f'cqf_training_{setting}.png')
        fig.savefig(path1)
        plt.close(fig)
        print(f'  Saved: {path1}')

        # === Figure 2: Pos/Neg score means ===
        fig2, ax3 = plt.subplots(figsize=(10, 4))
        ax3.fill_between(all_epochs, all_pos, all_neg, alpha=0.2, color='#4CAF50')
        ax3.plot(all_epochs, all_pos, 'o-', color='#2E7D32', lw=2, ms=4, label='Positive mean')
        ax3.plot(all_epochs, all_neg, 's-', color='#C62828', lw=2, ms=4, label='Negative mean')
        ax3.axhline(0, color='gray', ls=':', alpha=0.5)
        ax3.set_ylabel('Raw Score (logit)')
        ax3.set_xlabel('Epoch (cumulative)')
        ax3.set_title(f'CQF Score Separation {title_suffix}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        for pb in phase_boundaries:
            ax3.axvspan(pb['start'], pb['end'], alpha=0.06,
                        color=phase_colors[pb['phase']])
            ax3.axvline(pb['start'], color=phase_colors[pb['phase']],
                        ls='--', alpha=0.3, lw=1)

        path2 = os.path.join(output_dir, f'cqf_score_separation_{setting}.png')
        fig2.savefig(path2)
        plt.close(fig2)
        print(f'  Saved: {path2}')

    # === Figure 3: Side-by-side comparison (no-perturb vs perturb) ===
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, (setting, dirs) in enumerate(phase_dirs.items()):
        ax = axes[idx]
        epoch_offset = 0
        for phase_name in ['A', 'B', 'C']:
            records = extract_metrics(dirs[phase_name])
            epochs = [epoch_offset + r['epoch'] for r in records]
            spreads = [r['spread'] for r in records]
            ax.plot(epochs, spreads, 'o-', color=phase_colors[phase_name],
                    lw=2, ms=5, label=phase_labels[phase_name])
            if records:
                epoch_offset += records[-1]['epoch']

        ax.set_title('Without Perturbation' if setting == 'no_perturb'
                     else 'With Perturbation Negatives')
        ax.set_xlabel('Epoch (cumulative)')
        ax.set_ylabel('Score Spread')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig3.suptitle('CQF Training: Perturbation Comparison', fontsize=14)
    fig3.tight_layout()
    path3 = os.path.join(output_dir, 'cqf_perturbation_comparison.png')
    fig3.savefig(path3)
    plt.close(fig3)
    print(f'  Saved: {path3}')


# ============================================================
# 2 & 3. Reranking 分数分布 + 逐帧排名
# ============================================================
def run_reranking_and_visualize(args, output_dir):
    """Run reranking eval, collect per-frame data, generate visualizations."""
    from dp_reranking import TacDreamReranker
    import h5py
    from torchvision import transforms
    from tqdm import tqdm

    reranker = TacDreamReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        cqf_ckpt_path=args.cqf_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )

    data_dir = args.data_dir
    K = args.K
    rng = np.random.RandomState(args.seed)
    noise_scales = [0.2, 0.5, 1.0, 1.5, 2.0]
    h, cs = 10, 20

    ann_path = os.path.join(data_dir, "annotations.pkl")
    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    hdf5_files = []
    for subdir in ["success", "bounce", ""]:
        pattern_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.isdir(pattern_dir):
            for fn in os.listdir(pattern_dir):
                if fn.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(pattern_dir, fn))
    hdf5_files = sorted(set(hdf5_files))

    insertion_frames = []
    for hf in hdf5_files:
        ep_name = os.path.splitext(os.path.basename(hf))[0]
        if ep_name not in ann or ep_name == "_meta":
            continue
        labels = ann[ep_name]["labels"]
        T = len(labels)
        for t in np.where(labels == 1)[0]:
            if t + cs <= T and t + h < T:
                insertion_frames.append((hf, ep_name, t))

    n_eval = min(args.n_eval, len(insertion_frames))
    if len(insertion_frames) > n_eval:
        idx = rng.choice(len(insertion_frames), n_eval, replace=False)
        insertion_frames = [insertion_frames[i] for i in idx]

    print(f"\nRunning reranking on {len(insertion_frames)} frames (K={K})...")

    # Collect per-frame data
    all_frame_data = []

    for hdf5_path, ep_name, t in tqdm(insertion_frames, desc="Scoring"):
        try:
            with h5py.File(hdf5_path, "r") as f:
                qpos_raw = f["observations/proprio_joint"][t].astype(np.float32)
                eef_raw = f["observations/proprio_eef"][t].astype(np.float32)
                marker_all = f["observations/tac/left/marker_offset"][:]
                action_expert = f["actions/joint_abs"][t:t+cs].astype(np.float32)
                if len(action_expert) < cs:
                    action_expert = np.pad(action_expert,
                        ((0, cs - len(action_expert)), (0, 0)), mode='edge')

                has_global = "observations/images/global" in f
                has_wrist = "observations/images/wrist" in f
                img_global = reranker._preprocess_image(
                    f["observations/images/global"][t]).unsqueeze(0).to(reranker.device) \
                    if has_global else torch.zeros(1, 3, 480, 640, device=reranker.device)
                img_wrist = reranker._preprocess_image(
                    f["observations/images/wrist"][t]).unsqueeze(0).to(reranker.device) \
                    if has_wrist else torch.zeros(1, 3, 480, 640, device=reranker.device)

            marker_window = reranker._get_marker_window(marker_all, t)
            marker_window_dev = marker_window.unsqueeze(0).to(reranker.device)
            foresight_images = [img_global, img_wrist, marker_window_dev]

            expert_t = torch.tensor(action_expert, dtype=torch.float32,
                                    device=reranker.device)
            action_std = expert_t.std()
            candidates = [expert_t.clone()]
            candidate_scales = [0.0]
            for i in range(K - 1):
                scale = noise_scales[i % len(noise_scales)]
                noise = torch.randn_like(expert_t) * action_std * scale
                candidates.append(expert_t + noise)
                candidate_scales.append(scale)

            actions = torch.stack(candidates)
            scores, _ = reranker.score_candidates(
                actions, qpos_raw, eef_raw, marker_window, foresight_images)
            scores_np = scores.cpu().numpy()

            rank_order = np.argsort(-scores_np)
            expert_rank = int(np.where(rank_order == 0)[0][0])

            all_frame_data.append({
                'ep_name': ep_name, 't': t,
                'scores': scores_np,
                'expert_idx': 0,
                'expert_rank': expert_rank,
                'candidate_scales': candidate_scales,
            })
        except Exception as e:
            print(f"  Error {ep_name} t={t}: {e}")
            continue

    if not all_frame_data:
        print("No valid frames!")
        return

    # ============================================================
    # Viz 2: Score distribution
    # ============================================================
    expert_scores = [d['scores'][0] for d in all_frame_data]
    noisy_scores = [s for d in all_frame_data for s in d['scores'][1:]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 2a: Histogram
    ax = axes[0]
    bins = np.linspace(0, 1, 40)
    ax.hist(expert_scores, bins=bins, alpha=0.7, color='#2E7D32',
            label=f'Expert (n={len(expert_scores)})', edgecolor='white', lw=0.5)
    ax.hist(noisy_scores, bins=bins, alpha=0.5, color='#C62828',
            label=f'Noisy (n={len(noisy_scores)})', edgecolor='white', lw=0.5)
    ax.axvline(np.mean(expert_scores), color='#2E7D32', ls='--', lw=2,
               label=f'Expert mean={np.mean(expert_scores):.3f}')
    ax.axvline(np.mean(noisy_scores), color='#C62828', ls='--', lw=2,
               label=f'Noisy mean={np.mean(noisy_scores):.3f}')
    ax.set_xlabel('CQF Score (sigmoid)')
    ax.set_ylabel('Count')
    ax.set_title('Score Distribution: Expert vs Noisy Candidates')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2b: Boxplot by noise scale
    ax = axes[1]
    scale_groups = {}
    for d in all_frame_data:
        for i, scale in enumerate(d['candidate_scales']):
            key = f'{scale:.1f}' if scale > 0 else 'Expert'
            if key not in scale_groups:
                scale_groups[key] = []
            scale_groups[key].append(d['scores'][i])

    box_labels = ['Expert'] + sorted([k for k in scale_groups if k != 'Expert'],
                                     key=float)
    box_data = [scale_groups[k] for k in box_labels]
    box_colors = ['#2E7D32'] + ['#EF5350'] * (len(box_labels) - 1)

    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel('Noise Scale (0 = Expert)')
    ax.set_ylabel('CQF Score (sigmoid)')
    ax.set_title('CQF Score by Action Perturbation Level')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'TacDream Reranking: Score Distribution (K={K}, {len(all_frame_data)} frames)',
                 fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, 'reranking_score_distribution.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved: {path}')

    # ============================================================
    # Viz 3: Per-frame K=16 score ranking
    # ============================================================
    n_show = min(50, len(all_frame_data))

    fig, ax = plt.subplots(figsize=(14, 7))

    for frame_i in range(n_show):
        d = all_frame_data[frame_i]
        scores_sorted_idx = np.argsort(-d['scores'])
        scores_sorted = d['scores'][scores_sorted_idx]

        for rank, (idx, score) in enumerate(zip(scores_sorted_idx, scores_sorted)):
            if idx == 0:
                ax.scatter(frame_i, score, c='#C62828', s=40, zorder=5,
                           marker='*', edgecolors='black', linewidths=0.5)
            else:
                ax.scatter(frame_i, score, c='#90CAF9', s=12, alpha=0.5, zorder=3)

    expert_patch = mpatches.Patch(color='#C62828', label='Expert action')
    noisy_patch = mpatches.Patch(color='#90CAF9', label='Noisy candidates')
    ax.legend(handles=[expert_patch, noisy_patch], loc='lower right', fontsize=10)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('CQF Score (sigmoid)')
    ax.set_title(f'Per-Frame CQF Scores: Expert vs K={K} Candidates '
                 f'(first {n_show} frames)')
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'reranking_per_frame_scores.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved: {path}')

    # === Viz 3b: Expert rank histogram ===
    expert_ranks = [d['expert_rank'] for d in all_frame_data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(expert_ranks, bins=range(K + 1), color='#1565C0', edgecolor='white',
            alpha=0.8, align='left')
    ax.set_xlabel('Expert Rank (0 = best)')
    ax.set_ylabel('Count')
    ax.set_title(f'Expert Action Ranking Distribution (K={K}, '
                 f'{sum(1 for r in expert_ranks if r == 0)}/{len(expert_ranks)} rank-1)')
    ax.set_xticks(range(K))
    ax.grid(True, alpha=0.3, axis='y')

    path = os.path.join(output_dir, 'reranking_expert_rank_hist.png')
    fig.savefig(path)
    plt.close(fig)
    print(f'  Saved: {path}')

    # Print summary
    print(f"\n  Expert rank-1: {sum(1 for r in expert_ranks if r == 0)}/{len(expert_ranks)} "
          f"({sum(1 for r in expert_ranks if r == 0)/len(expert_ranks)*100:.1f}%)")
    print(f"  Expert score: {np.mean(expert_scores):.4f} ± {np.std(expert_scores):.4f}")
    print(f"  Noisy score:  {np.mean(noisy_scores):.4f} ± {np.std(noisy_scores):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all',
                        choices=['training', 'reranking', 'all'])
    parser.add_argument('--output_dir', type=str,
                        default='/home/chenshuai/Project/output/cqf_visualizations')

    # Reranking args (only needed for mode=reranking/all)
    parser.add_argument('--dp_ckpt', type=str,
                        default='/home/chenshuai/Project/output/dp_joint7/dp_best.pth')
    parser.add_argument('--dp_config', type=str,
                        default='/home/chenshuai/Project/output/dp_joint7/config.json')
    parser.add_argument('--cqf_ckpt', type=str,
                        default='/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt')
    parser.add_argument('--foresight_ckpt', type=str,
                        default='/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw/foresight_best.ckpt')
    parser.add_argument('--foresight_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str,
                        default='/home/chenshuai/data/dataset/260407')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--K', type=int, default=16)
    parser.add_argument('--n_eval', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    if args.mode in ('training', 'all'):
        print("\n=== 1. CQF Training Curves ===")
        plot_training_curves(args.output_dir)

    if args.mode in ('reranking', 'all'):
        print("\n=== 2 & 3. Reranking Visualizations ===")
        run_reranking_and_visualize(args, args.output_dir)

    print(f"\nDone! All figures saved to {args.output_dir}")


if __name__ == '__main__':
    main()
