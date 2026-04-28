"""
CQF 细粒度排序评估：检查同一帧下 expert > perturb_0.5 > perturb_1.0 > perturb_2.0 的排序能力。

按 qpos 分组找到同一帧的 4 个样本 (expert + 3 perturbation),
用 CQF 打分后计算:
  1. Spearman 相关: CQF score 排序 vs label 排序
  2. Pairwise accuracy: 所有 C(4,2)=6 对中, 高 label 的是否得分更高
  3. Expert rank-1: expert 是否在 4 个中得分最高
  4. Kendall tau: 排序一致性

Usage:
  python TFAC_V5/eval_cqf_fine_ranking.py \
    --cqf_ckpt /home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt \
    --data_dir /home/chenshuai/Project/output/cqf_latent_data_0407_perturb
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from cqf_model import ContactQualityScorer


def load_and_group(data_dir):
    """Load train+val samples, group by qpos to find same-frame sets."""
    train = torch.load(os.path.join(data_dir, "train_samples.pt"),
                       map_location='cpu', weights_only=False)
    val = torch.load(os.path.join(data_dir, "val_samples.pt"),
                     map_location='cpu', weights_only=False)
    all_samples = train + val
    print(f"Loaded {len(train)} train + {len(val)} val = {len(all_samples)} total")

    groups = {}
    for s in all_samples:
        key = tuple(np.round(s['qpos'], 4))
        groups.setdefault(key, []).append(s)

    # Keep groups with expert(1.0) + at least 2 perturbations
    valid_groups = []
    for key, samples in groups.items():
        labels = sorted(set(round(s['label'], 2) for s in samples))
        has_expert = any(abs(s['label'] - 1.0) < 0.01 for s in samples)
        n_perturb = sum(1 for s in samples if s.get('is_perturb'))
        if has_expert and n_perturb >= 2:
            valid_groups.append(samples)

    full_groups = [g for g in valid_groups if len(g) == 4]
    print(f"Total groups: {len(groups)}")
    print(f"Valid groups (expert + >=2 perturb): {len(valid_groups)}")
    print(f"Full groups (exactly 4): {len(full_groups)}")
    return valid_groups, full_groups


@torch.no_grad()
def evaluate(model, groups, device, use_gt=False):
    """Score each group and compute ranking metrics.

    Args:
        use_gt: if True, use z_future_gt instead of z_pred (upper bound test)
    """
    model.eval()

    spearman_list = []
    kendall_list = []
    pairwise_correct = 0
    pairwise_total = 0
    expert_rank1 = 0
    expert_total = 0
    score_by_label = {}

    for group in groups:
        group_sorted = sorted(group, key=lambda s: -s['label'])
        n = len(group_sorted)

        qpos = torch.stack([torch.from_numpy(s['qpos']) for s in group_sorted]).to(device)
        eef = torch.stack([torch.from_numpy(s['eef']) for s in group_sorted]).to(device)
        action = torch.stack([torch.from_numpy(s['action_chunk']) for s in group_sorted]).to(device)
        z_cur = torch.stack([torch.from_numpy(s['z_cur']) for s in group_sorted]).to(device)

        if use_gt:
            z_future = torch.stack([torch.from_numpy(s['z_future_gt']) for s in group_sorted]).to(device)
        else:
            z_future = torch.stack([torch.from_numpy(s['z_pred']) for s in group_sorted]).to(device)

        scores = model.predict(qpos, eef, action, z_cur, z_future).squeeze(-1).cpu().numpy()
        labels = np.array([s['label'] for s in group_sorted])

        # Record scores by label
        for i, s in enumerate(group_sorted):
            lbl = round(s['label'], 2)
            score_by_label.setdefault(lbl, []).append(scores[i])

        # Spearman correlation (labels already sorted descending)
        if len(set(labels)) > 1:
            rho, _ = stats.spearmanr(labels, scores)
            if not np.isnan(rho):
                spearman_list.append(rho)

            tau, _ = stats.kendalltau(labels, scores)
            if not np.isnan(tau):
                kendall_list.append(tau)

        # Pairwise accuracy
        for i in range(n):
            for j in range(i + 1, n):
                if abs(labels[i] - labels[j]) < 0.01:
                    continue
                pairwise_total += 1
                if (labels[i] > labels[j] and scores[i] > scores[j]) or \
                   (labels[i] < labels[j] and scores[i] < scores[j]):
                    pairwise_correct += 1

        # Expert rank-1
        expert_idx = np.argmax(labels)
        if np.argmax(scores) == expert_idx:
            expert_rank1 += 1
        expert_total += 1

    return {
        'spearman_mean': np.mean(spearman_list) if spearman_list else 0,
        'spearman_std': np.std(spearman_list) if spearman_list else 0,
        'kendall_mean': np.mean(kendall_list) if kendall_list else 0,
        'kendall_std': np.std(kendall_list) if kendall_list else 0,
        'pairwise_acc': pairwise_correct / max(pairwise_total, 1),
        'pairwise_n': pairwise_total,
        'expert_rank1': expert_rank1 / max(expert_total, 1),
        'expert_total': expert_total,
        'score_by_label': {k: (np.mean(v), np.std(v)) for k, v in sorted(score_by_label.items(), reverse=True)},
    }


def print_results(name, results):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Spearman rho:    {results['spearman_mean']:.4f} +/- {results['spearman_std']:.4f}")
    print(f"  Kendall tau:     {results['kendall_mean']:.4f} +/- {results['kendall_std']:.4f}")
    print(f"  Pairwise acc:    {results['pairwise_acc']:.1%} ({results['pairwise_n']} pairs)")
    print(f"  Expert rank-1:   {results['expert_rank1']:.1%} ({results['expert_total']} groups)")
    print(f"\n  CQF Score by label (sigmoid [0,1]):")
    print(f"  {'Label':>8s}  {'Mean':>8s}  {'Std':>8s}  {'Expected':>10s}")
    print(f"  {'-'*40}")
    for lbl, (mean, std) in results['score_by_label'].items():
        expected = "highest" if lbl == 1.0 else ("lowest" if lbl <= 0.01 else "")
        print(f"  {lbl:>8.2f}  {mean:>8.4f}  {std:>8.4f}  {expected:>10s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cqf_ckpt", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent_data_0407_perturb")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load CQF
    ckpt = torch.load(args.cqf_ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    model = ContactQualityScorer(
        tac_dim=ckpt_args.get("tac_dim", 144),
        hidden=ckpt_args.get("hidden", 256),
        action_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"CQF loaded (epoch {ckpt.get('epoch', '?')}, phase {ckpt.get('phase', '?')})")

    # Load and group data
    valid_groups, full_groups = load_and_group(args.data_dir)

    # Use full groups (4 samples) for clean evaluation
    groups = full_groups if len(full_groups) >= 50 else valid_groups
    print(f"\nUsing {len(groups)} groups for evaluation")

    # Eval with foresight predictions (real scenario)
    results_pred = evaluate(model, groups, device, use_gt=False)
    print_results("CQF Fine-Grained Ranking (z_pred, real scenario)", results_pred)

    # Eval with GT future (upper bound)
    results_gt = evaluate(model, groups, device, use_gt=True)
    print_results("CQF Fine-Grained Ranking (z_future_gt, upper bound)", results_gt)

    # Gap analysis
    print(f"\n{'='*60}")
    print(f"  Gap Analysis (GT upper bound vs Prediction)")
    print(f"{'='*60}")
    print(f"  Pairwise acc:  {results_gt['pairwise_acc']:.1%} (GT) vs {results_pred['pairwise_acc']:.1%} (Pred)")
    print(f"  Expert rank-1: {results_gt['expert_rank1']:.1%} (GT) vs {results_pred['expert_rank1']:.1%} (Pred)")
    print(f"  Spearman:      {results_gt['spearman_mean']:.4f} (GT) vs {results_pred['spearman_mean']:.4f} (Pred)")


if __name__ == "__main__":
    main()
