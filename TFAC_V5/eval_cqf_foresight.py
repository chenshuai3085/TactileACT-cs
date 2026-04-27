"""
CQF + Foresight End-to-End Evaluation.

This evaluates the full pipeline: Foresight predicts future tactile
for each candidate action, then CQF scores them.

Key difference from eval_cqf.py: marker_future is PREDICTED, not GT.
This is the actual inference behavior.

Tests:
  1. Ranking Accuracy with Foresight: K candidates, each with
     Foresight-predicted tactile → CQF score → rank expert #1?
  2. Bounce Detection with Foresight: insertion vs pre_bounce scoring
  3. Score distribution analysis

Usage:
  python TFAC_V5/eval_cqf_foresight.py \
    --cqf_ckpt /path/to/cqf_best.pt \
    --foresight_ckpt /path/to/foresight_best.pt
"""

import argparse
import os
import pickle
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from cqf_model import ContactQualityScorer
from lightweight_foresight import LightweightForesight


DATA_ROOT = "/home/chenshuai/data/dataset"
SPLIT_DATASETS = ["260309", "260310", "260402", "260403", "260407"]
MERGED_DATASETS = ["0414", "260401_0402", "260408_0409", "260417",
                   "0209-0210", "0331", "260309_0310"]
FORESIGHT_HORIZON = 10
CHUNK_SIZE = 20


def load_cqf(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ContactQualityScorer(
        tac_dim=args.get("tac_dim", 162),
        hidden=args.get("hidden", 256),
        action_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"CQF: epoch={ckpt.get('epoch', '?')}, "
          f"phase={ckpt.get('phase', 'A')}, "
          f"spread={ckpt.get('val_metrics', {}).get('spread', '?')}")
    return model


def load_foresight(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = LightweightForesight(
        hidden=args.get("hidden", 512),
        n_layers=args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Foresight: epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('val_metrics', {}).get('loss', '?'):.4f}")
    return model


def load_foresight_v2(ckpt_path, device):
    """Load FiLM-conditioned foresight V2."""
    from lightweight_foresight_v2 import LightweightForesightV2
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = LightweightForesightV2(
        hidden=args.get("hidden", 512),
        n_layers=args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Foresight V2: epoch={ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('val_metrics', {}).get('loss', '?'):.4f}")
    return model


def find_hdf5(ds_dir, ep_name):
    fname = f"{ep_name}.hdf5"
    path = os.path.join(ds_dir, fname)
    if os.path.exists(path):
        return path
    for subdir in ["success", "bounce"]:
        path = os.path.join(ds_dir, subdir, fname)
        if os.path.exists(path):
            return path
    return None


def load_all_episodes(dataset_names=None, data_root=DATA_ROOT):
    if dataset_names is None:
        dataset_names = SPLIT_DATASETS + MERGED_DATASETS
    episodes = []
    for ds_name in dataset_names:
        ds_dir = os.path.join(data_root, ds_name)
        ann_path = os.path.join(ds_dir, "annotations.pkl")
        if not os.path.exists(ann_path):
            continue
        with open(ann_path, "rb") as f:
            ann = pickle.load(f)
        for ep_name, ep_ann in ann.items():
            if ep_name == "_meta":
                continue
            hdf5_path = find_hdf5(ds_dir, ep_name)
            if hdf5_path is None:
                continue
            episodes.append({
                "ds_name": ds_name,
                "ep_name": ep_name,
                "hdf5_path": hdf5_path,
                "labels": ep_ann["labels"],
                "lifts": ep_ann["lifts"],
                "type": ep_ann.get("type", "unknown"),
            })
    return episodes


@torch.no_grad()
def eval_ranking_with_foresight(cqf, foresight, episodes, device,
                                 K=16, n_samples=500, seed=42):
    """
    True ranking test: for each insertion frame, create K action candidates,
    run Foresight on each to predict tactile, then CQF scores them.
    Expert action should rank #1.
    """
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    insertion_samples = []
    for ep in episodes:
        labels = ep["labels"]
        pos_frames = np.where(labels == 1)[0]
        T = len(labels)
        valid = [t for t in pos_frames if t + cs <= T and t + h < T]
        for t in valid:
            insertion_samples.append((ep, t))

    if len(insertion_samples) > n_samples:
        idx = rng.choice(len(insertion_samples), n_samples, replace=False)
        insertion_samples = [insertion_samples[i] for i in idx]

    print(f"\n{'='*60}")
    print(f"Ranking with Foresight ({len(insertion_samples)} samples, K={K})")
    print(f"{'='*60}")

    rank1_count = 0
    top3_count = 0
    total = 0
    score_gaps = []
    noise_scales = [0.3, 0.5, 1.0, 1.5, 2.0]

    for ep, t in insertion_samples:
        try:
            with h5py.File(ep["hdf5_path"], "r") as f:
                qpos = f["observations/proprio_joint"][t].astype(np.float32)
                eef = f["observations/proprio_eef"][t].astype(np.float32)
                action_all = f["actions/joint_abs"][:].astype(np.float32)
                marker_all = f["observations/tac/left/marker_offset"][:].astype(np.float32)
        except Exception:
            continue

        T = action_all.shape[0]
        if t + cs > T or t + h >= T:
            continue

        expert_action = action_all[t:t+cs]
        marker_cur = marker_all[t]
        action_std = np.std(action_all, axis=0).mean()

        candidates = [expert_action]
        for i in range(K - 1):
            scale = noise_scales[i % len(noise_scales)]
            noise = rng.randn(*expert_action.shape).astype(np.float32) * action_std * scale
            candidates.append(expert_action + noise)

        batch_qpos = torch.from_numpy(np.tile(qpos, (K, 1))).to(device)
        batch_eef = torch.from_numpy(np.tile(eef, (K, 1))).to(device)
        batch_action = torch.from_numpy(np.stack(candidates)).to(device)
        batch_mcur = torch.from_numpy(
            np.tile(marker_cur.flatten(), (K, 1))).to(device)

        batch_mfut = foresight(batch_qpos, batch_eef, batch_action, batch_mcur)

        scores = cqf(batch_qpos, batch_eef, batch_action,
                      batch_mcur, batch_mfut)
        scores = scores.squeeze(-1).cpu().numpy()

        rank = (scores > scores[0]).sum()
        if rank == 0:
            rank1_count += 1
        if rank < 3:
            top3_count += 1
        score_gaps.append(scores[0] - scores[1:].max())
        total += 1

    rank1_acc = rank1_count / max(total, 1)
    top3_acc = top3_count / max(total, 1)
    random = 1.0 / K

    print(f"  Rank-1 Accuracy: {rank1_acc:.3f} ({rank1_count}/{total}) "
          f"[random={random:.3f}]")
    print(f"  Top-3 Accuracy:  {top3_acc:.3f} ({top3_count}/{total})")
    if score_gaps:
        gaps = np.array(score_gaps)
        print(f"  Score gap (expert - best_other): "
              f"mean={gaps.mean():.3f}±{gaps.std():.3f}, "
              f"median={np.median(gaps):.3f}")

    return {"rank1_acc": rank1_acc, "top3_acc": top3_acc,
            "random": random, "n_samples": total}


@torch.no_grad()
def eval_bounce_with_foresight(cqf, foresight, episodes, device, seed=42):
    """Bounce detection using foresight-predicted tactile."""
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    bounce_eps = [ep for ep in episodes if ep["type"] == "bounce"]
    print(f"\n{'='*60}")
    print(f"Bounce Detection with Foresight ({len(bounce_eps)} bounce eps)")
    print(f"{'='*60}")

    correct = 0
    total = 0
    all_pos, all_neg = [], []

    for ep in bounce_eps:
        labels = ep["labels"]
        T = len(labels)
        pos_frames = np.where(labels == 1)[0]
        neg_frames = np.where(labels == 2)[0]
        valid_pos = [t for t in pos_frames if t + cs <= T and t + h < T]
        valid_neg = [t for t in neg_frames if t + cs <= T and t + h < T]

        if not valid_pos or not valid_neg:
            continue

        try:
            with h5py.File(ep["hdf5_path"], "r") as f:
                qpos_all = f["observations/proprio_joint"][:].astype(np.float32)
                eef_all = f["observations/proprio_eef"][:].astype(np.float32)
                action_all = f["actions/joint_abs"][:].astype(np.float32)
                marker_all = f["observations/tac/left/marker_offset"][:].astype(np.float32)
        except Exception:
            continue

        n_pos = min(5, len(valid_pos))
        n_neg = min(5, len(valid_neg))
        pos_sample = rng.choice(valid_pos, n_pos, replace=False)
        neg_sample = rng.choice(valid_neg, n_neg, replace=False)

        all_t = np.concatenate([pos_sample, neg_sample])
        all_lab = np.array([1.0] * n_pos + [0.0] * n_neg)

        batch_q = torch.from_numpy(qpos_all[all_t]).to(device)
        batch_e = torch.from_numpy(eef_all[all_t]).to(device)
        batch_a = torch.from_numpy(
            np.stack([action_all[t:t+cs] for t in all_t])).to(device)
        batch_mc = torch.from_numpy(
            marker_all[all_t].reshape(len(all_t), -1)).to(device)

        batch_mf = foresight(batch_q, batch_e, batch_a, batch_mc)

        scores = cqf(batch_q, batch_e, batch_a, batch_mc, batch_mf)
        scores = scores.squeeze(-1).cpu().numpy()

        pos_scores = scores[:n_pos]
        neg_scores = scores[n_pos:]
        all_pos.extend(pos_scores.tolist())
        all_neg.extend(neg_scores.tolist())

        if pos_scores.mean() > neg_scores.mean():
            correct += 1
        total += 1

    acc = correct / max(total, 1)
    all_pos, all_neg = np.array(all_pos), np.array(all_neg)

    print(f"  Episode-level accuracy: {acc:.3f} ({correct}/{total})")
    if len(all_pos) > 0 and len(all_neg) > 0:
        spread = all_pos.mean() - all_neg.mean()
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.ones(len(all_pos)), np.zeros(len(all_neg))])
        preds = np.concatenate([all_pos, all_neg])
        auc = roc_auc_score(labels, preds)
        print(f"  Frame-level: pos_mean={all_pos.mean():.3f}, "
              f"neg_mean={all_neg.mean():.3f}, spread={spread:.3f}")
        print(f"  AUC: {auc:.4f}")
        return {"accuracy": acc, "spread": spread, "auc": auc}
    return {"accuracy": acc}


@torch.no_grad()
def eval_gt_vs_foresight_comparison(cqf, foresight, episodes, device, n_samples=300, seed=42):
    """Compare CQF scores when using GT vs Foresight tactile."""
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    samples = []
    for ep in episodes:
        labels = ep["labels"]
        T = len(labels)
        for t_label in [1, 2]:
            frames = np.where(labels == t_label)[0]
            valid = [t for t in frames if t + cs <= T and t + h < T]
            for t in valid[:3]:
                samples.append((ep, t, t_label))

    if len(samples) > n_samples:
        idx = rng.choice(len(samples), n_samples, replace=False)
        samples = [samples[i] for i in idx]

    print(f"\n{'='*60}")
    print(f"GT vs Foresight Comparison ({len(samples)} samples)")
    print(f"{'='*60}")

    gt_scores, fs_scores, labels = [], [], []
    for ep, t, lab in samples:
        try:
            with h5py.File(ep["hdf5_path"], "r") as f:
                qpos = torch.from_numpy(
                    f["observations/proprio_joint"][t:t+1].astype(np.float32)).to(device)
                eef = torch.from_numpy(
                    f["observations/proprio_eef"][t:t+1].astype(np.float32)).to(device)
                action = torch.from_numpy(
                    f["actions/joint_abs"][t:t+cs][None].astype(np.float32)).to(device)
                mc = torch.from_numpy(
                    f["observations/tac/left/marker_offset"][t].flatten()[None].astype(np.float32)).to(device)
                mf_gt = torch.from_numpy(
                    f["observations/tac/left/marker_offset"][t+h].flatten()[None].astype(np.float32)).to(device)
        except Exception:
            continue

        mf_pred = foresight(qpos, eef, action, mc)

        s_gt = cqf(qpos, eef, action, mc, mf_gt).item()
        s_fs = cqf(qpos, eef, action, mc, mf_pred).item()
        gt_scores.append(s_gt)
        fs_scores.append(s_fs)
        labels.append(lab)

    gt_scores = np.array(gt_scores)
    fs_scores = np.array(fs_scores)
    labels = np.array(labels)

    corr = np.corrcoef(gt_scores, fs_scores)[0, 1]
    diff = np.abs(gt_scores - fs_scores)
    print(f"  Correlation(GT_score, Foresight_score): {corr:.4f}")
    print(f"  Mean absolute score diff: {diff.mean():.3f}±{diff.std():.3f}")

    for lab, name in [(1, "insertion"), (2, "pre_bounce")]:
        mask = labels == lab
        if mask.any():
            print(f"  {name}: GT_score={gt_scores[mask].mean():.3f}, "
                  f"FS_score={fs_scores[mask].mean():.3f}, "
                  f"diff={diff[mask].mean():.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cqf_ckpt", type=str, required=True)
    parser.add_argument("--foresight_ckpt", type=str, required=True)
    parser.add_argument("--foresight_version", type=str, default="v1",
                        choices=["v1", "v2"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("CQF + Foresight End-to-End Evaluation")
    print("=" * 60)

    cqf = load_cqf(args.cqf_ckpt, device)
    if args.foresight_version == "v2":
        foresight = load_foresight_v2(args.foresight_ckpt, device)
    else:
        foresight = load_foresight(args.foresight_ckpt, device)

    print("\nLoading episodes...")
    episodes = load_all_episodes()
    print(f"Loaded {len(episodes)} episodes "
          f"({sum(1 for e in episodes if e['type']=='success')} success, "
          f"{sum(1 for e in episodes if e['type']=='bounce')} bounce)")

    eval_ranking_with_foresight(
        cqf, foresight, episodes, device, K=args.K, n_samples=args.n_samples)
    eval_bounce_with_foresight(cqf, foresight, episodes, device)
    eval_gt_vs_foresight_comparison(cqf, foresight, episodes, device)

    print(f"\n{'='*60}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
