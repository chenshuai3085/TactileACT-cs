"""
CQF Offline Evaluation Script.

Three metrics:
  1. Ranking Accuracy: expert action should rank #1 among K=16 candidates
  2. Bounce Detection: CQF scores insertion > pre_bounce for same episode
  3. Score Distribution: analyze pos/neg score separation

Usage:
  python TFAC_V5/eval_cqf.py --checkpoint /path/to/cqf_best.pt
"""

import argparse
import json
import os
import pickle
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from cqf_model import ContactQualityScorer


DATA_ROOT = "/home/chenshuai/data/dataset"

SPLIT_DATASETS = ["260309", "260310", "260401", "260402", "260403", "260407"]
MERGED_DATASETS = [
    "0414", "260401_0402", "260408_0409", "260417",
    "0209-0210", "0331", "260309_0310",
]

FORESIGHT_HORIZON = 10
CHUNK_SIZE = 20


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    args = ckpt.get("args", {})
    model = ContactQualityScorer(
        tac_dim=args.get("tac_dim", 162),
        hidden=args.get("hidden", 256),
        action_dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    if "val_metrics" in ckpt:
        vm = ckpt["val_metrics"]
        print(f"  Val metrics: loss={vm.get('loss', 0):.4f} "
              f"acc={vm.get('accuracy', 0):.3f} "
              f"spread={vm.get('global_spread', 0):.3f}")
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
    """Load all annotated episodes with their HDF5 data paths."""
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
def eval_ranking_accuracy(model, episodes, device, K=16, n_samples=500, seed=42):
    """
    Ranking Accuracy: for each insertion frame, create K-1 perturbed actions
    and check if the expert action ranks #1.
    """
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    insertion_samples = []
    for ep in episodes:
        labels = ep["labels"]
        pos_frames = np.where(labels == 1)[0]
        T = len(labels)
        valid_frames = [t for t in pos_frames if t + cs <= T and t + h < T]
        for t in valid_frames:
            insertion_samples.append((ep, t))

    if len(insertion_samples) > n_samples:
        idx = rng.choice(len(insertion_samples), n_samples, replace=False)
        insertion_samples = [insertion_samples[i] for i in idx]

    print(f"\nRanking Accuracy Evaluation ({len(insertion_samples)} samples, K={K})")

    rank1_count = 0
    top3_count = 0
    total = 0
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
        marker_future = marker_all[t+h]

        action_std = np.std(action_all, axis=0).mean()

        candidates = [expert_action]
        for i in range(K - 1):
            scale = noise_scales[i % len(noise_scales)]
            noise = rng.randn(*expert_action.shape).astype(np.float32) * action_std * scale
            candidates.append(expert_action + noise)

        batch_qpos = torch.from_numpy(np.tile(qpos, (K, 1))).to(device)
        batch_eef = torch.from_numpy(np.tile(eef, (K, 1))).to(device)
        batch_action = torch.from_numpy(np.stack(candidates)).to(device)
        batch_marker_cur = torch.from_numpy(
            np.tile(marker_cur.flatten(), (K, 1))).to(device)
        batch_marker_future = torch.from_numpy(
            np.tile(marker_future.flatten(), (K, 1))).to(device)

        scores = model(batch_qpos, batch_eef, batch_action,
                       batch_marker_cur, batch_marker_future)
        scores = scores.squeeze(-1).cpu().numpy()

        rank = (scores > scores[0]).sum()
        if rank == 0:
            rank1_count += 1
        if rank < 3:
            top3_count += 1
        total += 1

    rank1_acc = rank1_count / max(total, 1)
    top3_acc = top3_count / max(total, 1)
    random_baseline = 1.0 / K

    print(f"  Rank-1 Accuracy: {rank1_acc:.3f} ({rank1_count}/{total})")
    print(f"  Top-3 Accuracy:  {top3_acc:.3f} ({top3_count}/{total})")
    print(f"  Random Baseline: {random_baseline:.3f}")

    return {"rank1_acc": rank1_acc, "top3_acc": top3_acc,
            "random_baseline": random_baseline, "n_samples": total}


@torch.no_grad()
def eval_bounce_detection(model, episodes, device, seed=42):
    """
    Bounce Detection: for bounce episodes, compare CQF scores of
    insertion frames vs pre_bounce frames.
    """
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    bounce_eps = [ep for ep in episodes if ep["type"] == "bounce"]
    print(f"\nBounce Detection Evaluation ({len(bounce_eps)} bounce episodes)")

    correct = 0
    total = 0
    all_pos_scores = []
    all_neg_scores = []

    for ep in bounce_eps:
        labels = ep["labels"]
        lifts = ep["lifts"]
        T = len(labels)

        pos_frames = np.where(labels == 1)[0]
        neg_frames = np.where(labels == 2)[0]

        valid_pos = [t for t in pos_frames if t + cs <= T and t + h < T]
        lift_starts = [s for s, e, r in lifts]
        valid_neg = [t for t in neg_frames
                     if t + cs <= T and t + h < T
                     and any(t + h >= ls for ls in lift_starts)]

        if len(valid_pos) == 0 or len(valid_neg) == 0:
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
        all_labels_local = np.array([1.0] * n_pos + [0.0] * n_neg)

        batch_qpos = torch.from_numpy(qpos_all[all_t]).to(device)
        batch_eef = torch.from_numpy(eef_all[all_t]).to(device)
        batch_action = torch.from_numpy(
            np.stack([action_all[t:t+cs] for t in all_t])).to(device)
        batch_marker_cur = torch.from_numpy(
            marker_all[all_t].reshape(len(all_t), -1)).to(device)
        batch_marker_future = torch.from_numpy(
            np.stack([marker_all[t+h] for t in all_t]).reshape(len(all_t), -1)).to(device)

        scores = model(batch_qpos, batch_eef, batch_action,
                       batch_marker_cur, batch_marker_future)
        scores = scores.squeeze(-1).cpu().numpy()

        pos_scores_ep = scores[:n_pos]
        neg_scores_ep = scores[n_pos:]

        all_pos_scores.extend(pos_scores_ep.tolist())
        all_neg_scores.extend(neg_scores_ep.tolist())

        if pos_scores_ep.mean() > neg_scores_ep.mean():
            correct += 1
        total += 1

    det_acc = correct / max(total, 1)
    pos_mean = np.mean(all_pos_scores) if all_pos_scores else 0
    neg_mean = np.mean(all_neg_scores) if all_neg_scores else 0

    print(f"  Detection Accuracy: {det_acc:.3f} ({correct}/{total} episodes)")
    print(f"  Avg Pos Score: {pos_mean:.3f}")
    print(f"  Avg Neg Score: {neg_mean:.3f}")
    print(f"  Score Gap:     {pos_mean - neg_mean:.3f}")

    return {"detection_acc": det_acc, "pos_mean": pos_mean, "neg_mean": neg_mean,
            "score_gap": pos_mean - neg_mean, "n_episodes": total}


@torch.no_grad()
def eval_score_distribution(model, episodes, device, n_samples=2000, seed=42):
    """
    Score Distribution Analysis: check pos/neg score separation on held-out data.
    """
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    all_samples = []
    for ep in episodes:
        labels = ep["labels"]
        lifts = ep["lifts"]
        T = len(labels)
        lift_starts = [s for s, e, r in lifts]

        pos_frames = np.where(labels == 1)[0]
        for t in pos_frames:
            if t + cs <= T and t + h < T:
                all_samples.append((ep, t, 1.0))

        neg_frames = np.where(labels == 2)[0]
        for t in neg_frames:
            if t + cs <= T and t + h < T and any(t + h >= ls for ls in lift_starts):
                all_samples.append((ep, t, 0.0))

    if len(all_samples) > n_samples:
        idx = rng.choice(len(all_samples), n_samples, replace=False)
        all_samples = [all_samples[i] for i in idx]

    print(f"\nScore Distribution Analysis ({len(all_samples)} samples)")

    scores_list = []
    labels_list = []

    batch_size = 256
    for start in range(0, len(all_samples), batch_size):
        batch = all_samples[start:start+batch_size]
        qpos_list, eef_list, action_list = [], [], []
        mcur_list, mfut_list, lbl_list = [], [], []

        for ep, t, label in batch:
            try:
                with h5py.File(ep["hdf5_path"], "r") as f:
                    qpos_list.append(f["observations/proprio_joint"][t].astype(np.float32))
                    eef_list.append(f["observations/proprio_eef"][t].astype(np.float32))
                    action_list.append(f["actions/joint_abs"][t:t+cs].astype(np.float32))
                    mcur_list.append(f["observations/tac/left/marker_offset"][t].flatten().astype(np.float32))
                    mfut_list.append(f["observations/tac/left/marker_offset"][t+h].flatten().astype(np.float32))
                    lbl_list.append(label)
            except Exception:
                continue

        if not qpos_list:
            continue

        batch_qpos = torch.from_numpy(np.stack(qpos_list)).to(device)
        batch_eef = torch.from_numpy(np.stack(eef_list)).to(device)
        batch_action = torch.from_numpy(np.stack(action_list)).to(device)
        batch_mcur = torch.from_numpy(np.stack(mcur_list)).to(device)
        batch_mfut = torch.from_numpy(np.stack(mfut_list)).to(device)

        scores = model(batch_qpos, batch_eef, batch_action,
                       batch_mcur, batch_mfut)
        scores_list.extend(scores.squeeze(-1).cpu().numpy().tolist())
        labels_list.extend(lbl_list)

    scores_arr = np.array(scores_list)
    labels_arr = np.array(labels_list)

    pos_scores = scores_arr[labels_arr > 0.5]
    neg_scores = scores_arr[labels_arr <= 0.5]

    print(f"  Positive samples: {len(pos_scores)}")
    print(f"    Mean: {pos_scores.mean():.3f}")
    print(f"    Std:  {pos_scores.std():.3f}")
    print(f"    Min:  {pos_scores.min():.3f}")
    print(f"    Max:  {pos_scores.max():.3f}")
    print(f"  Negative samples: {len(neg_scores)}")
    print(f"    Mean: {neg_scores.mean():.3f}")
    print(f"    Std:  {neg_scores.std():.3f}")
    print(f"    Min:  {neg_scores.min():.3f}")
    print(f"    Max:  {neg_scores.max():.3f}")

    spread = pos_scores.mean() - neg_scores.mean()
    overlap = ((neg_scores > pos_scores.mean() - pos_scores.std()).sum() +
               (pos_scores < neg_scores.mean() + neg_scores.std()).sum())
    overlap_rate = overlap / (len(pos_scores) + len(neg_scores))

    # AUC-like: fraction of (pos, neg) pairs where pos > neg
    n_pairs = min(len(pos_scores), 5000) * min(len(neg_scores), 5000)
    if n_pairs > 0:
        p_sample = pos_scores[:min(len(pos_scores), 5000)]
        n_sample = neg_scores[:min(len(neg_scores), 5000)]
        auc = (p_sample[:, None] > n_sample[None, :]).mean()
    else:
        auc = 0.0

    print(f"  ---")
    print(f"  Spread (pos_mean - neg_mean): {spread:.3f}")
    print(f"  Overlap rate: {overlap_rate:.3f}")
    print(f"  Pairwise AUC: {auc:.4f}")

    # Action dropout ablation: check how much action contributes
    return {"spread": spread, "overlap_rate": overlap_rate, "auc": auc,
            "pos_mean": pos_scores.mean(), "neg_mean": neg_scores.mean(),
            "pos_std": pos_scores.std(), "neg_std": neg_scores.std()}


@torch.no_grad()
def eval_action_dependency(model, episodes, device, n_samples=500, seed=42):
    """
    Action Dependency Test: check if CQF relies too much on action vs tactile.
    Replace action with random noise and see how much scores change.
    """
    rng = np.random.RandomState(seed)
    h = FORESIGHT_HORIZON
    cs = CHUNK_SIZE

    samples = []
    for ep in episodes:
        labels = ep["labels"]
        T = len(labels)
        pos_frames = np.where(labels == 1)[0]
        for t in pos_frames:
            if t + cs <= T and t + h < T:
                samples.append((ep, t))

    if len(samples) > n_samples:
        idx = rng.choice(len(samples), n_samples, replace=False)
        samples = [samples[i] for i in idx]

    print(f"\nAction Dependency Test ({len(samples)} samples)")

    score_original = []
    score_random_action = []
    score_zero_tactile_delta = []

    for ep, t in samples:
        try:
            with h5py.File(ep["hdf5_path"], "r") as f:
                qpos = f["observations/proprio_joint"][t].astype(np.float32)
                eef = f["observations/proprio_eef"][t].astype(np.float32)
                action = f["actions/joint_abs"][t:t+cs].astype(np.float32)
                marker_cur = f["observations/tac/left/marker_offset"][t].flatten().astype(np.float32)
                marker_future = f["observations/tac/left/marker_offset"][t+h].flatten().astype(np.float32)
        except Exception:
            continue

        qpos_t = torch.from_numpy(qpos).unsqueeze(0).to(device)
        eef_t = torch.from_numpy(eef).unsqueeze(0).to(device)
        action_t = torch.from_numpy(action).unsqueeze(0).to(device)
        mcur_t = torch.from_numpy(marker_cur).unsqueeze(0).to(device)
        mfut_t = torch.from_numpy(marker_future).unsqueeze(0).to(device)

        s_orig = model(qpos_t, eef_t, action_t, mcur_t, mfut_t).item()
        score_original.append(s_orig)

        rand_action = torch.randn_like(action_t)
        s_rand = model(qpos_t, eef_t, rand_action, mcur_t, mfut_t).item()
        score_random_action.append(s_rand)

        s_zero_delta = model(qpos_t, eef_t, action_t, mcur_t, mcur_t).item()
        score_zero_tactile_delta.append(s_zero_delta)

    orig = np.array(score_original)
    rand_act = np.array(score_random_action)
    zero_delta = np.array(score_zero_tactile_delta)

    act_sensitivity = np.abs(orig - rand_act).mean() / (np.abs(orig).mean() + 1e-8)
    tac_sensitivity = np.abs(orig - zero_delta).mean() / (np.abs(orig).mean() + 1e-8)

    print(f"  Original score mean:      {orig.mean():.3f}")
    print(f"  Random action score mean: {rand_act.mean():.3f}")
    print(f"  Zero tactile delta mean:  {zero_delta.mean():.3f}")
    print(f"  ---")
    print(f"  Action sensitivity:  {act_sensitivity:.3f} "
          f"(score change when action randomized)")
    print(f"  Tactile sensitivity: {tac_sensitivity:.3f} "
          f"(score change when tactile delta zeroed)")
    print(f"  Ratio (tac/act):     {tac_sensitivity / (act_sensitivity + 1e-8):.2f}x "
          f"(>1 means model relies more on tactile)")

    return {"act_sensitivity": act_sensitivity, "tac_sensitivity": tac_sensitivity,
            "ratio_tac_act": tac_sensitivity / (act_sensitivity + 1e-8),
            "orig_mean": orig.mean(), "rand_act_mean": rand_act.mean(),
            "zero_delta_mean": zero_delta.mean()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_ranking", type=int, default=500)
    parser.add_argument("--n_dist", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, device)

    print("\nLoading episodes...")
    t0 = time.time()
    episodes = load_all_episodes()
    print(f"Loaded {len(episodes)} episodes in {time.time()-t0:.1f}s")

    n_bounce = sum(1 for ep in episodes if ep["type"] == "bounce")
    n_success = sum(1 for ep in episodes if ep["type"] == "success")
    print(f"  Success: {n_success}, Bounce: {n_bounce}")

    # Use 20% as eval set (same split as training)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(episodes))
    n_val = int(len(episodes) * 0.2)
    val_episodes = [episodes[i] for i in indices[:n_val]]
    print(f"Using {len(val_episodes)} validation episodes")

    results = {}

    results["ranking"] = eval_ranking_accuracy(
        model, val_episodes, device, K=args.K, n_samples=args.n_ranking)

    results["bounce_detection"] = eval_bounce_detection(
        model, val_episodes, device)

    results["score_distribution"] = eval_score_distribution(
        model, val_episodes, device, n_samples=args.n_dist)

    results["action_dependency"] = eval_action_dependency(
        model, val_episodes, device)

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    r = results["ranking"]
    print(f"  Ranking Accuracy (K={args.K}): {r['rank1_acc']:.3f} "
          f"(baseline {r['random_baseline']:.3f})")
    b = results["bounce_detection"]
    print(f"  Bounce Detection:  {b['detection_acc']:.3f} "
          f"(gap={b['score_gap']:.3f})")
    s = results["score_distribution"]
    print(f"  Score Spread:      {s['spread']:.3f}")
    print(f"  Pairwise AUC:      {s['auc']:.4f}")
    a = results["action_dependency"]
    print(f"  Tac/Act Ratio:     {a['ratio_tac_act']:.2f}x")

    # Save results
    out_dir = os.path.dirname(args.checkpoint)
    save_path = os.path.join(out_dir, "eval_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
