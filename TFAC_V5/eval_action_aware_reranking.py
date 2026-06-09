"""Offline reranking evaluation for the action-aware tactile scorer.

This is the first integration test toward DP guidance:
  candidate action -> Foresight predicted tactile -> action-aware scorer -> rank.

The default mode uses expert action plus noisy perturbations.  It is cheaper and
more controlled than full DP sampling, and tests whether the scorer can rank a
better action above nearby bad candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "TFAC_V5"))

from TFAC_V5.action_aware_scorer_runtime import ActionAwareScorerRuntime, WINDOW
from TFAC_V5.dp_reranking import (
    TacDreamReranker,
    _collect_insertion_frames,
    _find_hdf5_files,
    _load_frame_data,
)


OUT_DIR = Path("/home/chenshuai/Project/output/action_aware_reranking")


class ForesightOnlyReranker(TacDreamReranker):
    """Reuse TacDream DP/Foresight loading without requiring a CQF checkpoint."""

    def __init__(self, dp_config_path, dp_ckpt_path, foresight_ckpt_path, foresight_dir=None, device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K
        with open(dp_config_path) as f:
            self.dp_config = json.load(f)
        self.dp_variant = self.dp_config.get("variant", "clip_tactile_image")
        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)


def load_tactile_vae_norm(vae_ckpt):
    ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
    ns = ckpt.get("norm_stats", {})
    mean = torch.tensor(ns.get("mean", [0.2101736068725586, -0.6422404050827026]), dtype=torch.float32)
    std = torch.tensor(ns.get("std", [1.6805468797683716, 3.6716601848602295]), dtype=torch.float32)
    return mean, std


@torch.no_grad()
def score_with_action_aware(reranker, scorer, actions_raw, qpos_raw, marker_window, foresight_images, score_mode, vae_mean, vae_std):
    """Run Foresight then action-aware scorer."""
    k = actions_raw.shape[0]
    fs_images = [img.expand(k, *img.shape[1:]) for img in foresight_images]
    fs_chunk = reranker.foresight_chunk
    action_fs = actions_raw[:, :fs_chunk, :]
    action_fs_norm = reranker._fs_norm_action(action_fs)

    if isinstance(qpos_raw, np.ndarray):
        qpos_raw_t = torch.tensor(qpos_raw, dtype=torch.float32, device=reranker.device)
    else:
        qpos_raw_t = qpos_raw.to(reranker.device)
    qpos_fs_norm = reranker._fs_norm_qpos(qpos_raw_t.unsqueeze(0)).expand(k, -1)

    z_pred, _, _, _, _, _ = reranker.foresight(fs_images, action_fs_norm, qpos=qpos_fs_norm)
    if z_pred.dim() == 3:
        z_pred = z_pred[:, -1]
    if reranker.foresight_config.get("residual_prediction", False):
        marker_win = marker_window.unsqueeze(0).to(reranker.device)
        z_cur_raw, _ = reranker.foresight.tactile_vae.encode_single_frame(marker_win)
        z_pred = z_cur_raw.reshape(1, -1).expand_as(z_pred) + z_pred
    c = reranker.foresight_config.get("tactile_vae_latent_dim", 16)
    marker_norm = reranker.foresight.tactile_vae.decoder(z_pred.reshape(k, c, 3, 3))
    vae_mean = vae_mean.to(reranker.device).view(1, 1, 1, 2)
    vae_std = vae_std.to(reranker.device).view(1, 1, 1, 2)
    marker_raw = marker_norm * vae_std + vae_mean

    # Scorer expects a short sequence.  Foresight predicts one future marker, so
    # repeat it as a constant consequence window for this first reranking test.
    marker_seq = marker_raw.unsqueeze(1).expand(-1, WINDOW, -1, -1, -1)
    action_seq = actions_raw[:, :WINDOW, :]
    task_id = torch.zeros(k, dtype=torch.long, device=reranker.device)
    scores = scorer.score(marker_seq, action_seq, task_id, mode=score_mode)
    return scores, marker_raw, z_pred


def summarize_arrays(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def simulated_eval(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    reranker = ForesightOnlyReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )
    scorer = ActionAwareScorerRuntime(args.scorer_ckpt, device=args.device)
    vae_ckpt = reranker.foresight_config.get(
        "tactile_vae_ckpt", "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt"
    )
    vae_mean, vae_std = load_tactile_vae_norm(vae_ckpt)

    ann_path = os.path.join(args.data_dir, "annotations.pkl")
    with open(ann_path, "rb") as f:
        ann = pickle.load(f)
    frames = _collect_insertion_frames(
        _find_hdf5_files(args.data_dir),
        ann,
        cs=reranker.pred_horizon,
        obs_horizon=reranker.obs_horizon,
    )
    rng = np.random.default_rng(args.seed)
    if len(frames) > args.n_eval:
        frames = [frames[i] for i in rng.choice(len(frames), args.n_eval, replace=False)]

    rows = []
    expert_rank1 = 0
    expert_top3 = 0
    expert_rank_count = 0
    action_better_than_random = 0
    score_expert_gt_random = 0
    noise_scales = [float(x) for x in args.noise_scales.split(",")]

    for hdf5_path, ep_name, t in tqdm(frames, desc="Action-aware rerank"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)
            expert = torch.tensor(data["action_expert"], dtype=torch.float32, device=reranker.device)
            action_std = expert.std().clamp(min=1e-4)
            candidates = []
            if args.include_expert:
                candidates.append(expert)
            n_noisy = args.K - len(candidates)
            for i in range(n_noisy):
                scale = noise_scales[i % len(noise_scales)]
                candidates.append(expert + torch.randn_like(expert) * action_std * scale)
            actions = torch.stack(candidates)

            aa_scores, _, _ = score_with_action_aware(
                reranker,
                scorer,
                actions,
                data["qpos_raw"],
                data["marker_window"],
                data["foresight_images"],
                args.score_mode,
                vae_mean,
                vae_std,
            )
            aa_np = aa_scores.detach().cpu().numpy()
            actions_np = actions.detach().cpu().numpy()
            expert_flat = data["action_expert"].reshape(-1)
            l1 = np.array([np.abs(actions_np[i].reshape(-1) - expert_flat).mean() for i in range(args.K)])

            rank = np.argsort(-aa_np)
            best_idx = int(rank[0])
            random_idx = int(rng.integers(args.K))
            if args.include_expert:
                expert_rank = int(np.where(rank == 0)[0][0])
                expert_rank1 += int(expert_rank == 0)
                expert_top3 += int(expert_rank < 3)
                expert_rank_count += 1
            else:
                expert_rank = None
            action_better_than_random += int(l1[best_idx] < l1[random_idx])
            if args.include_expert:
                score_expert_gt_random += int(aa_np[0] > aa_np[random_idx])

            corr = np.corrcoef(aa_np, -l1)[0, 1] if np.std(aa_np) > 1e-8 and np.std(l1) > 1e-8 else 0.0
            rows.append(
                {
                    "episode": ep_name,
                    "t": int(t),
                    "expert_rank": expert_rank,
                    "best_idx": best_idx,
                    "random_idx": random_idx,
                    "aa_score_expert": float(aa_np[0]),
                    "aa_score_best": float(aa_np[best_idx]),
                    "aa_score_random": float(aa_np[random_idx]),
                    "l1_best": float(l1[best_idx]),
                    "l1_random": float(l1[random_idx]),
                    "l1_oracle": float(l1.min()),
                    "l1_worst": float(l1[rank[-1]]),
                    "score_l1_corr": float(corr),
                    "aa_score_range": float(aa_np.max() - aa_np.min()),
                }
            )
        except Exception as exc:
            print(f"Skip {ep_name} t={t}: {exc}")

    n = len(rows)
    if n == 0:
        raise RuntimeError("No valid evaluation frames")
    result = {
        "config": vars(args),
        "n_frames": n,
        "expert_rank1": expert_rank1 / expert_rank_count if expert_rank_count else None,
        "expert_top3": expert_top3 / expert_rank_count if expert_rank_count else None,
        "aa_beats_random_l1": action_better_than_random / n,
        "expert_score_beats_random": score_expert_gt_random / expert_rank_count if expert_rank_count else None,
        "l1_best": summarize_arrays([r["l1_best"] for r in rows]),
        "l1_random": summarize_arrays([r["l1_random"] for r in rows]),
        "l1_oracle": summarize_arrays([r["l1_oracle"] for r in rows]),
        "score_l1_corr": summarize_arrays([r["score_l1_corr"] for r in rows]),
        "aa_score_range": summarize_arrays([r["aa_score_range"] for r in rows]),
        "rows": rows,
    }
    out_path = OUT_DIR / f"simulated_rerank_{args.score_mode}_K{args.K}_N{n}.json"
    if not args.include_expert:
        out_path = OUT_DIR / f"simulated_noexpert_rerank_{args.score_mode}_K{args.K}_N{n}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--foresight_ckpt", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full")
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/action_aware_marker_scorer_joint_abs/action_aware_marker_scorer_final.pt")
    parser.add_argument("--data_dir", default="/home/chenshuai/data/dataset/0414")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_mode", default="hybrid", choices=["log_p_good", "p_good", "quality", "hybrid"])
    parser.add_argument("--noise_scales", default="0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--include_expert", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    simulated_eval(parse_args())
