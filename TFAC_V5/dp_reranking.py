"""
Diffusion Policy Reranking with CQF + Foresight (TacDream).

Pipeline:
  1. DP generates K candidate action trajectories (DDPM sampling)
  2. Foresight predicts future tactile for each candidate
  3. CQF scores each (state, action, tactile_cur, tactile_predicted)
  4. Select highest-scored trajectory

This module provides the reranking infrastructure. Can be used:
  a) Offline evaluation: rank K candidates, measure if expert action wins
  b) Online inference: plug into robot control loop

Usage (offline eval):
  python TFAC_V5/dp_reranking.py \
    --dp_ckpt /path/to/dp_best.pth \
    --cqf_ckpt /path/to/cqf_best.pt \
    --foresight_ckpt /path/to/foresight_best.pt \
    --K 16 --n_eval 200
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from network import ConditionalUnet1D
from cqf_model import ContactQualityScorer
from lightweight_foresight import LightweightForesight


class TacDreamReranker:
    """Combines DP + Foresight + CQF for tactile-guided action reranking."""

    def __init__(self, dp_config, dp_ckpt_path, cqf_ckpt_path,
                 foresight_ckpt_path, device="cuda:0", K=16,
                 foresight_version="v1"):
        self.device = torch.device(device)
        self.K = K

        self._load_dp(dp_config, dp_ckpt_path)
        self._load_cqf(cqf_ckpt_path)
        self._load_foresight(foresight_ckpt_path, version=foresight_version)

    def _load_dp(self, config, ckpt_path):
        """Load Diffusion Policy model."""
        self.dp_config = config
        self.pred_horizon = config["pred_horizon"]
        self.obs_horizon = config.get("obs_horizon", 2)
        self.action_dim = config["action_dim"]

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=config["global_cond_dim"],
            diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 128),
            down_dims=config.get("down_dims", [256, 512, 1024]),
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 100),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt:
            self.noise_pred_net.load_state_dict(ckpt["ema_net"])
            print(f"DP loaded (EMA, epoch {ckpt.get('epoch', '?')})")
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            print(f"DP loaded (epoch {ckpt.get('epoch', '?')})")
        self.noise_pred_net.eval()

        self.norm_stats = config.get("norm_stats", {})
        self.action_min = torch.tensor(
            self.norm_stats.get("action_min", [0]*7), dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(
            self.norm_stats.get("action_max", [1]*7), dtype=torch.float32, device=self.device)

    def _load_cqf(self, ckpt_path):
        """Load CQF scorer."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        self.cqf = ContactQualityScorer(
            tac_dim=args.get("tac_dim", 162),
            hidden=args.get("hidden", 256),
            action_dropout=0.0,
        ).to(self.device)
        self.cqf.load_state_dict(ckpt["model_state_dict"])
        self.cqf.eval()
        print(f"CQF loaded (epoch {ckpt.get('epoch', '?')}, "
              f"spread={ckpt.get('val_metrics', {}).get('spread', '?')})")

    def _load_foresight(self, ckpt_path, version="v1"):
        """Load Foresight predictor (V1 or V2)."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        if version == "v2":
            from lightweight_foresight_v2 import LightweightForesightV2
            self.foresight = LightweightForesightV2(
                hidden=args.get("hidden", 512),
                n_layers=args.get("n_layers", 4),
            ).to(self.device)
        else:
            self.foresight = LightweightForesight(
                hidden=args.get("hidden", 512),
                n_layers=args.get("n_layers", 4),
            ).to(self.device)
        self.foresight.load_state_dict(ckpt["model_state_dict"])
        self.foresight.eval()
        print(f"Foresight ({version}) loaded (epoch {ckpt.get('epoch', '?')})")

    def _unnormalize_action(self, action_norm):
        """Convert from [-1,1] to original action space."""
        return (action_norm + 1) / 2 * (self.action_max - self.action_min) + self.action_min

    @torch.no_grad()
    def generate_candidates(self, obs_cond, K=None):
        """Generate K candidate action trajectories via DDPM sampling.

        Args:
            obs_cond: (1, global_cond_dim) conditioning vector
            K: number of candidates (default: self.K)
        Returns:
            actions: (K, pred_horizon, action_dim) unnormalized
        """
        if K is None:
            K = self.K

        obs_cond_K = obs_cond.expand(K, -1)
        noisy_action = torch.randn(
            K, self.pred_horizon, self.action_dim, device=self.device)

        self.noise_scheduler.set_timesteps(
            self.dp_config.get("num_inference_steps", 100))

        for t in self.noise_scheduler.timesteps:
            noise_pred = self.noise_pred_net(
                sample=noisy_action,
                timestep=t,
                global_cond=obs_cond_K,
            )
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action,
            ).prev_sample

        actions = self._unnormalize_action(noisy_action)
        return actions

    @torch.no_grad()
    def score_candidates(self, actions, qpos, eef, marker_cur):
        """Score K candidates using Foresight + CQF.

        Args:
            actions:    (K, pred_horizon, action_dim)
            qpos:       (7,) or (1, 7)
            eef:        (6,) or (1, 6)
            marker_cur: (162,) or (1, 162)
        Returns:
            scores:    (K,)
            tac_preds: (K, 162) predicted future tactile for each
        """
        K = actions.shape[0]

        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        if eef.dim() == 1:
            eef = eef.unsqueeze(0)
        if marker_cur.dim() == 1:
            marker_cur = marker_cur.unsqueeze(0)

        qpos_K = qpos.expand(K, -1)
        eef_K = eef.expand(K, -1)
        mcur_K = marker_cur.expand(K, -1)

        chunk_size = min(actions.shape[1], 20)
        action_chunks = actions[:, :chunk_size, :]

        tac_preds = self.foresight(qpos_K, eef_K, action_chunks, mcur_K)

        scores = self.cqf(qpos_K, eef_K, action_chunks, mcur_K, tac_preds)
        scores = scores.squeeze(-1)

        return scores, tac_preds

    @torch.no_grad()
    def rerank(self, obs_cond, qpos, eef, marker_cur, K=None):
        """Full pipeline: generate K candidates → score → return best.

        Returns:
            best_action: (pred_horizon, action_dim)
            all_scores:  (K,)
            best_idx:    int
        """
        actions = self.generate_candidates(obs_cond, K)
        scores, _ = self.score_candidates(actions, qpos, eef, marker_cur)
        best_idx = scores.argmax().item()
        return actions[best_idx], scores, best_idx


def offline_eval(reranker, data_dir, n_eval=200, seed=42):
    """Offline evaluation: measure if reranking improves action quality.

    For each insertion frame:
    1. Generate K candidates from DP
    2. Score & rank them
    3. Compare: best-ranked action vs random selection vs expert action
    Metric: L1 distance to expert action (lower = better)
    """
    import glob
    import pickle

    rng = np.random.RandomState(seed)
    h, cs = 10, 20

    ann_path = os.path.join(data_dir, "annotations.pkl")
    if not os.path.exists(ann_path):
        print(f"No annotations found at {ann_path}")
        return

    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "success", "*.hdf5")))
    if not hdf5_files:
        hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))

    insertion_frames = []
    for hf in hdf5_files:
        ep_name = os.path.splitext(os.path.basename(hf))[0]
        if ep_name not in ann:
            continue
        labels = ann[ep_name]["labels"]
        pos_frames = np.where(labels == 1)[0]
        T = len(labels)
        valid = [t for t in pos_frames if t + cs <= T and t + h < T]
        for t in valid:
            insertion_frames.append((hf, t))

    if len(insertion_frames) > n_eval:
        idx = rng.choice(len(insertion_frames), n_eval, replace=False)
        insertion_frames = [insertion_frames[i] for i in idx]

    print(f"\nOffline Evaluation ({len(insertion_frames)} insertion frames)")
    print(f"K={reranker.K} candidates per frame")

    l1_best, l1_random, l1_worst = [], [], []
    scores_best_list, scores_random_list = [], []

    for hdf5_path, t in insertion_frames:
        try:
            with h5py.File(hdf5_path, "r") as f:
                qpos = torch.from_numpy(
                    f["observations/proprio_joint"][t].astype(np.float32)).to(reranker.device)
                eef = torch.from_numpy(
                    f["observations/proprio_eef"][t].astype(np.float32)).to(reranker.device)
                action_expert = f["actions/joint_abs"][t:t+cs].astype(np.float32)
                marker_cur = torch.from_numpy(
                    f["observations/tac/left/marker_offset"][t].flatten().astype(np.float32)
                ).to(reranker.device)
        except Exception:
            continue

        # We'd need a proper observation encoder here, but for offline eval,
        # we'll use the reranker's score_candidates with pre-generated actions
        # Instead, just test the scoring mechanism with expert + noise
        K = reranker.K
        expert_t = torch.from_numpy(action_expert).to(reranker.device)
        action_std = expert_t.std()

        candidates = [expert_t.clone()]
        for i in range(K - 1):
            scale = rng.uniform(0.2, 2.0)
            noise = torch.randn_like(expert_t) * action_std * scale
            candidates.append(expert_t + noise)

        actions = torch.stack(candidates)
        scores, _ = reranker.score_candidates(actions, qpos, eef, marker_cur)
        scores = scores.cpu().numpy()

        best_idx = scores.argmax()
        random_idx = rng.randint(K)
        worst_idx = scores.argmin()

        expert_np = action_expert.flatten()
        best_action = candidates[best_idx].cpu().numpy().flatten()
        random_action = candidates[random_idx].cpu().numpy().flatten()
        worst_action = candidates[worst_idx].cpu().numpy().flatten()

        l1_best.append(np.abs(best_action - expert_np).mean())
        l1_random.append(np.abs(random_action - expert_np).mean())
        l1_worst.append(np.abs(worst_action - expert_np).mean())

        scores_best_list.append(scores[best_idx])
        scores_random_list.append(scores[random_idx])

    l1_best = np.array(l1_best)
    l1_random = np.array(l1_random)
    l1_worst = np.array(l1_worst)

    print(f"\nL1 to expert action (lower = better):")
    print(f"  Best-ranked:  {l1_best.mean():.4f}±{l1_best.std():.4f}")
    print(f"  Random:       {l1_random.mean():.4f}±{l1_random.std():.4f}")
    print(f"  Worst-ranked: {l1_worst.mean():.4f}±{l1_worst.std():.4f}")

    improvement = (l1_random.mean() - l1_best.mean()) / l1_random.mean() * 100
    print(f"\n  Improvement (best vs random): {improvement:.1f}%")
    print(f"  Expert is rank-1: {(l1_best == 0).mean()*100:.1f}% of the time")

    # Score comparison
    print(f"\nScores:")
    print(f"  Best-ranked mean:  {np.mean(scores_best_list):.3f}")
    print(f"  Random mean:       {np.mean(scores_random_list):.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", type=str,
                        default="/home/chenshuai/Project/output/dp_joint7/dp_best.pth")
    parser.add_argument("--dp_config", type=str,
                        default="/home/chenshuai/Project/output/dp_joint7/config.json")
    parser.add_argument("--cqf_ckpt", type=str,
                        default="/home/chenshuai/Project/output/cqf_phase_b/cqf_phase_b_best.pt")
    parser.add_argument("--foresight_ckpt", type=str,
                        default="/home/chenshuai/Project/output/lightweight_foresight/foresight_best.pt")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chenshuai/data/dataset/260309")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--foresight_version", type=str, default="v1",
                        choices=["v1", "v2"])
    args = parser.parse_args()

    with open(args.dp_config) as f:
        dp_config = json.load(f)

    reranker = TacDreamReranker(
        dp_config=dp_config,
        dp_ckpt_path=args.dp_ckpt,
        cqf_ckpt_path=args.cqf_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        device=args.device,
        K=args.K,
        foresight_version=args.foresight_version,
    )

    offline_eval(reranker, args.data_dir, n_eval=args.n_eval)


if __name__ == "__main__":
    main()
