"""
Diffusion Policy Reranking with CQF + LatentForesight (TacDream).

Pipeline:
  1. DP generates K candidate action trajectories (DDPM sampling)
  2. LatentForesightPretrainModel predicts future tactile latent for each candidate
  3. CQF scores each (state, action, z_cur, z_pred) in 144-dim latent space
  4. Select highest-scored trajectory

Normalization:
  - DP: min-max [-1, 1] (from dp config.json norm_stats)
  - Foresight: mean/std (from foresight args.json norm_stats)
  - CQF: raw values (qpos, eef, action)

Usage (offline eval):
  python TFAC_V5/dp_reranking.py \
    --dp_ckpt /home/chenshuai/Project/output/dp_joint7/dp_best.pth \
    --dp_config /home/chenshuai/Project/output/dp_joint7/config.json \
    --cqf_ckpt /home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt \
    --foresight_ckpt /home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw/foresight_best.ckpt \
    --data_dir /home/chenshuai/data/dataset/260407 \
    --K 16 --n_eval 200
"""

import argparse
import json
import os
import pickle
import sys
import time
import copy

import h5py
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from network import ConditionalUnet1D
from cqf_model import ContactQualityScorer
from pretrain_latent_foresight import LatentForesightPretrainModel

try:
    from clip_pretraining_xiaomi import modified_resnet18
except ImportError:
    from clip_pretraining import modified_resnet18


# ==================== DP Vision Encoder ====================
class VisionEncoder(nn.Module):
    """Same as train_dp_joint.py VisionEncoder."""

    def __init__(self, camera_names, clip_vision_path=None, clip_tac_path=None):
        super().__init__()
        self.camera_names = camera_names

        vis_bb = modified_resnet18()
        if clip_vision_path and os.path.exists(clip_vision_path):
            vis_bb.load_state_dict(
                torch.load(clip_vision_path, map_location='cpu'), strict=False)

        tac_bb = modified_resnet18()
        if clip_tac_path and os.path.exists(clip_tac_path):
            tac_bb.load_state_dict(
                torch.load(clip_tac_path, map_location='cpu'), strict=False)

        self.shared_vision = nn.Sequential(
            vis_bb, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gelsight_encoder = nn.Sequential(
            tac_bb, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, images_list):
        features = []
        for i, cam in enumerate(self.camera_names):
            img = images_list[i]
            if cam == 'gelsight':
                features.append(self.gelsight_encoder(img))
            else:
                features.append(self.shared_vision(img))
        return torch.cat(features, dim=-1)


# ==================== EMA (for loading DP checkpoints) ====================
class EMAModel:
    def __init__(self):
        self.shadow = {}

    @staticmethod
    def from_state_dict(sd):
        m = EMAModel()
        m.shadow = sd
        return m

    def apply_to(self, model):
        model.load_state_dict(self.shadow)


# ==================== TacDream Reranker ====================
class TacDreamReranker:
    """DP + LatentForesight + CQF for tactile-guided action reranking."""

    def __init__(self, dp_config_path, dp_ckpt_path, cqf_ckpt_path,
                 foresight_ckpt_path, foresight_dir=None,
                 device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K

        with open(dp_config_path) as f:
            self.dp_config = json.load(f)

        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)
        self._load_cqf(cqf_ckpt_path)

    def _load_dp(self, ckpt_path):
        """Load Diffusion Policy: noise_pred_net + vision_encoder."""
        config = self.dp_config
        self.pred_horizon = config["pred_horizon"]
        self.obs_horizon = config.get("obs_horizon", 2)
        self.action_dim = config["action_dim"]
        camera_names = config["camera_names"].split(",") if isinstance(
            config["camera_names"], str) else config["camera_names"]
        self.dp_camera_names = camera_names

        # Vision encoder
        self.dp_vision = VisionEncoder(
            camera_names,
            clip_vision_path=config.get("clip_vision_path"),
            clip_tac_path=config.get("clip_tac_path"),
        ).to(self.device)

        # Noise pred net
        down_dims = config.get("down_dims", [256, 512, 1024])
        if isinstance(down_dims, str):
            down_dims = [int(x) for x in down_dims.split(",")]
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=config["global_cond_dim"],
            diffusion_step_embed_dim=config.get("diffusion_step_embed_dim", 128),
            down_dims=down_dims,
            kernel_size=5,
        ).to(self.device)

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 100),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        # Load weights (prefer EMA)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt:
            ema_net = EMAModel.from_state_dict(ckpt["ema_net"])
            ema_net.apply_to(self.noise_pred_net)
            ema_vis = EMAModel.from_state_dict(ckpt["ema_vis"])
            ema_vis.apply_to(self.dp_vision)
            print(f"DP loaded (EMA, epoch {ckpt.get('epoch', '?')})")
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_vision.load_state_dict(ckpt["vision_encoder"])
            print(f"DP loaded (epoch {ckpt.get('epoch', '?')})")
        self.noise_pred_net.eval()
        self.dp_vision.eval()

        # DP normalization (min-max → [-1, 1])
        ns = config["norm_stats"]
        self.dp_action_min = torch.tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.dp_action_max = torch.tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.dp_qpos_min = torch.tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.dp_qpos_max = torch.tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)

    def _load_foresight(self, ckpt_path, foresight_dir=None):
        """Load LatentForesightPretrainModel."""
        if foresight_dir is None:
            foresight_dir = os.path.dirname(ckpt_path)
        args_path = os.path.join(foresight_dir, "args.json")

        with open(args_path) as f:
            self.foresight_config = json.load(f)

        config = self.foresight_config
        camera_names = config['camera_names']
        cam_backbone_mapping = {cam: 0 for cam in camera_names}

        self.foresight = LatentForesightPretrainModel(
            camera_names=camera_names,
            cam_backbone_mapping=cam_backbone_mapping,
            hidden_dim=config['hidden_dim'],
            state_dim=config['state_dim'],
            foresight_layers=config.get('foresight_layers', 3),
            foresight_nheads=config.get('foresight_nheads', 8),
            foresight_dim_feedforward=config.get('foresight_dim_feedforward', 2048),
            dropout=config.get('dropout', 0.1),
            tactile_mode=config.get('tactile_mode', 'marker'),
            max_history=config.get('max_history', 8),
            predict_horizon=config.get('predict_horizon', 1),
            tactile_vae_ckpt=config.get('tactile_vae_ckpt'),
            tactile_vae_latent_dim=config.get('tactile_vae_latent_dim', 16),
            use_delta_pred=config.get('use_delta_pred', False),
            residual_prediction=config.get('residual_prediction', False),
        ).to(self.device)

        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.foresight.load_state_dict(state_dict)
        self.foresight.eval()

        # Foresight normalization (mean/std)
        ns = config['norm_stats']
        self.fs_qpos_mean = torch.tensor(ns['qpos_mean'], dtype=torch.float32, device=self.device)
        self.fs_qpos_std = torch.tensor(ns['qpos_std'], dtype=torch.float32, device=self.device)
        self.fs_action_mean = torch.tensor(ns['action_mean'], dtype=torch.float32, device=self.device)
        self.fs_action_std = torch.tensor(ns['action_std'], dtype=torch.float32, device=self.device)
        self.fs_mo_mean = torch.tensor(ns['marker_offset_mean'], dtype=torch.float32, device=self.device)
        self.fs_mo_std = torch.tensor(ns['marker_offset_std'], dtype=torch.float32, device=self.device)

        self.foresight_chunk = config.get('chunk_size', 10)
        self.vae_window = config.get('tactile_vae_window', 8)
        print(f"Foresight loaded from {ckpt_path}")

    def _load_cqf(self, ckpt_path):
        """Load CQF scorer."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        args = ckpt.get("args", {})
        self.cqf = ContactQualityScorer(
            tac_dim=args.get("tac_dim", 144),
            hidden=args.get("hidden", 256),
            action_dropout=0.0,
        ).to(self.device)
        self.cqf.load_state_dict(ckpt["model_state_dict"])
        self.cqf.eval()
        spread = ckpt.get('val_metrics', {}).get('spread', '?')
        rank1 = ckpt.get('val_metrics', {}).get('rank1_acc', '?')
        print(f"CQF loaded (epoch {ckpt.get('epoch', '?')}, "
              f"spread={spread}, rank1={rank1})")

    # ========== Normalization helpers ==========

    def _dp_unnorm_action(self, action_norm):
        """[-1, 1] → raw action space."""
        return (action_norm + 1) / 2 * (
            self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def _dp_norm_qpos(self, qpos_raw):
        """raw qpos → [-1, 1]."""
        return (qpos_raw - self.dp_qpos_min) / (
            self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def _fs_norm_action(self, action_raw):
        """raw action → foresight mean/std normalized."""
        return (action_raw - self.fs_action_mean) / self.fs_action_std

    def _fs_norm_qpos(self, qpos_raw):
        """raw qpos → foresight mean/std normalized."""
        return (qpos_raw - self.fs_qpos_mean) / self.fs_qpos_std

    # ========== Image preprocessing ==========

    def _preprocess_image(self, img_uint8):
        """HDF5 image (H, W, 3) uint8 → (3, H, W) ImageNet-normalized tensor."""
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0)
        img = img.permute(2, 0, 1)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return normalize(img)

    def _get_marker_window(self, marker_all, t):
        """Get normalized marker window [t-W+1, ..., t] for TactileVAE."""
        T_total = marker_all.shape[0]
        mo_mean = self.fs_mo_mean.cpu().numpy().reshape(1, 1, 2)
        mo_std = self.fs_mo_std.cpu().numpy().reshape(1, 1, 2)

        frames = []
        for i in range(self.vae_window):
            idx = max(0, t - (self.vae_window - 1 - i))
            idx = min(idx, T_total - 1)
            frame = marker_all[idx].astype(np.float32)
            frame = (frame - mo_mean) / mo_std
            frames.append(torch.tensor(frame, dtype=torch.float32))
        return torch.stack(frames)  # (W=8, 9, 9, 2)

    # ========== Core pipeline ==========

    @torch.no_grad()
    def build_obs_cond(self, images_obs, qpos_obs):
        """Build DP observation conditioning from obs_horizon frames.

        Args:
            images_obs: list of obs_horizon dicts, each {cam_name: (H,W,3) uint8}
            qpos_obs:   list of obs_horizon raw qpos arrays (7,)
        Returns:
            obs_cond: (1, global_cond_dim) tensor
        """
        obs_feats = []
        for step_idx in range(self.obs_horizon):
            imgs_t = []
            for cam in self.dp_camera_names:
                img_uint8 = images_obs[step_idx][cam]
                img_t = self._preprocess_image(img_uint8).to(self.device).unsqueeze(0)
                imgs_t.append(img_t)
            vf = self.dp_vision(imgs_t)  # (1, 512*3)
            qpos_norm = self._dp_norm_qpos(
                torch.tensor(qpos_obs[step_idx], dtype=torch.float32,
                             device=self.device).unsqueeze(0))  # (1, 7)
            obs_feats.append(torch.cat([vf, qpos_norm], dim=-1))
        return torch.cat(obs_feats, dim=-1)  # (1, 3086)

    @torch.no_grad()
    def generate_candidates(self, obs_cond, K=None):
        """Generate K candidate action trajectories via DDPM sampling.

        Args:
            obs_cond: (1, global_cond_dim)
        Returns:
            actions_raw: (K, pred_horizon, action_dim) in raw action space
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
                sample=noisy_action, timestep=t, global_cond=obs_cond_K)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=t, sample=noisy_action,
            ).prev_sample

        actions_raw = self._dp_unnorm_action(noisy_action)
        return actions_raw

    @torch.no_grad()
    def score_candidates(self, actions_raw, qpos_raw, eef_raw,
                         marker_window, foresight_images):
        """Score K candidates using Foresight → CQF.

        Args:
            actions_raw:      (K, pred_horizon, 7) raw action space
            qpos_raw:         (7,) numpy or tensor
            eef_raw:          (6,) numpy or tensor
            marker_window:    (W=8, 9, 9, 2) normalized marker window
            foresight_images: list of 3 tensors [global, wrist, marker_win]
                              each (1, ...) for a single frame
        Returns:
            scores: (K,) sigmoid probabilities [0, 1]
            z_preds: (K, 144) predicted future latent
        """
        K = actions_raw.shape[0]

        # z_cur: encode current marker through TactileVAE
        marker_win = marker_window.unsqueeze(0).to(self.device)  # (1, W, 9, 9, 2)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(1, -1)  # (1, 144)
        z_cur_K = z_cur.expand(K, -1)  # (K, 144)

        # Prepare foresight inputs (expand to K)
        fs_images = [img.expand(K, *img.shape[1:]) for img in foresight_images]

        # Normalize action for foresight: raw → mean/std, take first chunk_size steps
        fs_chunk = self.foresight_chunk  # 10
        action_fs = actions_raw[:, :fs_chunk, :]  # (K, 10, 7)
        action_fs_norm = self._fs_norm_action(action_fs)

        # Normalize qpos for foresight
        if isinstance(qpos_raw, np.ndarray):
            qpos_raw_t = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device)
        else:
            qpos_raw_t = qpos_raw.to(self.device)
        qpos_fs_norm = self._fs_norm_qpos(qpos_raw_t.unsqueeze(0)).expand(K, -1)

        # Foresight forward → z_pred
        t_hat, _, _, _, _, _ = self.foresight(
            fs_images, action_fs_norm, qpos=qpos_fs_norm)
        z_preds = t_hat  # (K, 144)

        # CQF scoring (raw values)
        if isinstance(eef_raw, np.ndarray):
            eef_raw_t = torch.tensor(eef_raw, dtype=torch.float32, device=self.device)
        else:
            eef_raw_t = eef_raw.to(self.device)

        qpos_K = qpos_raw_t.unsqueeze(0).expand(K, -1)
        eef_K = eef_raw_t.unsqueeze(0).expand(K, -1)

        cqf_chunk = min(actions_raw.shape[1], 20)
        action_cqf = actions_raw[:, :cqf_chunk, :]  # (K, 20, 7)

        scores = self.cqf.predict(qpos_K, eef_K, action_cqf, z_cur_K, z_preds)
        scores = scores.squeeze(-1)  # (K,)

        return scores, z_preds

    @torch.no_grad()
    def rerank(self, obs_cond, qpos_raw, eef_raw,
               marker_window, foresight_images, K=None):
        """Full pipeline: generate → score → return best."""
        actions = self.generate_candidates(obs_cond, K)
        scores, _ = self.score_candidates(
            actions, qpos_raw, eef_raw, marker_window, foresight_images)
        best_idx = scores.argmax().item()
        return actions[best_idx], scores, best_idx


# ==================== Offline Evaluation ====================

def offline_eval(reranker, data_dir, n_eval=200, seed=42, noise_scales=None):
    """Offline evaluation: expert action + noisy candidates.

    For each insertion frame:
      1. Build K candidates: expert + K-1 noisy variants
      2. Score via full pipeline (Foresight + CQF)
      3. Measure: L1 to expert, rank-1 accuracy, score distribution
    """
    if noise_scales is None:
        noise_scales = [0.2, 0.5, 1.0, 1.5, 2.0]

    rng = np.random.RandomState(seed)
    K = reranker.K
    h = 10
    cs = 20

    ann_path = os.path.join(data_dir, "annotations.pkl")
    if not os.path.exists(ann_path):
        print(f"No annotations found at {ann_path}")
        return

    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    # Find HDF5 files
    hdf5_files = []
    for subdir in ["success", "bounce", ""]:
        pattern_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.isdir(pattern_dir):
            for fn in os.listdir(pattern_dir):
                if fn.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(pattern_dir, fn))
    hdf5_files = sorted(set(hdf5_files))

    # Collect insertion frames
    insertion_frames = []
    for hf in hdf5_files:
        ep_name = os.path.splitext(os.path.basename(hf))[0]
        if ep_name not in ann or ep_name == "_meta":
            continue
        labels = ann[ep_name]["labels"]
        T = len(labels)
        pos_frames = np.where(labels == 1)[0]
        for t in pos_frames:
            if t + cs <= T and t + h < T:
                insertion_frames.append((hf, ep_name, t))

    if len(insertion_frames) > n_eval:
        idx = rng.choice(len(insertion_frames), n_eval, replace=False)
        insertion_frames = [insertion_frames[i] for i in idx]

    print(f"\nOffline Evaluation ({len(insertion_frames)} insertion frames, K={K})")

    # Metrics
    expert_rank1 = 0
    expert_top3 = 0
    l1_best, l1_random, l1_worst = [], [], []
    scores_expert_list, scores_best_list, scores_random_list = [], [], []

    for frame_idx, (hdf5_path, ep_name, t) in enumerate(
            tqdm(insertion_frames, desc="Evaluating")):
        try:
            with h5py.File(hdf5_path, "r") as f:
                qpos_raw = f["observations/proprio_joint"][t].astype(np.float32)
                eef_raw = f["observations/proprio_eef"][t].astype(np.float32)
                marker_all = f["observations/tac/left/marker_offset"][:]

                # Expert action chunk (raw)
                action_expert = f["actions/joint_abs"][t:t+cs].astype(np.float32)
                if len(action_expert) < cs:
                    action_expert = np.pad(
                        action_expert,
                        ((0, cs - len(action_expert)), (0, 0)),
                        mode='edge')

                # Foresight images: global, wrist, marker_window
                has_global = "observations/images/global" in f
                has_wrist = "observations/images/wrist" in f

                if has_global:
                    img_global = reranker._preprocess_image(
                        f["observations/images/global"][t]).unsqueeze(0).to(reranker.device)
                else:
                    img_global = torch.zeros(1, 3, 480, 640, device=reranker.device)

                if has_wrist:
                    img_wrist = reranker._preprocess_image(
                        f["observations/images/wrist"][t]).unsqueeze(0).to(reranker.device)
                else:
                    img_wrist = torch.zeros(1, 3, 480, 640, device=reranker.device)

            # Marker window for TactileVAE / foresight
            marker_window = reranker._get_marker_window(marker_all, t)  # (W, 9, 9, 2)
            marker_window_dev = marker_window.unsqueeze(0).to(reranker.device)  # (1, W, 9, 9, 2)

            foresight_images = [img_global, img_wrist, marker_window_dev]

            # Build candidates: expert(idx=0) + K-1 noisy
            expert_t = torch.tensor(action_expert, dtype=torch.float32,
                                    device=reranker.device)
            action_std = expert_t.std()
            candidates = [expert_t.clone()]
            for i in range(K - 1):
                scale = noise_scales[i % len(noise_scales)]
                noise = torch.randn_like(expert_t) * action_std * scale
                candidates.append(expert_t + noise)

            actions = torch.stack(candidates)  # (K, cs, 7)

            scores, z_preds = reranker.score_candidates(
                actions, qpos_raw, eef_raw, marker_window, foresight_images)
            scores_np = scores.cpu().numpy()

            # Rank analysis (expert is index 0)
            rank_order = np.argsort(-scores_np)  # descending
            expert_rank = np.where(rank_order == 0)[0][0]  # 0-indexed

            if expert_rank == 0:
                expert_rank1 += 1
            if expert_rank < 3:
                expert_top3 += 1

            best_idx = rank_order[0]
            random_idx = rng.randint(K)
            worst_idx = rank_order[-1]

            expert_np = action_expert.flatten()
            best_action = candidates[best_idx].cpu().numpy().flatten()
            random_action = candidates[random_idx].cpu().numpy().flatten()
            worst_action = candidates[worst_idx].cpu().numpy().flatten()

            l1_best.append(np.abs(best_action - expert_np).mean())
            l1_random.append(np.abs(random_action - expert_np).mean())
            l1_worst.append(np.abs(worst_action - expert_np).mean())

            scores_expert_list.append(scores_np[0])
            scores_best_list.append(scores_np[best_idx])
            scores_random_list.append(scores_np[random_idx])

        except Exception as e:
            import traceback
            print(f"Error on {ep_name} t={t}: {e}")
            traceback.print_exc()
            continue

    # Report
    n = len(l1_best)
    if n == 0:
        print("No valid frames evaluated!")
        return

    l1_best = np.array(l1_best)
    l1_random = np.array(l1_random)
    l1_worst = np.array(l1_worst)

    print(f"\n{'='*60}")
    print(f"Results ({n} frames, K={K})")
    print(f"{'='*60}")

    print(f"\nExpert Ranking:")
    print(f"  Rank-1 accuracy: {expert_rank1/n*100:.1f}% ({expert_rank1}/{n})")
    print(f"  Top-3 accuracy:  {expert_top3/n*100:.1f}% ({expert_top3}/{n})")

    print(f"\nL1 to expert action (lower = better):")
    print(f"  CQF best-ranked:  {l1_best.mean():.4f} +/- {l1_best.std():.4f}")
    print(f"  Random selection:  {l1_random.mean():.4f} +/- {l1_random.std():.4f}")
    print(f"  CQF worst-ranked: {l1_worst.mean():.4f} +/- {l1_worst.std():.4f}")

    if l1_random.mean() > 0:
        improvement = (l1_random.mean() - l1_best.mean()) / l1_random.mean() * 100
        print(f"  Improvement (best vs random): {improvement:.1f}%")

    print(f"\nCQF Scores (sigmoid, [0,1]):")
    print(f"  Expert mean:      {np.mean(scores_expert_list):.4f} +/- {np.std(scores_expert_list):.4f}")
    print(f"  Best-ranked mean: {np.mean(scores_best_list):.4f} +/- {np.std(scores_best_list):.4f}")
    print(f"  Random mean:      {np.mean(scores_random_list):.4f} +/- {np.std(scores_random_list):.4f}")


def main():
    parser = argparse.ArgumentParser(description="TacDream: DP + Foresight + CQF Reranking")
    parser.add_argument("--dp_ckpt", type=str,
                        default="/home/chenshuai/Project/output/dp_joint7/dp_best.pth")
    parser.add_argument("--dp_config", type=str,
                        default="/home/chenshuai/Project/output/dp_joint7/config.json")
    parser.add_argument("--cqf_ckpt", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt")
    parser.add_argument("--foresight_ckpt", type=str,
                        default="/home/chenshuai/data/xiaomi_act/latent_foresight_pretrain_dw/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", type=str, default=None,
                        help="Dir containing args.json (default: same as foresight_ckpt)")
    parser.add_argument("--data_dir", type=str,
                        default="/home/chenshuai/data/dataset/260407")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reranker = TacDreamReranker(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        cqf_ckpt_path=args.cqf_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )

    offline_eval(reranker, args.data_dir, n_eval=args.n_eval, seed=args.seed)


if __name__ == "__main__":
    main()
