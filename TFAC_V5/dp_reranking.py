"""
Diffusion Policy Reranking with CQF + LatentForesight (TacDream).

Pipeline:
  1. DP generates K candidate action trajectories (DDPM sampling)
  2. LatentForesightPretrainModel predicts future tactile latent for each candidate
  3. CQF scores each (state, action, z_cur, z_pred) in 144-dim latent space
  4. Select highest-scored trajectory

Supports two DP variants:
  - "clip_tactile_image": CLIP ResNet18 + GelSight image (dp_joint7)
  - "tactile_vae_frozen": Official ResNet18 + SpatialSoftmax + FrozenTactileVAE (dp_tac_vae_shift4)

Normalization:
  - DP: min-max [-1, 1] (from dp config.json norm_stats)
  - Foresight: mean/std (from dataset_stats.pkl or args.json)
  - CQF: raw values (qpos, eef, action)

Usage:
  python TFAC_V5/dp_reranking.py \
    --mode dp_sampling \
    --dp_ckpt /home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth \
    --dp_config /home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json \
    --cqf_ckpt /home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt \
    --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full \
    --data_dir /home/chenshuai/data/dataset/0414 \
    --K 16 --n_eval 200
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from network import ConditionalUnet1D, replace_bn_with_gn
from cqf_model import ContactQualityScorer
from pretrain_latent_foresight import LatentForesightPretrainModel
from tactile_vae import TactileVAE, build_tactile_vae


# ==================== DP Vision Encoders ====================

class SpatialSoftmax(nn.Module):
    def __init__(self, h, w, num_channels):
        super().__init__()
        self.register_buffer('pos_x', torch.linspace(0.0, 1.0, w).reshape(1, 1, 1, w))
        self.register_buffer('pos_y', torch.linspace(0.0, 1.0, h).reshape(1, 1, h, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        attention = torch.softmax(x.reshape(B, C, H * W), dim=-1).reshape(B, C, H, W)
        expected_x = (attention * self.pos_x).sum(dim=(2, 3))
        expected_y = (attention * self.pos_y).sum(dim=(2, 3))
        return torch.cat([expected_x, expected_y], dim=-1)


def _make_resnet18_backbone():
    resnet = torchvision.models.resnet18()
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    replace_bn_with_gn(backbone, features_per_group=16)
    return backbone


class OfficialVisionEncoder(nn.Module):
    """Per-camera independent ResNet18 + SpatialSoftmax → 1024 dim/camera."""

    def __init__(self, camera_names, input_h=200, input_w=266):
        super().__init__()
        self.camera_names = camera_names
        base_backbone = _make_resnet18_backbone()
        self.encoders = nn.ModuleDict()
        for cam in camera_names:
            self.encoders[cam] = copy.deepcopy(base_backbone)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_h, input_w)
            feat = base_backbone(dummy)
            _, C, H, W = feat.shape
        self.spatial_softmax = SpatialSoftmax(H, W, C)
        self.feat_dim = C * 2

    def forward(self, images_dict):
        features = []
        for cam in self.camera_names:
            feat_map = self.encoders[cam](images_dict[cam])
            features.append(self.spatial_softmax(feat_map))
        return torch.cat(features, dim=-1)


class FrozenTactileVAEEncoder(nn.Module):
    """Frozen TactileVAE: (B, 8, 9, 9, 2) raw marker_offset → (B, 144)."""
    TAC_MEAN = np.array([0.2102, -0.6422], dtype=np.float32)
    TAC_STD = np.array([1.6805, 3.6717], dtype=np.float32)

    def __init__(self, vae_checkpoint_path, latent_dim=16, temporal_window=8):
        super().__init__()
        self.vae = build_tactile_vae(latent_dim=latent_dim, temporal_window=temporal_window)
        if vae_checkpoint_path and os.path.exists(vae_checkpoint_path):
            ckpt = torch.load(vae_checkpoint_path, map_location='cpu')
            sd = ckpt.get('model_state_dict', ckpt)
            self.vae.load_state_dict(sd)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.feat_dim = latent_dim * 3 * 3
        self.temporal_window = temporal_window
        self.register_buffer('tac_mean', torch.tensor(self.TAC_MEAN))
        self.register_buffer('tac_std', torch.tensor(self.TAC_STD))

    @torch.no_grad()
    def forward(self, marker_seq):
        marker_norm = (marker_seq - self.tac_mean) / self.tac_std
        z_last, _ = self.vae.encode_single_frame(marker_norm)
        return z_last.flatten(1)


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

        self.dp_variant = self.dp_config.get("variant", "clip_tactile_image")
        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)
        self._load_cqf(cqf_ckpt_path)

    def _load_dp(self, ckpt_path):
        config = self.dp_config
        self.pred_horizon = config["pred_horizon"]
        self.obs_horizon = config.get("obs_horizon", 2)
        self.action_dim = config["action_dim"]
        self.action_shift = config.get("action_shift", 0)

        camera_names = config["camera_names"]
        if isinstance(camera_names, str):
            camera_names = camera_names.split(",")
        self.dp_camera_names = camera_names

        if self.dp_variant == "tactile_vae_frozen":
            input_h = config.get("input_h", 200)
            input_w = config.get("input_w", 266)
            self.dp_vision = OfficialVisionEncoder(
                camera_names, input_h=input_h, input_w=input_w
            ).to(self.device)

            vae_ckpt = config.get("vae_checkpoint",
                                  "/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
            self.dp_tac_encoder = FrozenTactileVAEEncoder(
                vae_ckpt,
                latent_dim=config.get("vae_latent_dim", 16),
                temporal_window=config.get("tac_history", 8),
            ).to(self.device)
            self.dp_tac_encoder.eval()

            self.dp_input_h = input_h
            self.dp_input_w = input_w
            self.dp_tac_history = config.get("tac_history", 8)
        else:
            raise ValueError(f"Unsupported DP variant: {self.dp_variant}")

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

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 100),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if "ema_net" in ckpt:
            EMAModel.from_state_dict(ckpt["ema_net"]).apply_to(self.noise_pred_net)
            EMAModel.from_state_dict(ckpt["ema_vis"]).apply_to(self.dp_vision)
            print(f"DP loaded (EMA, epoch {ckpt.get('epoch', '?')}, variant={self.dp_variant})")
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_vision.load_state_dict(ckpt["vision_encoder"])
            print(f"DP loaded (epoch {ckpt.get('epoch', '?')}, variant={self.dp_variant})")
        self.noise_pred_net.eval()
        self.dp_vision.eval()

        ns = config["norm_stats"]
        self.dp_action_min = torch.tensor(ns["action_min"], dtype=torch.float32, device=self.device)
        self.dp_action_max = torch.tensor(ns["action_max"], dtype=torch.float32, device=self.device)
        self.dp_qpos_min = torch.tensor(ns["qpos_min"], dtype=torch.float32, device=self.device)
        self.dp_qpos_max = torch.tensor(ns["qpos_max"], dtype=torch.float32, device=self.device)

    def _load_foresight(self, ckpt_path, foresight_dir=None):
        if foresight_dir is None:
            foresight_dir = os.path.dirname(ckpt_path)

        args_path = os.path.join(foresight_dir, "args.json")
        with open(args_path) as f:
            self.foresight_config = json.load(f)
        config = self.foresight_config

        self.use_state_trajectory = bool(config.get('use_state_trajectory', False))

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
        missing, unexpected = self.foresight.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(unexpected)
        print(f"Foresight loaded: {n_loaded} keys matched, "
              f"{len(missing)} missing (backbone/VAE), {len(unexpected)} unexpected")
        if self.use_state_trajectory:
            print("  *** use_state_trajectory=True: candidate actions treated as qpos ***")
        self.foresight.eval()

        # Load norm stats from dataset_stats.pkl (preferred) or args.json
        stats_path = os.path.join(foresight_dir, "dataset_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                ns = pickle.load(f)
            print(f"  Foresight norm stats from dataset_stats.pkl")
        else:
            ns = config.get('norm_stats', config)
            print(f"  Foresight norm stats from args.json")

        self.fs_qpos_mean = torch.tensor(ns['qpos_mean'], dtype=torch.float32, device=self.device)
        self.fs_qpos_std = torch.tensor(ns['qpos_std'], dtype=torch.float32, device=self.device)
        self.fs_action_mean = torch.tensor(ns['action_mean'], dtype=torch.float32, device=self.device)
        self.fs_action_std = torch.tensor(ns['action_std'], dtype=torch.float32, device=self.device)

        self.foresight_chunk = config.get('chunk_size', 10)
        self.vae_window = config.get('tactile_vae_window', 8)
        if self.vae_window is None:
            self.vae_window = 8

    def _load_cqf(self, ckpt_path):
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
        print(f"CQF loaded (epoch {ckpt.get('epoch', '?')}, spread={spread}, rank1={rank1})")

    # ========== Normalization helpers ==========

    def _dp_unnorm_action(self, action_norm):
        return (action_norm + 1) / 2 * (self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def _dp_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.dp_qpos_min) / (self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def _fs_norm_action(self, action_raw):
        return (action_raw - self.fs_action_mean) / self.fs_action_std

    def _fs_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.fs_qpos_mean) / self.fs_qpos_std

    # ========== Image preprocessing ==========

    def _preprocess_image(self, img_uint8, resize=None):
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        if resize is not None:
            img = transforms.functional.resize(img, resize)
        return img

    def _get_marker_window(self, marker_all, t, normalize_for_foresight=True):
        """Get marker window [t-W+1, ..., t] for TactileVAE.

        Args:
            normalize_for_foresight: if True, normalize with foresight stats
                                     if False, return raw (for DP TactileVAE which normalizes internally)
        """
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.vae_window):
            idx = max(0, t - (self.vae_window - 1 - i))
            idx = min(idx, T_total - 1)
            frame = marker_all[idx].astype(np.float32)
            frames.append(frame)
        window = np.stack(frames)  # (W, 9, 9, 2)

        if normalize_for_foresight:
            mo_mean = self.fs_qpos_mean.new_tensor(
                FrozenTactileVAEEncoder.TAC_MEAN).reshape(1, 1, 1, 2)
            mo_std = self.fs_qpos_mean.new_tensor(
                FrozenTactileVAEEncoder.TAC_STD).reshape(1, 1, 1, 2)
            window_t = torch.tensor(window, dtype=torch.float32)
            window_t = (window_t - mo_mean.cpu()) / mo_std.cpu()
            return window_t
        return torch.tensor(window, dtype=torch.float32)

    def _get_dp_marker_history(self, marker_all, t):
        """Get raw marker history for DP's FrozenTactileVAEEncoder (normalizes internally)."""
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.dp_tac_history):
            idx = max(0, t - (self.dp_tac_history - 1 - i))
            idx = min(idx, T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        return torch.tensor(np.stack(frames), dtype=torch.float32)

    # ========== Core pipeline ==========

    @torch.no_grad()
    def build_obs_cond(self, images_obs, qpos_obs, marker_hists):
        """Build DP observation conditioning.

        Args:
            images_obs:   list of obs_horizon dicts {cam: (H,W,3) uint8}
            qpos_obs:     list of obs_horizon raw qpos arrays (7,)
            marker_hists: list of obs_horizon marker histories (tac_history, 9, 9, 2)
        """
        obs_feats = []
        dp_resize = [self.dp_input_h, self.dp_input_w]

        for step_idx in range(self.obs_horizon):
            imgs_dict = {}
            for cam in self.dp_camera_names:
                img_uint8 = images_obs[step_idx][cam]
                img_t = self._preprocess_image(img_uint8, resize=dp_resize)
                imgs_dict[cam] = img_t.to(self.device).unsqueeze(0)
            vf = self.dp_vision(imgs_dict)

            tf = self.dp_tac_encoder(
                marker_hists[step_idx].unsqueeze(0).to(self.device))

            qpos_norm = self._dp_norm_qpos(
                torch.tensor(qpos_obs[step_idx], dtype=torch.float32,
                             device=self.device).unsqueeze(0))

            obs_feats.append(torch.cat([vf, tf, qpos_norm], dim=-1))

        return torch.cat(obs_feats, dim=-1)

    @torch.no_grad()
    def generate_candidates(self, obs_cond, K=None):
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

        return self._dp_unnorm_action(noisy_action)

    @torch.no_grad()
    def score_candidates(self, actions_raw, qpos_raw, eef_raw,
                         marker_window, foresight_images):
        """Score K candidates using Foresight → CQF.

        Args:
            actions_raw:      (K, pred_horizon, 7) raw action space
            qpos_raw:         (7,) numpy or tensor
            eef_raw:          (6,) numpy or tensor
            marker_window:    (W=8, 9, 9, 2) normalized marker window (for foresight)
            foresight_images: list of 3 tensors [global, wrist, marker_win]
        """
        K = actions_raw.shape[0]

        # z_cur via foresight's TactileVAE
        marker_win = marker_window.unsqueeze(0).to(self.device)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(1, -1).expand(K, -1)

        fs_images = [img.expand(K, *img.shape[1:]) for img in foresight_images]

        # Normalize candidates for foresight
        fs_chunk = self.foresight_chunk
        action_fs = actions_raw[:, :fs_chunk, :]

        if self.use_state_trajectory:
            # action_mean/std was used for qpos trajectory during training
            action_fs_norm = self._fs_norm_action(action_fs)
        else:
            action_fs_norm = self._fs_norm_action(action_fs)

        if isinstance(qpos_raw, np.ndarray):
            qpos_raw_t = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device)
        else:
            qpos_raw_t = qpos_raw.to(self.device)
        qpos_fs_norm = self._fs_norm_qpos(qpos_raw_t.unsqueeze(0)).expand(K, -1)

        t_hat, _, _, _, _, _ = self.foresight(
            fs_images, action_fs_norm, qpos=qpos_fs_norm)
        z_preds = t_hat

        # CQF scoring (raw values)
        if isinstance(eef_raw, np.ndarray):
            eef_raw_t = torch.tensor(eef_raw, dtype=torch.float32, device=self.device)
        else:
            eef_raw_t = eef_raw.to(self.device)

        qpos_K = qpos_raw_t.unsqueeze(0).expand(K, -1)
        eef_K = eef_raw_t.unsqueeze(0).expand(K, -1)
        cqf_chunk = min(actions_raw.shape[1], 20)
        action_cqf = actions_raw[:, :cqf_chunk, :]

        scores = self.cqf.predict(qpos_K, eef_K, action_cqf, z_cur, z_preds).squeeze(-1)
        return scores, z_preds

    @torch.no_grad()
    def rerank(self, obs_cond, qpos_raw, eef_raw,
               marker_window, foresight_images, K=None):
        actions = self.generate_candidates(obs_cond, K)
        scores, _ = self.score_candidates(
            actions, qpos_raw, eef_raw, marker_window, foresight_images)
        best_idx = scores.argmax().item()
        return actions[best_idx], scores, best_idx


# ==================== Evaluation Helpers ====================

def _find_hdf5_files(data_dir):
    hdf5_files = []
    for subdir in ["success", "bounce", ""]:
        pattern_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        if os.path.isdir(pattern_dir):
            for fn in os.listdir(pattern_dir):
                if fn.endswith(".hdf5"):
                    hdf5_files.append(os.path.join(pattern_dir, fn))
    return sorted(set(hdf5_files))


def _collect_insertion_frames(hdf5_files, ann, cs=20, h=10, obs_horizon=2):
    insertion_frames = []
    for hf in hdf5_files:
        ep_name = os.path.splitext(os.path.basename(hf))[0]
        if ep_name not in ann or ep_name == "_meta":
            continue
        labels = ann[ep_name]["labels"]
        T = len(labels)
        for t_val in np.where(labels == 1)[0]:
            if t_val + cs <= T and t_val + h < T and t_val >= obs_horizon - 1:
                insertion_frames.append((hf, ep_name, int(t_val)))
    return insertion_frames


def _load_frame_data(reranker, hdf5_path, t, obs_horizon):
    """Load all data needed for one frame evaluation."""
    with h5py.File(hdf5_path, "r") as f:
        qpos_raw = f["observations/proprio_joint"][t].astype(np.float32)
        eef_raw = f["observations/proprio_eef"][t].astype(np.float32)
        marker_all = f["observations/tac/left/marker_offset"][:]
        ep_len = marker_all.shape[0]

        cs = reranker.pred_horizon
        action_start = t + reranker.action_shift
        action_expert = f["actions/joint_abs"][action_start:action_start + cs].astype(np.float32)
        if len(action_expert) < cs:
            action_expert = np.pad(action_expert,
                                   ((0, cs - len(action_expert)), (0, 0)), mode='edge')

        # obs_cond data (obs_horizon frames)
        images_obs = []
        qpos_obs = []
        marker_hists = []
        for step in range(obs_horizon):
            t_obs = max(0, t - obs_horizon + 1 + step)
            obs_dict = {}
            for cam in reranker.dp_camera_names:
                key = f"observations/images/{cam}"
                if key in f:
                    obs_dict[cam] = f[key][t_obs]
                else:
                    obs_dict[cam] = np.zeros((480, 640, 3), dtype=np.uint8)
            images_obs.append(obs_dict)
            qpos_obs.append(f["observations/proprio_joint"][t_obs].astype(np.float32))
            marker_hists.append(reranker._get_dp_marker_history(marker_all, t_obs))

        # Foresight images (single frame)
        foresight_imgs = []
        for cam_name in ['global', 'wrist']:
            key = f"observations/images/{cam_name}"
            if key in f:
                img = reranker._preprocess_image(f[key][t]).unsqueeze(0).to(reranker.device)
            else:
                img = torch.zeros(1, 3, 480, 640, device=reranker.device)
            foresight_imgs.append(img)

        marker_window = reranker._get_marker_window(marker_all, t, normalize_for_foresight=True)
        marker_window_dev = marker_window.unsqueeze(0).to(reranker.device)
        foresight_imgs.append(marker_window_dev)

    return {
        'qpos_raw': qpos_raw,
        'eef_raw': eef_raw,
        'action_expert': action_expert,
        'images_obs': images_obs,
        'qpos_obs': qpos_obs,
        'marker_hists': marker_hists,
        'marker_window': marker_window,
        'foresight_images': foresight_imgs,
    }


# ==================== Offline Evaluation ====================

def offline_eval(reranker, data_dir, n_eval=200, seed=42, noise_scales=None):
    """Expert action + noisy candidates."""
    if noise_scales is None:
        noise_scales = [0.2, 0.5, 1.0, 1.5, 2.0]

    rng = np.random.RandomState(seed)
    K = reranker.K
    cs = reranker.pred_horizon

    ann_path = os.path.join(data_dir, "annotations.pkl")
    if not os.path.exists(ann_path):
        print(f"No annotations found at {ann_path}")
        return
    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    hdf5_files = _find_hdf5_files(data_dir)
    insertion_frames = _collect_insertion_frames(
        hdf5_files, ann, cs=cs, obs_horizon=reranker.obs_horizon)

    if len(insertion_frames) > n_eval:
        idx = rng.choice(len(insertion_frames), n_eval, replace=False)
        insertion_frames = [insertion_frames[i] for i in idx]

    print(f"\nOffline Evaluation ({len(insertion_frames)} insertion frames, K={K})")

    expert_rank1 = 0
    expert_top3 = 0
    l1_best, l1_random, l1_worst = [], [], []
    scores_expert_list, scores_best_list, scores_random_list = [], [], []

    for hdf5_path, ep_name, t in tqdm(insertion_frames, desc="Evaluating"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)

            expert_t = torch.tensor(data['action_expert'], dtype=torch.float32,
                                    device=reranker.device)
            action_std = expert_t.std()
            candidates = [expert_t.clone()]
            for i in range(K - 1):
                scale = noise_scales[i % len(noise_scales)]
                candidates.append(expert_t + torch.randn_like(expert_t) * action_std * scale)
            actions = torch.stack(candidates)

            scores, _ = reranker.score_candidates(
                actions, data['qpos_raw'], data['eef_raw'],
                data['marker_window'], data['foresight_images'])
            scores_np = scores.cpu().numpy()

            rank_order = np.argsort(-scores_np)
            expert_rank = np.where(rank_order == 0)[0][0]
            if expert_rank == 0:
                expert_rank1 += 1
            if expert_rank < 3:
                expert_top3 += 1

            best_idx = rank_order[0]
            random_idx = rng.randint(K)
            worst_idx = rank_order[-1]
            expert_np = data['action_expert'].flatten()

            l1_best.append(np.abs(candidates[best_idx].cpu().numpy().flatten() - expert_np).mean())
            l1_random.append(np.abs(candidates[random_idx].cpu().numpy().flatten() - expert_np).mean())
            l1_worst.append(np.abs(candidates[worst_idx].cpu().numpy().flatten() - expert_np).mean())
            scores_expert_list.append(scores_np[0])
            scores_best_list.append(scores_np[best_idx])
            scores_random_list.append(scores_np[random_idx])

        except Exception as e:
            import traceback
            print(f"Error on {ep_name} t={t}: {e}")
            traceback.print_exc()
            continue

    n = len(l1_best)
    if n == 0:
        print("No valid frames evaluated!")
        return

    l1_best, l1_random, l1_worst = np.array(l1_best), np.array(l1_random), np.array(l1_worst)

    print(f"\n{'='*60}")
    print(f"Results ({n} frames, K={K})")
    print(f"{'='*60}")
    print(f"\nExpert Ranking:")
    print(f"  Rank-1: {expert_rank1/n*100:.1f}% ({expert_rank1}/{n})")
    print(f"  Top-3:  {expert_top3/n*100:.1f}% ({expert_top3}/{n})")
    print(f"\nL1 to expert (lower = better):")
    print(f"  CQF best:   {l1_best.mean():.4f} +/- {l1_best.std():.4f}")
    print(f"  Random:     {l1_random.mean():.4f} +/- {l1_random.std():.4f}")
    print(f"  CQF worst:  {l1_worst.mean():.4f} +/- {l1_worst.std():.4f}")
    if l1_random.mean() > 0:
        print(f"  Improvement: {(l1_random.mean() - l1_best.mean()) / l1_random.mean() * 100:.1f}%")
    print(f"\nCQF Scores:")
    print(f"  Expert: {np.mean(scores_expert_list):.4f} +/- {np.std(scores_expert_list):.4f}")
    print(f"  Best:   {np.mean(scores_best_list):.4f} +/- {np.std(scores_best_list):.4f}")
    print(f"  Random: {np.mean(scores_random_list):.4f} +/- {np.std(scores_random_list):.4f}")


def dp_sampling_eval(reranker, data_dir, n_eval=200, seed=42):
    """Evaluation with REAL DP-sampled candidates (no expert in candidate set)."""
    rng = np.random.RandomState(seed)
    K = reranker.K
    cs = reranker.pred_horizon

    ann_path = os.path.join(data_dir, "annotations.pkl")
    if not os.path.exists(ann_path):
        print(f"No annotations found at {ann_path}")
        return
    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    hdf5_files = _find_hdf5_files(data_dir)
    insertion_frames = _collect_insertion_frames(
        hdf5_files, ann, cs=cs, obs_horizon=reranker.obs_horizon)

    if len(insertion_frames) > n_eval:
        idx = rng.choice(len(insertion_frames), n_eval, replace=False)
        insertion_frames = [insertion_frames[i] for i in idx]

    print(f"\nDP Sampling Evaluation ({len(insertion_frames)} frames, K={K})")
    print(f"  obs_horizon={reranker.obs_horizon}, "
          f"DDPM steps={reranker.dp_config.get('num_inference_steps', 100)}, "
          f"action_shift={reranker.action_shift}")

    l1_cqf_best, l1_random, l1_cqf_worst, l1_mean_action = [], [], [], []
    scores_cqf_best, scores_random_list, scores_all_list = [], [], []
    l1_all_candidates = []

    for hdf5_path, ep_name, t in tqdm(insertion_frames, desc="DP Sampling Eval"):
        try:
            data = _load_frame_data(reranker, hdf5_path, t, reranker.obs_horizon)

            obs_cond = reranker.build_obs_cond(
                data['images_obs'], data['qpos_obs'], data['marker_hists'])
            actions_raw = reranker.generate_candidates(obs_cond, K=K)

            scores, _ = reranker.score_candidates(
                actions_raw, data['qpos_raw'], data['eef_raw'],
                data['marker_window'], data['foresight_images'])
            scores_np = scores.cpu().numpy()
            actions_np = actions_raw.cpu().numpy()

            rank_order = np.argsort(-scores_np)
            best_idx, worst_idx = rank_order[0], rank_order[-1]
            random_idx = rng.randint(K)

            expert_flat = data['action_expert'].flatten()
            l1_cqf_best.append(np.abs(actions_np[best_idx].flatten() - expert_flat).mean())
            l1_random.append(np.abs(actions_np[random_idx].flatten() - expert_flat).mean())
            l1_cqf_worst.append(np.abs(actions_np[worst_idx].flatten() - expert_flat).mean())
            l1_mean_action.append(np.abs(actions_np.mean(axis=0).flatten() - expert_flat).mean())

            all_l1 = [np.abs(actions_np[i].flatten() - expert_flat).mean() for i in range(K)]
            l1_all_candidates.append(all_l1)
            scores_cqf_best.append(scores_np[best_idx])
            scores_random_list.append(scores_np[random_idx])
            scores_all_list.append(scores_np)

        except Exception as e:
            import traceback
            print(f"Error on {ep_name} t={t}: {e}")
            traceback.print_exc()
            continue

    n = len(l1_cqf_best)
    if n == 0:
        print("No valid frames evaluated!")
        return

    l1_cqf_best = np.array(l1_cqf_best)
    l1_random = np.array(l1_random)
    l1_cqf_worst = np.array(l1_cqf_worst)
    l1_mean_action = np.array(l1_mean_action)
    l1_oracle = np.array([min(l1s) for l1s in l1_all_candidates])

    all_scores_flat = np.concatenate(scores_all_list)
    all_l1_flat = np.concatenate(l1_all_candidates)
    correlation = np.corrcoef(all_scores_flat, -all_l1_flat)[0, 1]
    cqf_wins = sum(1 for i in range(n) if l1_cqf_best[i] < l1_random[i])

    print(f"\n{'='*60}")
    print(f"DP Sampling Results ({n} frames, K={K})")
    print(f"{'='*60}")

    print(f"\nL1 to expert (lower = better):")
    print(f"  Oracle (best candidate):  {l1_oracle.mean():.4f} +/- {l1_oracle.std():.4f}")
    print(f"  CQF best-ranked:          {l1_cqf_best.mean():.4f} +/- {l1_cqf_best.std():.4f}")
    print(f"  Mean action (ensemble):   {l1_mean_action.mean():.4f} +/- {l1_mean_action.std():.4f}")
    print(f"  Random selection:          {l1_random.mean():.4f} +/- {l1_random.std():.4f}")
    print(f"  CQF worst-ranked:          {l1_cqf_worst.mean():.4f} +/- {l1_cqf_worst.std():.4f}")

    if l1_random.mean() > 0:
        impr_vs_random = (l1_random.mean() - l1_cqf_best.mean()) / l1_random.mean() * 100
        impr_vs_mean = (l1_mean_action.mean() - l1_cqf_best.mean()) / l1_mean_action.mean() * 100
        gap_to_oracle = (l1_cqf_best.mean() - l1_oracle.mean()) / max(l1_oracle.mean(), 1e-8) * 100
        print(f"\n  CQF vs Random improvement: {impr_vs_random:+.1f}%")
        print(f"  CQF vs Mean improvement:   {impr_vs_mean:+.1f}%")
        print(f"  CQF vs Oracle gap:         +{gap_to_oracle:.1f}%")

    print(f"\n  CQF beats random: {cqf_wins}/{n} frames ({cqf_wins/n*100:.1f}%)")
    print(f"  Score-L1 correlation: {correlation:.4f} (>0 means higher score → lower L1)")

    print(f"\nCQF Score distribution:")
    print(f"  Mean:   {all_scores_flat.mean():.4f} +/- {all_scores_flat.std():.4f}")
    print(f"  Range:  [{all_scores_flat.min():.4f}, {all_scores_flat.max():.4f}]")

    score_ranges = [s.max() - s.min() for s in scores_all_list]
    print(f"  Per-frame range: {np.mean(score_ranges):.4f} +/- {np.std(score_ranges):.4f}")


def main():
    parser = argparse.ArgumentParser(description="TacDream: DP + Foresight + CQF Reranking")
    parser.add_argument("--mode", type=str, default="simulated",
                        choices=["simulated", "dp_sampling"])
    parser.add_argument("--dp_ckpt", type=str,
                        default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth")
    parser.add_argument("--dp_config", type=str,
                        default="/home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json")
    parser.add_argument("--cqf_ckpt", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent_0407_perturb_phaseC/cqf_latent_best.pt")
    parser.add_argument("--foresight_ckpt", type=str,
                        default="/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt")
    parser.add_argument("--foresight_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str,
                        default="/home/chenshuai/data/dataset/0414")
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

    if args.mode == "simulated":
        offline_eval(reranker, args.data_dir, n_eval=args.n_eval, seed=args.seed)
    elif args.mode == "dp_sampling":
        dp_sampling_eval(reranker, args.data_dir, n_eval=args.n_eval, seed=args.seed)


if __name__ == "__main__":
    main()
