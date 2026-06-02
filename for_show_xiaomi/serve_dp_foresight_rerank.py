"""
Diffusion Policy + Foresight Reranking Server (TacScore).

Based on serve_dp_policy.py, adds Foresight-based candidate reranking:
  1. Generate K candidate action trajectories (batch DDPM sampling)
  2. For each candidate, LTFT predicts future tactile latent z_pred
  3. Score candidates using multi-dimensional CQV scoring:
     - Primary: Phase-fit (demo distribution matching, 80% Oracle Match)
     - Auxiliary: Smoothness (phase-normalized delta) + Safety (force bound)
  4. Execute the best candidate with action_skip=2

Scoring modes:
  - "fit": Phase-aware demo distribution matching (default, best Oracle agreement)
  - "smoothness": Phase-normalized tactile delta
  - "naive": Legacy -||z_pred - z_cur||
  - "multidim": Weighted combination of fit + smoothness + safety

Usage:
    python -m for_show_xiaomi.serve_dp_foresight_rerank \
        --ckpt_dir /path/to/dp_foresight_joint_ckpt \
        --foresight_dir /path/to/foresight_norm_stats \
        --K 16 --action_skip 2 --scoring_mode fit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from collections import deque

import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_DIFFUSION = os.path.join(_ROOT, "diffusion")
if _DIFFUSION not in sys.path:
    sys.path.append(_DIFFUSION)

from utils import set_seed
from network import ConditionalUnet1D
from for_show_xiaomi.ws_server import TactileACTServer, ClientDisconnected
from TFAC_V5.pretrain_latent_foresight import LatentForesightPretrainModel


_IMG_NORM = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class PhaseAwareScorer:
    """
    Multi-dimensional CQV scorer using phase-annotated demo statistics.

    Scoring dimensions:
      1. fit: standardized distance to phase demo distribution (best Oracle Match: 80%)
      2. smoothness: phase-normalized tactile delta
      3. safety: penalty if z_int exceeds phase force bound (p95)
    """

    def __init__(self, phase_stats_path, device):
        self.device = device
        with open(phase_stats_path, 'rb') as f:
            data = pickle.load(f)
        self.phase_stats = data['phase_stats']
        self.phase_tensors = {}
        for pid, stats in self.phase_stats.items():
            if stats is None:
                continue
            self.phase_tensors[pid] = {
                'delta_norm_mu': stats['delta_norm_mu'],
                'delta_norm_std': stats['delta_norm_std'],
                'z_int_norm_p95': stats['z_int_norm_p95'],
                'z_mu': torch.tensor(stats['z_mu'], dtype=torch.float32, device=device),
                'z_std': torch.tensor(stats['z_std'], dtype=torch.float32, device=device),
            }

    def estimate_phase(self, progress):
        """Estimate current phase from episode progress (0→1)."""
        if progress < 0.3:
            return 0  # approach
        elif progress < 0.6:
            return 1  # insertion
        elif progress < 0.75:
            return 2  # pre_bounce
        elif progress < 0.9:
            return 3  # lift
        else:
            return 4  # reposition

    def score_fit(self, z_pred, phase):
        """Phase-aware demo distribution matching (best method: 80% Oracle Match)."""
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        mu = self.phase_tensors[phase]['z_mu']
        std = self.phase_tensors[phase]['z_std']
        return -((z_pred - mu) / std).abs().mean(dim=-1)

    def score_smoothness(self, z_pred, z_cur, phase):
        """Phase-normalized smoothness."""
        delta = torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        if phase not in self.phase_tensors:
            return -delta
        mu = self.phase_tensors[phase]['delta_norm_mu']
        std = self.phase_tensors[phase]['delta_norm_std']
        return -(delta - mu) / (std + 1e-8)

    def score_safety(self, z_pred, phase):
        """Penalty for exceeding force bound."""
        z_int = z_pred[:, :9]
        z_int_norm = torch.norm(z_int, dim=-1)
        if phase not in self.phase_tensors:
            return torch.zeros(z_pred.shape[0], device=self.device)
        bound = self.phase_tensors[phase]['z_int_norm_p95']
        return -torch.clamp(z_int_norm - bound, min=0)

    def score(self, z_pred, z_cur, phase, mode='fit'):
        """Compute score for K candidates."""
        if mode == 'fit':
            return self.score_fit(z_pred, phase)
        elif mode == 'smoothness':
            return self.score_smoothness(z_pred, z_cur, phase)
        elif mode == 'naive':
            return -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)
        elif mode == 'multidim':
            s_fit = self.score_fit(z_pred, phase)
            s_smooth = self.score_smoothness(z_pred, z_cur, phase)
            s_safe = self.score_safety(z_pred, phase)
            return 0.7 * s_fit + 0.2 * s_smooth + 0.1 * s_safe
        else:
            return -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)


def build_dp_model(config: dict, device: torch.device):
    """Build DP models (vision encoder + UNet + tac encoder)."""
    from train_dp_tac_concat import OfficialVisionEncoder, FrozenTactileVAEEncoder

    camera_names = config["camera_names"]
    if isinstance(camera_names, str):
        camera_names = camera_names.split(",")
    action_dim = config["action_dim"]
    global_cond_dim = config["global_cond_dim"]
    down_dims = config["down_dims"]
    if isinstance(down_dims, str):
        down_dims = [int(x) for x in down_dims.split(",")]
    diffusion_step_embed_dim = config.get("diffusion_step_embed_dim", 128)

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=diffusion_step_embed_dim,
        down_dims=down_dims,
        kernel_size=5,
    ).to(device)

    vis_cams = [c for c in camera_names if c != "gelsight"]
    vision_encoder = OfficialVisionEncoder(vis_cams).to(device)

    vae_checkpoint = config.get("vae_checkpoint", "")
    vae_latent_dim = config.get("vae_latent_dim", 16)
    tac_history = config.get("tac_history", 8)
    tac_encoder = FrozenTactileVAEEncoder(
        vae_checkpoint, latent_dim=vae_latent_dim, temporal_window=tac_history
    ).to(device)

    return vision_encoder, noise_pred_net, tac_encoder


def load_checkpoint(ckpt_path, vision_encoder, noise_pred_net, foresight, device):
    """Load joint-trained checkpoint (DP + Foresight)."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Vision encoder
    vis_sd = ckpt.get("ema_vis", ckpt.get("vision_encoder"))
    vision_encoder.load_state_dict(vis_sd)
    print(f"  loaded {'EMA' if 'ema_vis' in ckpt else 'raw'} vision weights")

    # Noise pred net
    if "ema_net" in ckpt:
        noise_pred_net.load_state_dict(ckpt["ema_net"])
        print(f"  loaded EMA noise_pred_net")
    else:
        noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
        print(f"  loaded raw noise_pred_net")

    # Foresight (joint-trained)
    if "foresight" in ckpt:
        fs_sd = ckpt["foresight"]
        missing, unexpected = foresight.load_state_dict(fs_sd, strict=False)
        print(f"  loaded foresight: {len(fs_sd)} keys, "
              f"{len(missing)} missing (backbone/VAE), {len(unexpected)} unexpected")
    else:
        print(f"  WARNING: no foresight in checkpoint, using pretrained weights")

    epoch = ckpt.get("epoch", "?")
    print(f"  checkpoint epoch={epoch}")


def load_foresight_model(foresight_ckpt, foresight_dir, device):
    """Load LatentForesightPretrainModel (same as train_dp_foresight_joint.py)."""
    args_path = os.path.join(foresight_dir, 'args.json')
    with open(args_path) as f:
        fs_config = json.load(f)

    camera_names = fs_config.get('camera_names', ['global', 'wrist', 'gelsight'])
    cam_backbone_mapping = {cam: 0 for cam in camera_names}

    model = LatentForesightPretrainModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=fs_config.get('hidden_dim', 512),
        state_dim=fs_config.get('state_dim', 7),
        foresight_layers=fs_config.get('foresight_layers', 3),
        foresight_nheads=fs_config.get('foresight_nheads', 8),
        foresight_dim_feedforward=fs_config.get('foresight_dim_feedforward', 2048),
        dropout=fs_config.get('dropout', 0.1),
        tactile_mode=fs_config.get('tactile_mode', 'marker'),
        max_history=fs_config.get('max_history', 8),
        predict_horizon=fs_config.get('predict_horizon', 1),
        tactile_vae_ckpt=fs_config.get('tactile_vae_ckpt'),
        tactile_vae_latent_dim=fs_config.get('tactile_vae_latent_dim', 16),
        use_delta_pred=fs_config.get('use_delta_pred', False),
        residual_prediction=fs_config.get('residual_prediction', False),
    ).to(device)

    # Load pretrained weights (will be overwritten by joint ckpt if available)
    if os.path.exists(foresight_ckpt):
        sd = torch.load(foresight_ckpt, map_location=device)
        if 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(sd) - len(unexpected)
        print(f"[foresight] Loaded pretrained: {n_loaded} keys, "
              f"{len(missing)} missing (backbone/VAE)")

    model.eval()
    return model, fs_config


def load_foresight_norm_stats(foresight_dir, device):
    """Load Foresight normalization stats."""
    stats_path = os.path.join(foresight_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return {
        'action_mean': torch.tensor(stats['action_mean'], dtype=torch.float32, device=device),
        'action_std': torch.tensor(stats['action_std'], dtype=torch.float32, device=device),
        'qpos_mean': torch.tensor(stats['qpos_mean'], dtype=torch.float32, device=device),
        'qpos_std': torch.tensor(stats['qpos_std'], dtype=torch.float32, device=device),
    }


def preprocess_image(raw_img, resize_tf=None, crop_tf=None):
    """uint8 HWC → normalized CHW float tensor (1, C, H, W)."""
    img = np.asarray(raw_img, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    if resize_tf is not None:
        t = resize_tf(t)
    if crop_tf is not None:
        t = crop_tf(t)
    return _IMG_NORM(t).unsqueeze(0)


@torch.no_grad()
def batch_ddpm_inference(noise_pred_net, noise_scheduler, obs_cond, action_dim,
                         pred_horizon, num_inference_steps, K, device):
    """Batch DDPM sampling: generate K candidates in parallel."""
    noise_scheduler.set_timesteps(num_inference_steps)
    # obs_cond: (1, cond_dim) → expand to (K, cond_dim)
    obs_cond_K = obs_cond.expand(K, -1)
    action = torch.randn((K, pred_horizon, action_dim), device=device)

    for t in noise_scheduler.timesteps:
        noise_pred = noise_pred_net(
            action, t.unsqueeze(0).expand(K).to(device), global_cond=obs_cond_K
        )
        action = noise_scheduler.step(noise_pred, t, action).prev_sample

    return action  # (K, pred_horizon, action_dim)


@torch.no_grad()
def foresight_rerank(candidates, foresight, fs_norm, fs_config,
                     obs_images, marker_window, qpos_raw,
                     action_min_t, action_max_t, device,
                     scorer=None, scoring_mode='fit', progress=0.0):
    """
    Score K candidates using LTFT predicted tactile + CQV scoring.

    Args:
        candidates: (K, pred_horizon, action_dim) in [-1, 1]
        obs_images: dict with 'global' and 'wrist' tensors (1, C, H, W)
        marker_window: (1, tac_history, 9, 9, 2) current marker history
        qpos_raw: (1, 7) raw qpos
        scorer: PhaseAwareScorer instance (None for legacy naive scoring)
        scoring_mode: 'fit', 'smoothness', 'naive', 'multidim'
        progress: episode progress [0, 1] for phase estimation

    Returns:
        best_idx: index of best candidate
        scores: (K,) scores for all candidates
    """
    K = candidates.shape[0]
    chunk_size = fs_config.get('chunk_size', 10)

    # Convert DP [-1,1] → raw → Foresight mean/std normalization
    x0_raw = (candidates + 1) / 2 * (action_max_t - action_min_t) + action_min_t
    x0_fs = (x0_raw - fs_norm['action_mean']) / fs_norm['action_std']
    x0_fs_chunk = x0_fs[:, :chunk_size, :]  # (K, chunk_size, 7)

    # Prepare foresight image inputs ordered by fs_config camera_names
    fs_camera_names = fs_config.get('camera_names', ['global', 'wrist', 'gelsight'])
    fs_images = []
    for cam in fs_camera_names:
        if cam == 'gelsight':
            fs_images.append(marker_window.expand(K, -1, -1, -1, -1))  # (K, 8, 9, 9, 2)
        elif cam in obs_images:
            fs_images.append(obs_images[cam].expand(K, -1, -1, -1))  # (K, C, H, W)

    # Qpos for foresight
    qpos_fs = (qpos_raw - fs_norm['qpos_mean']) / fs_norm['qpos_std']
    qpos_fs = qpos_fs.expand(K, -1)  # (K, 7)

    # Foresight forward — get z_pred and z_current
    z_pred, _, _, z_current, _, _ = foresight(
        fs_images, x0_fs_chunk, future_images=None, qpos=qpos_fs
    )
    z_cur = z_current[:1]  # (1, 144) — all K are the same current obs

    if z_pred.dim() == 3:
        z_pred = z_pred[:, -1, :]  # take last prediction step

    # Score using phase-aware CQV scorer
    if scorer is not None:
        phase = scorer.estimate_phase(progress)
        scores = scorer.score(z_pred, z_cur, phase, mode=scoring_mode)
    else:
        # Fallback: legacy naive scoring
        scores = -torch.norm(z_pred - z_cur.expand_as(z_pred), dim=-1)

    best_idx = scores.argmax().item()

    # Decode z_pred back to marker space for logging/analysis
    # z_pred: (K, 144) → (K, 16, 3, 3) → decoder → (K, 9, 9, 2)
    C = fs_config.get('tactile_vae_latent_dim', 16)
    z_pred_spatial = z_pred.reshape(K, C, 3, 3)
    z_cur_spatial = z_cur.reshape(1, C, 3, 3)
    with torch.no_grad():
        marker_pred = foresight.tactile_vae.decoder(z_pred_spatial)   # (K, 9, 9, 2)
        marker_cur_hat = foresight.tactile_vae.decoder(z_cur_spatial) # (1, 9, 9, 2)

    return best_idx, scores, z_pred, z_current, marker_pred, marker_cur_hat


def main():
    parser = argparse.ArgumentParser(description="DP + Foresight Reranking Server")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Joint-trained DP+Foresight checkpoint directory")
    parser.add_argument("--ckpt_name", type=str, default="dp_best.pth")
    parser.add_argument("--foresight_dir", type=str, required=True,
                        help="Foresight config/norm_stats directory")
    parser.add_argument("--foresight_ckpt", type=str, default=None,
                        help="Foresight pretrained ckpt (if not in joint ckpt)")
    parser.add_argument("--K", type=int, default=16,
                        help="Number of candidates to generate and rank")
    parser.add_argument("--scoring_mode", type=str, default="fit",
                        choices=["fit", "smoothness", "naive", "multidim"],
                        help="Scoring method: fit (80%% Oracle Match), smoothness, naive, multidim")
    parser.add_argument("--phase_stats", type=str,
                        default="/home/chenshuai/Project/output/phase_scoring_stats.pkl",
                        help="Phase scoring statistics file")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--action_skip", type=int, default=2,
                        help="Skip first N steps (default=2, aligns with Foresight chunk=10)")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Override denoising steps (default: 100 for DDPM, 20 for DDIM)")
    parser.add_argument("--scheduler", type=str, default="ddim", choices=["ddpm", "ddim"],
                        help="Noise scheduler type (ddim is faster)")
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to save foresight prediction logs (pickle). "
                             "If not set, no logging.")
    cli = parser.parse_args()

    set_seed(cli.seed)
    device = torch.device(
        f"cuda:{cli.gpu}" if torch.cuda.is_available() and cli.gpu >= 0 else "cpu"
    )
    print(f"[rerank-server] device: {device}")

    # Load DP config
    config_path = os.path.join(cli.ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    camera_names = config["camera_names"]
    if isinstance(camera_names, str):
        camera_names = camera_names.split(",")
    action_dim = config["action_dim"]
    pred_horizon = config["pred_horizon"]
    obs_horizon = config.get("obs_horizon", 2)
    num_train_timesteps = config.get("num_train_timesteps", 100)
    tac_history = config.get("tac_history", 8)

    ns = config["norm_stats"]
    action_min = np.array(ns["action_min"], dtype=np.float32)
    action_max = np.array(ns["action_max"], dtype=np.float32)
    qpos_min = np.array(ns["qpos_min"], dtype=np.float32)
    qpos_max = np.array(ns["qpos_max"], dtype=np.float32)
    action_min_t = torch.tensor(action_min, dtype=torch.float32, device=device)
    action_max_t = torch.tensor(action_max, dtype=torch.float32, device=device)

    resize_shape = config.get("resize_shape")
    crop_shape = config.get("crop_shape")
    resize_tf = transforms.Resize(tuple(resize_shape)) if resize_shape else None
    crop_tf = transforms.CenterCrop(tuple(crop_shape)) if crop_shape else None

    # Build DP models
    vision_encoder, noise_pred_net, tac_encoder = build_dp_model(config, device)

    # Build Foresight model
    foresight_ckpt = cli.foresight_ckpt or os.path.join(cli.foresight_dir, 'foresight_best.ckpt')
    foresight, fs_config = load_foresight_model(foresight_ckpt, cli.foresight_dir, device)
    fs_norm = load_foresight_norm_stats(cli.foresight_dir, device)
    fs_chunk_size = fs_config.get('chunk_size', 10)

    # Load joint checkpoint (overwrites both DP and Foresight weights)
    ckpt_path = os.path.join(cli.ckpt_dir, cli.ckpt_name)
    load_checkpoint(ckpt_path, vision_encoder, noise_pred_net, foresight, device)

    vision_encoder.eval()
    noise_pred_net.eval()
    tac_encoder.eval()
    foresight.eval()

    # Load phase-aware scorer
    scorer = None
    if os.path.exists(cli.phase_stats):
        scorer = PhaseAwareScorer(cli.phase_stats, device)
        print(f"[rerank-server] Phase-aware scorer loaded ({cli.scoring_mode} mode)")
    else:
        print(f"[rerank-server] WARNING: no phase_stats at {cli.phase_stats}, using naive scoring")

    print(f"[rerank-server] K={cli.K} candidates, chunk_size={fs_chunk_size}")
    print(f"[rerank-server] scoring_mode={cli.scoring_mode}")
    print(f"[rerank-server] action_skip={cli.action_skip}, action_horizon={cli.action_horizon}")
    print(f"[rerank-server] pred_horizon={pred_horizon}, obs_horizon={obs_horizon}")

    # Noise scheduler
    if cli.scheduler == "ddim":
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        default_inference_steps = 20
    else:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        default_inference_steps = num_train_timesteps

    num_inference_steps = cli.num_inference_steps or default_inference_steps
    print(f"[rerank-server] scheduler={cli.scheduler}, inference_steps={num_inference_steps}")

    action_skip = cli.action_skip
    action_horizon = min(cli.action_horizon, pred_horizon - action_skip)
    query_freq = action_horizon
    print(f"[rerank-server] query_freq={query_freq} (re-plan every {query_freq} steps)")

    # Server
    server = TactileACTServer(
        host=cli.host,
        port=cli.port,
        metadata={
            "protocol": "dp_foresight_rerank",
            "K": cli.K,
            "action_skip": action_skip,
            "action_horizon": action_horizon,
            "camera_names": camera_names,
        },
    )
    server.start()
    print(f"[rerank-server] listening on {cli.host}:{cli.port}")

    # Create log directory if specified
    log_dir = cli.log_dir
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        print(f"[rerank-server] logging foresight predictions to {log_dir}")

    try:
        ep = 0
        while True:
            try:
                print(f"\n[rerank-server] === episode {ep} ===")
                obs = server.recv_obs()
            except ClientDisconnected:
                continue

            obs_buffer = deque(maxlen=obs_horizon)
            marker_buffer = []
            marker_step = 0

            # Per-episode log
            episode_log = {
                "episode": ep,
                "steps": [],  # list of per-rerank-step dicts
            }

            with torch.inference_mode():
                try:
                    for step in range(cli.max_timesteps):
                        # Preprocess observation (DP: resize+crop, Foresight: raw)
                        images_dict = {}
                        images_fs = {}
                        for cam in camera_names:
                            if cam == "gelsight":
                                continue
                            raw = obs["images"][cam]
                            images_dict[cam] = preprocess_image(
                                raw, resize_tf, crop_tf
                            ).to(device)
                            images_fs[cam] = preprocess_image(
                                raw, None, None
                            ).to(device)

                        # Tactile marker
                        tac = obs["tac"]
                        side = list(tac.keys())[0]
                        side_data = tac[side]
                        marker = np.asarray(
                            side_data["marker_offset"] if isinstance(side_data, dict) else side_data,
                            dtype=np.float32
                        )
                        marker_buffer.append(marker)

                        qpos_raw = np.asarray(obs["qpos"], dtype=np.float32)

                        # Build obs_cond
                        processed = {
                            "images_dict": images_dict,
                            "qpos_raw": qpos_raw,
                            "_marker_idx": marker_step,
                        }
                        marker_step += 1
                        obs_buffer.append(processed)

                        while len(obs_buffer) < obs_horizon:
                            pad = dict(processed)
                            pad["_marker_idx"] = 0
                            obs_buffer.appendleft(pad)

                        # Build observation condition
                        obs_feats = []
                        for t_idx in range(len(obs_buffer)):
                            frame = obs_buffer[t_idx]
                            vf = vision_encoder(frame["images_dict"])
                            # Marker history for tac_encoder
                            m_end = frame["_marker_idx"]
                            frames = []
                            for k in range(tac_history):
                                idx = max(0, min(m_end - tac_history + 1 + k, len(marker_buffer) - 1))
                                frames.append(marker_buffer[idx])
                            marker_seq = np.stack(frames, axis=0)
                            marker_t = torch.tensor(marker_seq, dtype=torch.float32).unsqueeze(0).to(device)
                            tf = tac_encoder(marker_t)
                            qpos_t = torch.tensor(
                                (frame["qpos_raw"] - qpos_min) / (qpos_max - qpos_min + 1e-8) * 2 - 1,
                                dtype=torch.float32
                            ).unsqueeze(0).to(device)
                            obs_feats.append(torch.cat([vf, tf, qpos_t], dim=-1))

                        obs_cond = torch.cat(obs_feats, dim=-1)  # (1, cond_dim)

                        # Generate + Rerank at query frequency
                        if step % query_freq == 0:
                            # Batch generate K candidates
                            candidates = batch_ddpm_inference(
                                noise_pred_net, noise_scheduler, obs_cond,
                                action_dim, pred_horizon,
                                num_inference_steps, cli.K, device,
                            )  # (K, pred_horizon, action_dim)

                            # Build marker window for Foresight
                            mw_frames = []
                            for k in range(tac_history):
                                idx = max(0, min(len(marker_buffer) - tac_history + k, len(marker_buffer) - 1))
                                mw_frames.append(marker_buffer[idx])
                            marker_window = torch.tensor(
                                np.stack(mw_frames, axis=0), dtype=torch.float32
                            ).unsqueeze(0).to(device)  # (1, 8, 9, 9, 2)

                            # Current qpos raw
                            qpos_raw_t = torch.tensor(
                                qpos_raw, dtype=torch.float32
                            ).unsqueeze(0).to(device)

                            # Rerank (use raw-size images for Foresight)
                            progress = step / cli.max_timesteps
                            best_idx, scores, z_pred, z_current, marker_pred, marker_cur_hat = foresight_rerank(
                                candidates, foresight, fs_norm, fs_config,
                                images_fs, marker_window, qpos_raw_t,
                                action_min_t, action_max_t, device,
                                scorer=scorer, scoring_mode=cli.scoring_mode,
                                progress=progress,
                            )

                            best_actions = candidates[best_idx].unsqueeze(0)  # (1, pred_horizon, 7)

                            # Log foresight predictions
                            if log_dir:
                                step_log = {
                                    "step": step,
                                    "progress": progress,
                                    # Current marker_offset (raw, from sensor)
                                    "marker_current": marker_buffer[-1].copy(),
                                    # Marker window used as foresight input (tac_history, 9, 9, 2)
                                    "marker_window": np.stack(
                                        [marker_buffer[max(0, min(len(marker_buffer) - tac_history + i, len(marker_buffer) - 1))]
                                         for i in range(tac_history)], axis=0
                                    ).copy(),
                                    # All K candidate actions (K, pred_horizon, action_dim) in [-1,1]
                                    "candidates": candidates.cpu().numpy().copy(),
                                    # Scores for all K candidates
                                    "scores": scores.cpu().numpy().copy(),
                                    # Selected candidate index
                                    "best_idx": best_idx,
                                    # z_pred latent for all K candidates (K, 144)
                                    "z_pred": z_pred.cpu().numpy().copy(),
                                    # z_current latent (144,)
                                    "z_current": z_current.cpu().numpy().copy(),
                                    # Decoded marker predictions for all K candidates (K, 9, 9, 2)
                                    "marker_pred": marker_pred.cpu().numpy().copy(),
                                    # Decoded current marker from z_current (1, 9, 9, 2)
                                    "marker_cur_hat": marker_cur_hat.cpu().numpy().copy(),
                                    # Actual executed action (in raw joint space)
                                    "executed_action": action.copy(),
                                }
                                episode_log["steps"].append(step_log)

                            score_info = (f"scores: best={scores[best_idx]:.3f}, "
                                          f"worst={scores.min():.3f}, "
                                          f"spread={scores.max()-scores.min():.3f}")
                            if step % (query_freq * 5) == 0:
                                print(f"  step {step}: {score_info}")

                        # Execute action
                        raw = best_actions[:, action_skip + step % query_freq]
                        raw_np = raw.squeeze(0).cpu().numpy()
                        action = (raw_np + 1) / 2 * (action_max - action_min) + action_min
                        action = action.astype(np.float32)

                        server.send_action({
                            "actions": action[None, :],
                            "step": step,
                        })

                        if step + 1 < cli.max_timesteps:
                            obs = server.recv_obs()

                except ClientDisconnected:
                    print(f"[rerank-server] client disconnected at step {step}")

            # Save episode log
            if log_dir and episode_log["steps"]:
                # Also save full marker trajectory (every step, not just rerank steps)
                episode_log["marker_trajectory"] = np.stack(marker_buffer, axis=0)  # (T_total, 9, 9, 2)
                log_path = os.path.join(log_dir, f"episode_{ep:04d}.pkl")
                with open(log_path, 'wb') as f:
                    pickle.dump(episode_log, f)
                print(f"[rerank-server] saved {len(episode_log['steps'])} rerank steps, "
                      f"{len(marker_buffer)} marker frames → {log_path}")

            ep += 1
    except KeyboardInterrupt:
        print("\n[rerank-server] shutting down")
    finally:
        server.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
