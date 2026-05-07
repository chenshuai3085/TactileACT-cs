"""
Precompute CQF training data using real DP-generated candidates.

For each sampled frame:
  1. DP generates K=16 candidate action trajectories (DDPM sampling)
  2. Foresight predicts z_pred (144-dim) for each candidate
  3. L1 distance to expert → normalized to [0, 1] label per group
  4. Save: {qpos, action_chunk, z_cur, z_pred, label, group_id}

Usage:
  python TFAC_V5/precompute_cqf_dp_candidates.py \
    --dp_ckpt /home/chenshuai/Project/output/dp_tac_vae_shift4_0414/dp_best.pth \
    --dp_config /home/chenshuai/Project/output/dp_tac_vae_shift4_0414/config.json \
    --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full/foresight_best.ckpt \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_full \
    --data_dirs /home/chenshuai/data/dataset/0414_truncated \
    --output_dir /home/chenshuai/Project/output/cqf_dp_candidates \
    --K 16 --frame_stride 3 --device cuda:0
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import h5py
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'diffusion'))
sys.path.insert(0, os.path.join(ROOT, 'TFAC_V5'))

from dp_reranking import (
    OfficialVisionEncoder, FrozenTactileVAEEncoder, EMAModel,
    TacDreamReranker,
)
from network import ConditionalUnet1D
from pretrain_latent_foresight import LatentForesightPretrainModel


class DPForesightGenerator:
    """DP + Foresight for generating candidates and computing z_pred (no CQF needed)."""

    def __init__(self, dp_config_path, dp_ckpt_path,
                 foresight_ckpt_path, foresight_dir=None,
                 device="cuda:0", K=16):
        self.device = torch.device(device)
        self.K = K

        with open(dp_config_path) as f:
            self.dp_config = json.load(f)

        self.dp_variant = self.dp_config.get("variant", "tactile_vae_frozen")
        self._load_dp(dp_ckpt_path)
        self._load_foresight(foresight_ckpt_path, foresight_dir)

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

        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
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
            print(f"DP loaded (EMA, epoch {ckpt.get('epoch', '?')})")
        else:
            self.noise_pred_net.load_state_dict(ckpt["noise_pred_net"])
            self.dp_vision.load_state_dict(ckpt["vision_encoder"])
            print(f"DP loaded (epoch {ckpt.get('epoch', '?')})")
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
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        self.foresight.eval()

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
        self.vae_window = config.get('tactile_vae_window', 8) or 8

    def _dp_unnorm_action(self, action_norm):
        return (action_norm + 1) / 2 * (self.dp_action_max - self.dp_action_min) + self.dp_action_min

    def _dp_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.dp_qpos_min) / (self.dp_qpos_max - self.dp_qpos_min + 1e-8) * 2 - 1

    def _fs_norm_action(self, action_raw):
        return (action_raw - self.fs_action_mean) / self.fs_action_std

    def _fs_norm_qpos(self, qpos_raw):
        return (qpos_raw - self.fs_qpos_mean) / self.fs_qpos_std

    def _preprocess_image(self, img_uint8, resize=None):
        img = torch.tensor(img_uint8.astype(np.float32) / 255.0).permute(2, 0, 1)
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = normalize(img)
        if resize is not None:
            img = transforms.functional.resize(img, resize)
        return img

    def _get_marker_window(self, marker_all, t, normalize=True):
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.vae_window):
            idx = max(0, t - (self.vae_window - 1 - i))
            idx = min(idx, T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        window = np.stack(frames)
        if normalize:
            mo_mean = np.array(FrozenTactileVAEEncoder.TAC_MEAN).reshape(1, 1, 1, 2)
            mo_std = np.array(FrozenTactileVAEEncoder.TAC_STD).reshape(1, 1, 1, 2)
            window = (window - mo_mean) / mo_std
        return torch.tensor(window, dtype=torch.float32)

    def _get_dp_marker_history(self, marker_all, t):
        T_total = marker_all.shape[0]
        frames = []
        for i in range(self.dp_tac_history):
            idx = max(0, t - (self.dp_tac_history - 1 - i))
            idx = min(idx, T_total - 1)
            frames.append(marker_all[idx].astype(np.float32))
        return torch.tensor(np.stack(frames), dtype=torch.float32)

    @torch.no_grad()
    def build_obs_cond(self, images_obs, qpos_obs, marker_hists):
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
    def compute_z_cur_and_preds(self, marker_window, foresight_images,
                                 candidates_raw, qpos_raw):
        """Compute z_cur and z_pred for K candidates.

        Returns:
            z_cur: (144,) numpy
            z_preds: (K, 144) numpy
        """
        K = candidates_raw.shape[0]

        marker_win = marker_window.unsqueeze(0).to(self.device)
        z_cur_raw, _ = self.foresight.tactile_vae.encode_single_frame(marker_win)
        z_cur = z_cur_raw.reshape(1, -1)

        fs_images = [img.expand(K, *img.shape[1:]) for img in foresight_images]

        fs_chunk = self.foresight_chunk
        action_fs = candidates_raw[:, :fs_chunk, :]
        action_fs_norm = self._fs_norm_action(action_fs)

        if isinstance(qpos_raw, np.ndarray):
            qpos_raw_t = torch.tensor(qpos_raw, dtype=torch.float32, device=self.device)
        else:
            qpos_raw_t = qpos_raw.to(self.device)
        qpos_fs_norm = self._fs_norm_qpos(qpos_raw_t.unsqueeze(0)).expand(K, -1)

        t_hat, _, _, _, _, _ = self.foresight(
            fs_images, action_fs_norm, qpos=qpos_fs_norm)

        return z_cur.squeeze(0).cpu().numpy(), t_hat.cpu().numpy()


def find_episodes(data_dir):
    """Find all episode HDF5 files in a directory."""
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not files:
        for subdir in ["success", "bounce"]:
            sub_path = os.path.join(data_dir, subdir)
            if os.path.isdir(sub_path):
                files.extend(sorted(glob.glob(os.path.join(sub_path, "episode_*.hdf5"))))
    return files


@torch.no_grad()
def process_episode(gen, hdf5_path, frame_stride, K, cqf_chunk_size):
    """Process one episode: generate candidates for sampled frames."""
    samples = []

    with h5py.File(hdf5_path, 'r') as f:
        qpos_all = f['observations/proprio_joint'][:]
        actions_all = f['actions/joint_abs'][:]
        marker_all = f['observations/tac/left/marker_offset'][:]
        T = qpos_all.shape[0]

        has_global = 'observations/images/global' in f
        has_wrist = 'observations/images/wrist' in f

        min_t = gen.obs_horizon - 1
        max_t = T - gen.action_shift - gen.pred_horizon

        if max_t <= min_t:
            return samples

        frame_indices = list(range(min_t, max_t, frame_stride))

        for t in frame_indices:
            # Build obs_cond
            images_obs = []
            qpos_obs = []
            marker_hists = []
            for step in range(gen.obs_horizon):
                t_obs = max(0, t - gen.obs_horizon + 1 + step)
                obs_dict = {}
                for cam in gen.dp_camera_names:
                    key = f'observations/images/{cam}'
                    if key in f:
                        obs_dict[cam] = f[key][t_obs]
                    else:
                        obs_dict[cam] = np.zeros(
                            (gen.dp_input_h, gen.dp_input_w, 3), dtype=np.uint8)
                images_obs.append(obs_dict)
                qpos_obs.append(qpos_all[t_obs])
                marker_hists.append(gen._get_dp_marker_history(marker_all, t_obs))

            obs_cond = gen.build_obs_cond(images_obs, qpos_obs, marker_hists)

            # Generate K candidates
            candidates_raw = gen.generate_candidates(obs_cond, K=K)

            # Expert action (aligned with action_shift)
            expert_start = t + gen.action_shift
            expert = actions_all[expert_start:expert_start + gen.pred_horizon]
            if len(expert) < gen.pred_horizon:
                expert = np.pad(expert,
                                ((0, gen.pred_horizon - len(expert)), (0, 0)), mode='edge')
            expert_t = torch.tensor(expert, dtype=torch.float32, device=gen.device)

            # L1 labels
            candidates_np = candidates_raw.cpu().numpy()
            l1_per_k = np.abs(candidates_np - expert[None]).mean(axis=(1, 2))
            l1_min, l1_max = l1_per_k.min(), l1_per_k.max()
            if l1_max - l1_min > 1e-8:
                labels = 1.0 - (l1_per_k - l1_min) / (l1_max - l1_min)
            else:
                labels = np.ones(K, dtype=np.float32)

            # Foresight: z_cur + z_pred
            marker_window = gen._get_marker_window(marker_all, t, normalize=True)

            foresight_imgs = []
            for cam_name in ['global', 'wrist']:
                key = f'observations/images/{cam_name}'
                if key in f:
                    img = gen._preprocess_image(f[key][t]).unsqueeze(0).to(gen.device)
                else:
                    img = torch.zeros(1, 3, gen.dp_input_h, gen.dp_input_w,
                                      device=gen.device)
                foresight_imgs.append(img)
            marker_window_dev = marker_window.unsqueeze(0).to(gen.device)
            foresight_imgs.append(marker_window_dev)

            z_cur_np, z_preds_np = gen.compute_z_cur_and_preds(
                marker_window, foresight_imgs, candidates_raw, qpos_all[t])

            # Build CQF action_chunk (pad to cqf_chunk_size if needed)
            for k in range(K):
                action_chunk = candidates_np[k]
                if action_chunk.shape[0] < cqf_chunk_size:
                    action_chunk = np.pad(
                        action_chunk,
                        ((0, cqf_chunk_size - action_chunk.shape[0]), (0, 0)),
                        mode='edge')
                elif action_chunk.shape[0] > cqf_chunk_size:
                    action_chunk = action_chunk[:cqf_chunk_size]

                samples.append({
                    "qpos": qpos_all[t].astype(np.float32),
                    "action_chunk": action_chunk.astype(np.float32),
                    "z_cur": z_cur_np.astype(np.float32),
                    "z_pred": z_preds_np[k].astype(np.float32),
                    "label": float(labels[k]),
                })

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_ckpt", type=str, required=True)
    parser.add_argument("--dp_config", type=str, required=True)
    parser.add_argument("--foresight_ckpt", type=str, required=True)
    parser.add_argument("--foresight_dir", type=str, default=None)
    parser.add_argument("--data_dirs", type=str, required=True,
                        help="Comma-separated data directories")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--cqf_chunk_size", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save checkpoint every N episodes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Precompute CQF DP Candidates")
    print("=" * 60)
    print(f"  K={args.K}, frame_stride={args.frame_stride}, "
          f"cqf_chunk_size={args.cqf_chunk_size}")

    gen = DPForesightGenerator(
        dp_config_path=args.dp_config,
        dp_ckpt_path=args.dp_ckpt,
        foresight_ckpt_path=args.foresight_ckpt,
        foresight_dir=args.foresight_dir,
        device=args.device,
        K=args.K,
    )
    print(f"  pred_horizon={gen.pred_horizon}, action_shift={gen.action_shift}, "
          f"obs_horizon={gen.obs_horizon}")

    data_dirs = [d.strip() for d in args.data_dirs.split(",")]
    all_episodes = []
    for d in data_dirs:
        eps = find_episodes(d)
        print(f"  {d}: {len(eps)} episodes")
        all_episodes.extend(eps)

    print(f"\nTotal episodes: {len(all_episodes)}")

    all_samples = []
    group_id = 0
    t_start = time.time()

    for ep_idx, hdf5_path in enumerate(tqdm(all_episodes, desc="Episodes")):
        try:
            ep_samples = process_episode(
                gen, hdf5_path, args.frame_stride, args.K, args.cqf_chunk_size)

            n_groups = len(ep_samples) // args.K
            for s in ep_samples:
                s["group_id"] = group_id + (ep_samples.index(s) // args.K)
            # Fix: assign group_id properly
            for i, s in enumerate(ep_samples):
                s["group_id"] = group_id + i // args.K
            group_id += n_groups

            all_samples.extend(ep_samples)

        except Exception as e:
            print(f"\n  Error on {hdf5_path}: {e}")
            import traceback
            traceback.print_exc()
            continue

        if (ep_idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = (ep_idx + 1) / elapsed
            eta = (len(all_episodes) - ep_idx - 1) / rate
            n_groups_total = len(all_samples) // args.K
            print(f"\n  [{ep_idx+1}/{len(all_episodes)}] "
                  f"{len(all_samples)} samples ({n_groups_total} groups), "
                  f"{rate:.1f} ep/s, ETA {eta/60:.0f}min")

        if args.save_every > 0 and (ep_idx + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"samples_ckpt_{ep_idx+1}.pt")
            torch.save(all_samples, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    elapsed = time.time() - t_start
    n_total = len(all_samples)
    n_groups_total = n_total // args.K
    print(f"\n{'='*60}")
    print(f"Done! {n_total} samples ({n_groups_total} groups) in {elapsed/60:.1f} min")

    # Label statistics
    all_labels = np.array([s["label"] for s in all_samples])
    print(f"  Label stats: mean={all_labels.mean():.3f}, std={all_labels.std():.3f}, "
          f"min={all_labels.min():.3f}, max={all_labels.max():.3f}")

    # Train/val split (by group)
    rng = np.random.RandomState(args.seed)
    group_indices = np.arange(n_groups_total)
    rng.shuffle(group_indices)
    n_val_groups = int(n_groups_total * args.val_ratio)
    val_group_set = set(group_indices[:n_val_groups].tolist())

    train_samples = []
    val_samples = []
    for s in all_samples:
        if s["group_id"] in val_group_set:
            val_samples.append(s)
        else:
            train_samples.append(s)

    print(f"\n  Train: {len(train_samples)} samples ({len(train_samples)//args.K} groups)")
    print(f"  Val:   {len(val_samples)} samples ({len(val_samples)//args.K} groups)")

    train_path = os.path.join(args.output_dir, "train_dp_samples.pt")
    val_path = os.path.join(args.output_dir, "val_dp_samples.pt")
    torch.save(train_samples, train_path)
    torch.save(val_samples, val_path)
    print(f"\n  Saved: {train_path} ({os.path.getsize(train_path)/1e6:.1f} MB)")
    print(f"         {val_path} ({os.path.getsize(val_path)/1e6:.1f} MB)")

    meta = {
        "dp_ckpt": args.dp_ckpt,
        "dp_config": args.dp_config,
        "foresight_ckpt": args.foresight_ckpt,
        "data_dirs": data_dirs,
        "K": args.K,
        "frame_stride": args.frame_stride,
        "cqf_chunk_size": args.cqf_chunk_size,
        "pred_horizon": gen.pred_horizon,
        "action_shift": gen.action_shift,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_groups_train": len(train_samples) // args.K,
        "n_groups_val": len(val_samples) // args.K,
        "tac_dim": 144,
    }
    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()
