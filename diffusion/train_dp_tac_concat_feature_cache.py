"""Train DP from cached vision/tactile features.

This is the production-scale training path for board experiments when raw image
preload is too memory-heavy and naive lazy HDF5 image loading is too slow.

Cache format, one npz per episode:
  - vis_feat: (T, 512 * n_cameras) float16
  - tac_feat: (T, 144) float16
  - qpos: (T, qpos_dim) float32
  - action: (T, action_dim) float32

The DP global condition stays compatible with train_dp_tac_concat.py:
  obs_cond = [vis_feat | tac_feat | qpos] * obs_horizon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TFAC_V5"))

from diffusion.network import ConditionalUnet1D
from diffusion.train_dp_tac_concat import (  # noqa: E402
    EMAModel,
    FrozenTactileVAEEncoder,
    OfficialVisionEncoder,
    get_minmax_stats,
)
from utils import set_seed  # noqa: E402


def episode_files(dataset_dir: str):
    return sorted(
        f for f in os.listdir(dataset_dir)
        if f.startswith("episode_") and f.endswith(".hdf5")
    )


def cache_path(cache_dir: str, dataset_dir: str, ep_file: str):
    parent = os.path.basename(dataset_dir.rstrip("/"))
    ep_name = os.path.splitext(ep_file)[0]
    out_dir = os.path.join(cache_dir, parent)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{ep_name}_dpfeat.npz")


def marker_history(marker, t, window):
    frames = []
    for k in range(window):
        idx = max(0, t - window + 1 + k)
        frames.append(marker[idx])
    return np.stack(frames, axis=0)


@torch.no_grad()
def build_feature_cache(args):
    dataset_dirs = [d.strip() for d in args.dataset_dir.split(",")]
    camera_names = args.camera_names.split(",")
    resize_shape = tuple(int(x) for x in args.resize_shape.split(","))
    crop_shape = tuple(int(x) for x in args.crop_shape.split(","))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    vision = OfficialVisionEncoder(camera_names).to(device).eval()
    if args.vision_ckpt:
        ckpt = torch.load(args.vision_ckpt, map_location=device)
        state = ckpt.get("ema_vis", ckpt.get("vision_encoder"))
        if state is None:
            raise KeyError(f"No vision_encoder/ema_vis in {args.vision_ckpt}")
        vision.load_state_dict(state)
        print(f"Loaded vision encoder from {args.vision_ckpt}")

    tactile = FrozenTactileVAEEncoder(
        args.vae_checkpoint,
        latent_dim=args.vae_latent_dim,
        temporal_window=args.tac_history,
    ).to(device).eval()

    resize = transforms.Resize(resize_shape)
    crop = transforms.CenterCrop(crop_shape)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    result = {"built": 0, "reused": 0, "skipped": 0, "cache_dir": args.feature_cache_dir}
    for ds_dir in dataset_dirs:
        for ep_file in tqdm(episode_files(ds_dir), desc=f"Feature cache ({os.path.basename(ds_dir)})"):
            out_path = cache_path(args.feature_cache_dir, ds_dir, ep_file)
            if os.path.exists(out_path) and not args.rebuild_cache:
                result["reused"] += 1
                continue
            ep_path = os.path.join(ds_dir, ep_file)
            try:
                with h5py.File(ep_path, "r") as f:
                    qpos = f[f"observations/{args.proprio_key}"][()].astype(np.float32)
                    action = f[args.action_key][()].astype(np.float32)
                    marker = f[f"observations/tac/{args.tac_side}/marker_offset"][()].astype(np.float32)
                    T = len(qpos)
                    vis_chunks = []
                    tac_chunks = []
                    for start in range(0, T, args.cache_batch_size):
                        end = min(start + args.cache_batch_size, T)
                        imgs = {}
                        for cam in camera_names:
                            raw = f[f"observations/images/{cam}"][start:end]
                            batch = []
                            for img in raw:
                                x = torch.from_numpy(img).float().div_(255.0).permute(2, 0, 1)
                                batch.append(crop(norm(resize(x))))
                            imgs[cam] = torch.stack(batch).to(device)
                        vis_chunks.append(vision(imgs).detach().cpu().half().numpy())

                        windows = [marker_history(marker, t, args.tac_history) for t in range(start, end)]
                        m = torch.from_numpy(np.stack(windows)).float().to(device)
                        tac_chunks.append(tactile(m).detach().cpu().half().numpy())

                np.savez_compressed(
                    out_path,
                    vis_feat=np.concatenate(vis_chunks, axis=0),
                    tac_feat=np.concatenate(tac_chunks, axis=0),
                    qpos=qpos,
                    action=action,
                    source_path=ep_path,
                )
                result["built"] += 1
            except Exception as exc:
                print(f"  [feature-cache] skip {ep_path}: {exc}")
                result["skipped"] += 1

    Path(args.feature_cache_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.feature_cache_dir, "feature_cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


class FeatureCacheDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dirs, feature_cache_dir, norm_stats, pred_horizon, obs_horizon, max_train_windows=None, seed=0):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_min = np.asarray(norm_stats["action_min"], dtype=np.float32)
        self.action_max = np.asarray(norm_stats["action_max"], dtype=np.float32)
        self.qpos_min = np.asarray(norm_stats["qpos_min"], dtype=np.float32)
        self.qpos_max = np.asarray(norm_stats["qpos_max"], dtype=np.float32)
        self.episodes = []
        for ds_dir in dataset_dirs:
            for ep_file in episode_files(ds_dir):
                p = cache_path(feature_cache_dir, ds_dir, ep_file)
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                with np.load(p) as data:
                    self.episodes.append({
                        "path": p,
                        "length": int(data["qpos"].shape[0]),
                    })
        self.indices = []
        for ep_idx, ep in enumerate(self.episodes):
            for start_ts in range(max(1, ep["length"] - pred_horizon + 1)):
                self.indices.append((ep_idx, start_ts))
        if max_train_windows is not None and len(self.indices) > max_train_windows:
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(len(self.indices), max_train_windows, replace=False))
            self.indices = [self.indices[i] for i in chosen]
        print(f"FeatureCacheDataset: {len(self.episodes)} episodes, {len(self.indices)} windows")

    def __len__(self):
        return len(self.indices)

    def _minmax_norm(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8) * 2 - 1

    def __getitem__(self, index):
        ep_idx, start_ts = self.indices[index]
        ep = self.episodes[ep_idx]
        with np.load(ep["path"]) as data:
            obs_idx = [max(0, start_ts - self.obs_horizon + 1 + k) for k in range(self.obs_horizon)]
            vis = data["vis_feat"][obs_idx].astype(np.float32)
            tac = data["tac_feat"][obs_idx].astype(np.float32)
            qpos = data["qpos"][obs_idx].astype(np.float32)
            action_end = min(start_ts + self.pred_horizon, ep["length"])
            action = data["action"][start_ts:action_end].astype(np.float32)
            if len(action) < self.pred_horizon:
                pad = np.repeat(action[-1:], self.pred_horizon - len(action), axis=0)
                action = np.concatenate([action, pad], axis=0)
        obs = np.concatenate([vis, tac, self._minmax_norm(qpos, self.qpos_min, self.qpos_max)], axis=-1)
        return {
            "obs_cond": torch.from_numpy(obs.reshape(-1)),
            "action": torch.from_numpy(self._minmax_norm(action, self.action_min, self.action_max)),
        }


def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    dataset_dirs = [d.strip() for d in args.dataset_dir.split(",")]
    norm_stats = get_minmax_stats(dataset_dirs, args.proprio_key, args.action_key)
    if args.build_feature_cache:
        build_feature_cache(args)

    dataset = FeatureCacheDataset(
        dataset_dirs,
        args.feature_cache_dir,
        norm_stats,
        args.pred_horizon,
        args.obs_horizon,
        max_train_windows=args.max_train_windows,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    sample = dataset[0]
    global_cond_dim = int(sample["obs_cond"].numel())
    action_dim = int(norm_stats["action_min"].shape[0])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        down_dims=[int(x) for x in args.down_dims.split(",")],
        kernel_size=5,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    ema_net = EMAModel(net) if not args.no_ema else None
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    config = vars(args).copy()
    config.update({
        "dataset_dirs": dataset_dirs,
        "action_dim": action_dim,
        "global_cond_dim": global_cond_dim,
        "vis_feat_dim": 512,
        "tac_feat_dim": 144,
        "n_train": len(dataset.episodes),
        "n_windows": len(dataset),
        "variant": "feature_cache_tactile_vae_frozen",
        "norm_stats": {k: v.tolist() if hasattr(v, "tolist") else v for k, v in norm_stats.items()},
    })
    with open(os.path.join(args.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    losses = []
    global_step = 0
    for epoch in range(args.epochs):
        net.train()
        ep_losses = []
        for batch in loader:
            obs_cond = batch["obs_cond"].to(device)
            action = batch["action"].to(device)
            noise = torch.randn_like(action)
            ts = torch.randint(0, args.num_train_timesteps, (action.shape[0],), device=device).long()
            noisy = scheduler.add_noise(action, noise, ts)
            pred = net(noisy, ts, global_cond=obs_cond)
            loss = nn.functional.mse_loss(pred, noise)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            lr_sched.step()
            global_step += 1
            if ema_net:
                ema_net.update(net)
            ep_losses.append(float(loss.item()))
        train_loss = float(np.mean(ep_losses))
        losses.append(train_loss)
        print(f"Ep {epoch+1}/{args.epochs} | train={train_loss:.6f} | lr={opt.param_groups[0]['lr']:.2e}")
        if (epoch + 1) % args.save_freq == 0:
            sd = {"noise_pred_net": net.state_dict(), "epoch": epoch}
            if ema_net:
                sd["ema_net"] = ema_net.state_dict()
            torch.save(sd, os.path.join(args.save_dir, f"dp_epoch{epoch+1}.pth"))

    sd = {"noise_pred_net": net.state_dict(), "epoch": args.epochs - 1}
    if ema_net:
        sd["ema_net"] = ema_net.state_dict()
    torch.save(sd, os.path.join(args.save_dir, "dp_final.pth"))
    np.save(os.path.join(args.save_dir, "train_losses.npy"), np.asarray(losses, dtype=np.float32))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--feature_cache_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--camera_names", default="global,wrist")
    parser.add_argument("--proprio_key", default="proprio_joint")
    parser.add_argument("--action_key", default="actions/joint_abs")
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--tac_history", type=int, default=8)
    parser.add_argument("--vae_checkpoint", default="/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
    parser.add_argument("--vae_latent_dim", type=int, default=16)
    parser.add_argument("--vision_ckpt", default=None)
    parser.add_argument("--resize_shape", default="240,320")
    parser.add_argument("--crop_shape", default="216,288")
    parser.add_argument("--cache_batch_size", type=int, default=64)
    parser.add_argument("--build_feature_cache", action="store_true")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--pred_horizon", type=int, default=16)
    parser.add_argument("--obs_horizon", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_train_timesteps", type=int, default=20)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--diffusion_step_embed_dim", type=int, default=64)
    parser.add_argument("--down_dims", default="128,256")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
