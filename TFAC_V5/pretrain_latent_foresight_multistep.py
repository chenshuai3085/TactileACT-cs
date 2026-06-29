"""
TFAC_V5 multi-step latent foresight pretraining.

This is the clean multi-step path for tactile consequence prediction:

    current V/T observation + future action/state chunk
        -> future tactile VAE latent sequence z[t+1:t+H]

Loss:
    L = L_latent + lambda_marker * L_marker + lambda_delta * L_delta

No task-quality labels, no CVAE sampling, no G-TCL.  The goal is first to make
future tactile prediction itself accurate and temporally faithful.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from detr.models.backbone import Backbone, Joiner, PositionEmbeddingSine
from TFAC_V5.dataset import ForesightEpisodicDataset
from TFAC_V5.foresight_multistep import MultiStepSpatialForesightTransformer
from TFAC_V5.tactile_vae import TactileVAE
from utils import set_seed


class MultiStepLatentForesightModel(nn.Module):
    """Frozen vision/TactileVAE encoders plus trainable multi-step foresight."""

    def __init__(self, camera_names, cam_backbone_mapping,
                 hidden_dim=512, state_dim=7,
                 foresight_layers=3, foresight_nheads=8,
                 foresight_dim_feedforward=2048, dropout=0.1,
                 tactile_mode="marker", max_history=8,
                 predict_horizon=16,
                 tactile_vae_ckpt=None,
                 tactile_vae_latent_dim=16):
        super().__init__()
        if tactile_mode != "marker":
            raise ValueError("Multi-step latent foresight currently expects tactile_mode='marker'")
        if predict_horizon <= 1:
            raise ValueError("Use predict_horizon > 1 for multi-step foresight")

        self.camera_names = camera_names
        self.cam_backbone_mapping = cam_backbone_mapping
        self.hidden_dim = hidden_dim
        self.tactile_mode = tactile_mode
        self.predict_horizon = predict_horizon
        self.tactile_vae_latent_dim = tactile_vae_latent_dim
        self.n_tactile_spatial = 9

        n_steps = hidden_dim // 2
        position_embedding = PositionEmbeddingSine(n_steps, normalize=True)
        backbone = Backbone(name="resnet18", train_backbone=False,
                            return_interm_layers=False, dilation=False)
        backbone_model = Joiner(backbone, position_embedding)
        backbone_model.num_channels = backbone.num_channels
        self.backbone = nn.ModuleList([backbone_model])
        self.backbone.requires_grad_(False)
        self.input_proj = nn.Conv2d(backbone_model.num_channels, hidden_dim, kernel_size=1)

        self.tactile_vae = TactileVAE(latent_dim=tactile_vae_latent_dim, temporal_window=8)
        if tactile_vae_ckpt and os.path.exists(tactile_vae_ckpt):
            ckpt = torch.load(tactile_vae_ckpt, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.tactile_vae.load_state_dict(state)
            print(f"Loaded TactileVAE from: {tactile_vae_ckpt}")
        self.tactile_vae.requires_grad_(False)

        self.tac_latent_proj = nn.Linear(tactile_vae_latent_dim, hidden_dim)
        self.tac_spatial_pos_embed = nn.Parameter(torch.randn(9, 1, hidden_dim) * 0.02)

        n_cams = len([c for c in camera_names if c not in ("gelsight", "blank")])
        self.cam_embed = nn.Parameter(torch.randn(n_cams, 1, 1, hidden_dim) * 0.02)

        self.foresight = MultiStepSpatialForesightTransformer(
            d_model=hidden_dim,
            action_dim=state_dim,
            num_layers=foresight_layers,
            nhead=foresight_nheads,
            dim_feedforward=foresight_dim_feedforward,
            dropout=dropout,
            latent_dim=tactile_vae_latent_dim,
            n_tactile_spatial=self.n_tactile_spatial,
            predict_horizon=predict_horizon,
            max_history=max_history,
            state_dim=state_dim,
        )

    def _encode_images(self, images):
        all_cam_features = []
        n_vision = 0
        n_tactile = 0
        vis_cam_idx = 0
        z_current = None

        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name == "gelsight":
                with torch.no_grad():
                    z_t, _ = self.tactile_vae.encode_single_frame(images[cam_id])
                B = z_t.size(0)
                C = self.tactile_vae_latent_dim
                z_current = z_t.reshape(B, -1)
                z_tokens = z_t.reshape(B, C, 9).permute(2, 0, 1)
                proj = self.tac_latent_proj(z_tokens) + self.tac_spatial_pos_embed
                n_tactile += 9
            else:
                features, pos = self.backbone[self.cam_backbone_mapping[cam_name]](images[cam_id])
                features = features[0]
                pos = pos[0]
                proj = self.input_proj(features).flatten(2)
                pos = pos.flatten(2)
                n_tokens = proj.size(2)
                n_vision += n_tokens
                proj = proj.permute(2, 0, 1) + pos.permute(2, 0, 1)
                if cam_name != "blank":
                    proj = proj + self.cam_embed[vis_cam_idx]
                    vis_cam_idx += 1
            all_cam_features.append(proj)

        src = torch.cat(all_cam_features, dim=0)
        return src, n_vision, n_tactile, z_current

    def _encode_future_tactile(self, future_images):
        for cam_id, cam_name in enumerate(self.camera_names):
            if cam_name != "gelsight":
                continue
            future_marker = future_images[cam_id]
            B = future_marker.shape[0]
            H = min(future_marker.shape[1], self.predict_horizon)
            z_gt = []
            with torch.no_grad():
                if future_marker.dim() == 6:
                    # (B, H, T_vae, 9, 9, 2)
                    for h in range(H):
                        z_h, _ = self.tactile_vae.encode_single_frame(future_marker[:, h])
                        z_gt.append(z_h.reshape(B, -1))
                    raw_gt = future_marker[:, :H, -1]
                elif future_marker.dim() == 5:
                    # (B, H, 9, 9, 2), legacy fallback.
                    for h in range(H):
                        z_h, _ = self.tactile_vae.encode_single_frame(
                            future_marker[:, h].unsqueeze(1))
                        z_gt.append(z_h.reshape(B, -1))
                    raw_gt = future_marker[:, :H]
                else:
                    raise ValueError(f"Unsupported future marker shape: {future_marker.shape}")
            return torch.stack(z_gt, dim=1), raw_gt
        raise RuntimeError("No gelsight camera found in future_images")

    def forward(self, images, actions, future_images=None, qpos=None):
        src, n_vision, _, z_current = self._encode_images(images)
        v_tokens = src[:n_vision]
        t_tokens = src[n_vision:]
        t_hat, _, _ = self.foresight(
            v_tokens, t_tokens, actions, n_vision, proprio=qpos)

        z_gt = None
        raw_gt = None
        if future_images is not None:
            z_gt, raw_gt = self._encode_future_tactile(future_images)
            if z_gt.shape[1] != self.predict_horizon:
                t_hat = t_hat[:, :z_gt.shape[1]]
        return t_hat, z_gt, raw_gt, z_current

    def decode_latent_sequence(self, z_seq):
        B, H, D = z_seq.shape
        C = self.tactile_vae_latent_dim
        z_spatial = z_seq.reshape(B * H, C, 3, 3)
        marker = self.tactile_vae.decoder(z_spatial)
        return marker.reshape(B, H, 9, 9, 2)


def multistep_loss(t_hat, z_gt, marker_gt, model,
                   lambda_marker=0.3, lambda_delta=0.5,
                   final_weight=1.0):
    """Contact-dynamics preserving multi-step tactile foresight loss."""
    if t_hat.shape != z_gt.shape:
        raise ValueError(f"t_hat shape {t_hat.shape} != z_gt shape {z_gt.shape}")

    loss_latent_seq = F.smooth_l1_loss(t_hat, z_gt)
    loss_latent_final = F.smooth_l1_loss(t_hat[:, -1], z_gt[:, -1])
    loss_latent = loss_latent_seq + final_weight * loss_latent_final

    marker_hat = model.decode_latent_sequence(t_hat)
    loss_marker = F.smooth_l1_loss(marker_hat, marker_gt)

    if t_hat.size(1) > 1:
        pred_delta = t_hat[:, 1:] - t_hat[:, :-1]
        gt_delta = z_gt[:, 1:] - z_gt[:, :-1]
        loss_delta = F.smooth_l1_loss(pred_delta, gt_delta)
    else:
        loss_delta = torch.zeros((), device=t_hat.device)

    total = loss_latent + lambda_marker * loss_marker + lambda_delta * loss_delta
    metrics = {
        "total": total.detach().item(),
        "latent": loss_latent.detach().item(),
        "latent_seq": loss_latent_seq.detach().item(),
        "latent_final": loss_latent_final.detach().item(),
        "marker": loss_marker.detach().item(),
        "delta": loss_delta.detach().item(),
    }
    return total, metrics


def scan_episode_paths(dataset_dir):
    direct = sorted(glob.glob(os.path.join(dataset_dir, "episode_*.hdf5")))
    if direct:
        return direct
    paths = []
    for sub in sorted(os.listdir(dataset_dir)):
        sub_dir = os.path.join(dataset_dir, sub)
        if os.path.isdir(sub_dir):
            paths.extend(sorted(glob.glob(os.path.join(sub_dir, "episode_*.hdf5"))))
    return paths


def infer_meta_from_path(episode_path, overrides=None):
    import h5py
    overrides = overrides or {}
    with h5py.File(episode_path, "r") as root:
        if overrides.get("camera_names"):
            camera_names = overrides["camera_names"]
        else:
            camera_names = sorted(root["observations/images"].keys())
            if "observations/tac" in root and "gelsight" not in camera_names:
                camera_names.append("gelsight")

        proprio_key = overrides.get("proprio_key")
        if not proprio_key:
            for candidate in ["proprio_joint", "qpos", "proprio_eef"]:
                if f"observations/{candidate}" in root:
                    proprio_key = candidate
                    break
            if proprio_key is None:
                raise KeyError("Cannot infer proprio key")

        action_key = overrides.get("action_key")
        if not action_key:
            action_key = "actions/joint_abs" if "actions/joint_abs" in root else "action"

        state_dim = overrides.get("state_dim")
        if not state_dim:
            state_dim = root[f"observations/{proprio_key}"].shape[-1]

        tac_side = overrides.get("tac_side", "left")
        tac_img_key = overrides.get("tac_img_key", "img")

    return {
        "camera_names": camera_names,
        "state_dim": state_dim,
        "proprio_key": proprio_key,
        "action_key": action_key,
        "tac_side": tac_side,
        "tac_img_key": tac_img_key,
    }


def compute_norm_stats(episode_paths, proprio_key, action_key, tactile_mode, tac_side,
                       max_episodes=200):
    import h5py
    if len(episode_paths) > max_episodes:
        ids = np.linspace(0, len(episode_paths) - 1, max_episodes, dtype=int)
        sample_paths = [episode_paths[i] for i in ids]
    else:
        sample_paths = episode_paths

    qpos_list, action_list, marker_list = [], [], []
    for path in sample_paths:
        with h5py.File(path, "r") as root:
            qpos_list.append(root[f"observations/{proprio_key}"][()])
            action_list.append(root[action_key][()])
            marker_path = f"observations/tac/{tac_side}/marker_offset"
            if tactile_mode == "marker" and marker_path in root:
                marker_list.append(root[marker_path][()])

    qpos = np.concatenate(qpos_list, axis=0)
    action = np.concatenate(action_list, axis=0)
    stats = {
        "qpos_mean": qpos.mean(axis=0).astype(np.float32),
        "qpos_std": np.clip(qpos.std(axis=0), 1e-4, None).astype(np.float32),
        "action_mean": action.mean(axis=0).astype(np.float32),
        "action_std": np.clip(action.std(axis=0), 1e-4, None).astype(np.float32),
    }
    if marker_list:
        marker = np.concatenate(marker_list, axis=0)
        stats["marker_offset_mean"] = marker.mean(axis=(0, 1, 2)).astype(np.float32)
        stats["marker_offset_std"] = np.clip(
            marker.std(axis=(0, 1, 2)), 1e-4, None).astype(np.float32)
    return stats


def resolve_preload(args, episode_paths):
    preload_cfg = args.get("preload", "auto")
    if isinstance(preload_cfg, bool):
        return preload_cfg
    if str(preload_cfg).lower() in ("true", "yes", "1"):
        return True
    if str(preload_cfg).lower() in ("false", "no", "0"):
        return False

    total_bytes = sum(os.path.getsize(p) for p in episode_paths)
    total_gb = total_bytes / (1024 ** 3)
    max_gb = args.get("max_preload_gb", None)
    if max_gb is None:
        mem_available_gb = None
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        mem_available_gb = int(line.split()[1]) / (1024 ** 2)
                        break
        except OSError:
            mem_available_gb = None

        if mem_available_gb is None:
            max_gb = 48.0
            reason = "fallback threshold"
        else:
            frac = float(args.get("preload_mem_fraction", 0.75))
            max_gb = mem_available_gb * frac
            reason = f"{frac:.0%} of MemAvailable={mem_available_gb:.1f} GB"
    else:
        max_gb = float(max_gb)
        reason = "config threshold"

    preload = total_gb <= max_gb
    print(f"Preload auto: dataset files {total_gb:.1f} GB, "
          f"limit {max_gb:.1f} GB ({reason}) -> {preload}")
    return preload


def to_device(batch, device):
    all_cam_images, qpos_data, action_data, is_pad, future_cam_images, history_cam_images = batch
    images = [img.to(device, non_blocking=True) for img in all_cam_images]
    qpos = qpos_data.to(device, non_blocking=True)
    actions = action_data.to(device, non_blocking=True)
    future_images = [img.to(device, non_blocking=True) for img in future_cam_images]
    return images, qpos, actions, future_images


def aggregate_epoch(metrics_sum, metrics, batch_size):
    for key, value in metrics.items():
        metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value) * batch_size


def average_metrics(metrics_sum, count):
    return {key: value / max(count, 1) for key, value in metrics_sum.items()}


def plot_history(history, path):
    names = ["total", "latent", "marker", "delta", "latent_final"]
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4))
    for ax, name in zip(axes, names):
        train_key = f"train_{name}"
        val_key = f"val_{name}"
        if train_key in history:
            ax.plot(history[train_key], label="train", alpha=0.8)
        if val_key in history:
            ax.plot(history[val_key], label="val", alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main(args):
    seed = int(args.get("seed", 42))
    set_seed(seed)

    gpu = int(args.get("gpu", -1))
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset_dirs = args.get("dataset_dirs")
    if dataset_dirs:
        episode_paths = []
        for dataset_dir in dataset_dirs:
            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(dataset_dir)
            paths = scan_episode_paths(dataset_dir)
            print(f"{dataset_dir}: {len(paths)} episodes")
            episode_paths.extend(paths)
        dataset_root = os.path.dirname(dataset_dirs[0])
    else:
        dataset_root = args["dataset_dir"]
        episode_paths = scan_episode_paths(dataset_root)
        print(f"{dataset_root}: {len(episode_paths)} episodes")
    if not episode_paths:
        raise RuntimeError("No episode_*.hdf5 files found")
    print(f"Total episodes: {len(episode_paths)}")

    meta = infer_meta_from_path(episode_paths[0], args)
    camera_names = meta["camera_names"]
    state_dim = meta["state_dim"]
    proprio_key = meta["proprio_key"]
    action_key = meta["action_key"]
    tac_side = meta["tac_side"]
    tac_img_key = meta["tac_img_key"]
    tactile_mode = args.get("tactile_mode", "marker")
    print(f"Meta: cameras={camera_names}, state_dim={state_dim}, "
          f"proprio={proprio_key}, action={action_key}, tac_side={tac_side}")

    norm_stats = compute_norm_stats(
        episode_paths, proprio_key, action_key, tactile_mode, tac_side,
        max_episodes=int(args.get("norm_max_episodes", 200)))
    vae_stats_path = args.get("tactile_vae_stats")
    if vae_stats_path and os.path.exists(vae_stats_path):
        with open(vae_stats_path) as f:
            vae_stats = json.load(f)
        norm_stats["marker_offset_mean"] = np.array(vae_stats["mean"], dtype=np.float32)
        norm_stats["marker_offset_std"] = np.array(vae_stats["std"], dtype=np.float32)
        print(f"Loaded TactileVAE marker stats from {vae_stats_path}")

    save_dir = args.get("save_dir", "/home/chenshuai/Project/output/foresight_ckpt")
    model_name = args.get("name", "latent_foresight_multistep")
    ckpt_dir = os.path.join(save_dir, model_name)
    if os.path.exists(ckpt_dir):
        i = 0
        while os.path.exists(f"{ckpt_dir}_{i}"):
            i += 1
        ckpt_dir = f"{ckpt_dir}_{i}"
        print(f"Output exists, using {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=False)

    args_to_save = {
        **args,
        **meta,
        "num_episodes": len(episode_paths),
        "norm_stats": {k: v.tolist() if hasattr(v, "tolist") else v
                       for k, v in norm_stats.items()},
    }
    with open(os.path.join(ckpt_dir, "args.json"), "w") as f:
        json.dump(args_to_save, f, indent=4)
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(norm_stats, f)

    shuffled = np.random.permutation(episode_paths).tolist()
    train_ratio = float(args.get("train_ratio", 0.9))
    n_train = max(1, int(len(shuffled) * train_ratio))
    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:]
    if not val_paths:
        val_paths = train_paths[-1:]
        train_paths = train_paths[:-1]
    print(f"Train episodes: {len(train_paths)}, Val episodes: {len(val_paths)}")

    preload = resolve_preload(args, episode_paths)
    use_state_traj = bool(args.get("use_state_trajectory", True))
    if use_state_traj:
        print("Conditioning on future qpos trajectory as action chunk")

    chunk_size = int(args.get("chunk_size", 16))
    foresight_horizon = int(args.get("foresight_horizon", 16))
    predict_horizon = int(args.get("predict_horizon", foresight_horizon))
    future_offset = int(args.get("future_offset", 1))
    temporal_stride = int(args.get("temporal_stride", 1))
    if temporal_stride < 1:
        raise ValueError(f"temporal_stride must be >= 1, got {temporal_stride}")
    action_offset = int(args.get(
        "action_offset",
        future_offset if use_state_traj else 0,
    ))
    action_last = action_offset + (chunk_size - 1) * temporal_stride
    future_last = future_offset + (foresight_horizon - 1) * temporal_stride
    print(
        "Temporal alignment: "
        f"obs[t] + condition[t+{action_offset}:step{temporal_stride}:t+{action_last}] "
        f"-> tactile[t+{future_offset}:step{temporal_stride}:t+{future_last}]"
    )
    if predict_horizon != foresight_horizon:
        print(f"Warning: predict_horizon={predict_horizon}, "
              f"foresight_horizon={foresight_horizon}; using min length in forward")

    train_dataset = ForesightEpisodicDataset(
        train_paths, dataset_root, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode,
        history_len=int(args.get("history_len", 1)),
        tactile_vae_window=int(args.get("tactile_vae_window", 8)),
        preload=preload,
        use_state_trajectory=use_state_traj,
        future_offset=future_offset,
        action_offset=action_offset,
        temporal_stride=temporal_stride,
    )
    val_dataset = ForesightEpisodicDataset(
        val_paths, dataset_root, camera_names, norm_stats,
        chunk_size=chunk_size, foresight_horizon=foresight_horizon,
        proprio_key=proprio_key, action_key=action_key,
        tac_side=tac_side, tac_img_key=tac_img_key,
        tactile_mode=tactile_mode,
        history_len=int(args.get("history_len", 1)),
        tactile_vae_window=int(args.get("tactile_vae_window", 8)),
        preload=preload,
        use_state_trajectory=use_state_traj,
        future_offset=future_offset,
        action_offset=action_offset,
        temporal_stride=temporal_stride,
    )

    num_workers_cfg = args.get("num_workers", "auto")
    if str(num_workers_cfg).lower() == "auto":
        num_workers = 0 if preload else int(args.get("disk_num_workers", 8))
    else:
        num_workers = int(num_workers_cfg)
    print(f"DataLoader workers: {num_workers}")
    batch_size = int(args.get("batch_size", 32))
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available())

    cam_backbone_mapping = {cam_name: 0 for cam_name in camera_names}
    model = MultiStepLatentForesightModel(
        camera_names=camera_names,
        cam_backbone_mapping=cam_backbone_mapping,
        hidden_dim=int(args.get("hidden_dim", 512)),
        state_dim=state_dim,
        foresight_layers=int(args.get("foresight_layers", 3)),
        foresight_nheads=int(args.get("foresight_nheads", 8)),
        foresight_dim_feedforward=int(args.get("foresight_dim_feedforward", 2048)),
        dropout=float(args.get("dropout", 0.1)),
        tactile_mode=tactile_mode,
        max_history=int(args.get("max_history", 8)),
        predict_horizon=predict_horizon,
        tactile_vae_ckpt=args.get("tactile_vae_ckpt"),
        tactile_vae_latent_dim=int(args.get("tactile_vae_latent_dim", 16)),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(args.get("lr", 4e-5)),
        weight_decay=float(args.get("weight_decay", 1e-4)))

    lambda_marker = float(args.get("lambda_marker", 0.3))
    lambda_delta = float(args.get("lambda_delta", 0.5))
    final_weight = float(args.get("final_weight", 1.0))
    num_epochs = int(args.get("num_epochs", 500))
    log_interval = int(args.get("log_interval", 5))
    save_interval = int(args.get("save_interval", 50))

    history = {
        "train_total": [], "val_total": [],
        "train_latent": [], "val_latent": [],
        "train_latent_seq": [], "val_latent_seq": [],
        "train_latent_final": [], "val_latent_final": [],
        "train_marker": [], "val_marker": [],
        "train_delta": [], "val_delta": [],
    }

    best_val = float("inf")
    best_epoch = -1
    pbar = tqdm(range(num_epochs), desc="multistep foresight")
    for epoch in pbar:
        model.train()
        model.backbone.eval()
        model.tactile_vae.eval()
        train_sum = {}
        train_count = 0
        for batch in train_loader:
            images, qpos, actions, future_images = to_device(batch, device)
            t_hat, z_gt, marker_gt, _ = model(
                images, actions, future_images=future_images, qpos=qpos)
            loss, metrics = multistep_loss(
                t_hat, z_gt, marker_gt, model,
                lambda_marker=lambda_marker,
                lambda_delta=lambda_delta,
                final_weight=final_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.get("grad_clip", 1.0)))
            optimizer.step()

            bs = actions.size(0)
            aggregate_epoch(train_sum, metrics, bs)
            train_count += bs

        train_metrics = average_metrics(train_sum, train_count)

        model.eval()
        val_sum = {}
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                images, qpos, actions, future_images = to_device(batch, device)
                t_hat, z_gt, marker_gt, _ = model(
                    images, actions, future_images=future_images, qpos=qpos)
                _, metrics = multistep_loss(
                    t_hat, z_gt, marker_gt, model,
                    lambda_marker=lambda_marker,
                    lambda_delta=lambda_delta,
                    final_weight=final_weight)
                bs = actions.size(0)
                aggregate_epoch(val_sum, metrics, bs)
                val_count += bs
        val_metrics = average_metrics(val_sum, val_count)

        for key in ["total", "latent", "latent_seq", "latent_final", "marker", "delta"]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])

        is_best = val_metrics["total"] < best_val
        if is_best:
            best_val = val_metrics["total"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "foresight_best.ckpt"))

        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            print(
                f"\nEpoch {epoch}: "
                f"train={train_metrics['total']:.4f} "
                f"(lat={train_metrics['latent']:.4f}, marker={train_metrics['marker']:.4f}, "
                f"delta={train_metrics['delta']:.4f}) "
                f"val={val_metrics['total']:.4f} "
                f"(lat={val_metrics['latent']:.4f}, marker={val_metrics['marker']:.4f}, "
                f"delta={val_metrics['delta']:.4f}) "
                f"best=epoch {best_epoch}, {best_val:.4f}")

        pbar.set_postfix(
            train=f"{train_metrics['total']:.4f}",
            val=f"{val_metrics['total']:.4f}",
            lat=f"{val_metrics['latent']:.4f}",
            mrk=f"{val_metrics['marker']:.4f}",
            dlt=f"{val_metrics['delta']:.4f}",
            best=f"{best_val:.4f}",
        )

        if epoch % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"foresight_epoch_{epoch}.ckpt"))
        if epoch % log_interval == 0 or epoch == num_epochs - 1:
            plot_history(history, os.path.join(ckpt_dir, "pretrain_loss.png"))
            with open(os.path.join(ckpt_dir, "pretrain_history.pkl"), "wb") as f:
                pickle.dump(history, f)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "foresight_last.ckpt"))
    with open(os.path.join(ckpt_dir, "pretrain_history.pkl"), "wb") as f:
        pickle.dump(history, f)
    plot_history(history, os.path.join(ckpt_dir, "pretrain_loss.png"))
    print(f"\nDone. Best val total loss {best_val:.4f} at epoch {best_epoch}")
    print(f"Saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    cli = parser.parse_args()
    with open(cli.config, "r") as f:
        config = json.load(f)
    main(config)
