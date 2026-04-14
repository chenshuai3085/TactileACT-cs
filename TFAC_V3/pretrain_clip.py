"""
TFAC_V3 Stage 0: Vision-Tactile CLIP 预训练。
视觉: ResNet18 (GroupNorm) → projection → (B, clip_dim)
触觉: PointNet (marker_offset 9×9×2) → projection → (B, clip_dim)
Loss: InfoNCE 对比学习, 同时刻视觉-触觉对齐。

支持多数据源: 可以用所有可用的 episode 数据。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import Normalize
import argparse
import json

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_meta_data


# ---- Vision encoder: 直接复用 clip_pretraining_xiaomi 的 modified_resnet18 ----
try:
    from clip_pretraining_xiaomi import modified_resnet18
except ImportError:
    from clip_pretraining import modified_resnet18


# ---- Tactile encoder: 直接复用 marker_encoder.py 的 MarkerEncoderPointNet ----
from TFAC_V3.marker_encoder import MarkerEncoderPointNet


# ---- Projection heads ----

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, conditioning_dim=0, normalize=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim + conditioning_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )
        self.normalize = normalize

    def forward(self, x, conditioning=None):
        if conditioning is not None:
            x = torch.cat([x, conditioning], dim=-1)
        x = self.proj(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


class VisionProjectionHead(nn.Module):
    """ResNet feature map → pooled → projected."""
    def __init__(self, num_channels=512, out_dim=512, normalize=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(1, -1)
        self.proj = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, out_dim),
        )
        self.normalize = normalize

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.proj(x)
        if self.normalize:
            x = F.normalize(x, dim=-1)
        return x


# ---- Dataset ----

class VTClipDataset(Dataset):
    """
    从 HDF5 采样同一 episode 中 N 个间隔足够远的时刻,
    返回对应的视觉图像和 marker_offset。
    """

    def __init__(self, episode_ids, dataset_dir, camera_names,
                 tac_side="left", n_samples=5, min_distance=10,
                 proprio_key="proprio_eef", proprio_dims=3,
                 proprio_mean=None, proprio_std=None):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.tac_side = tac_side
        self.n_samples = n_samples
        self.min_distance = min_distance
        self.proprio_key = proprio_key
        self.proprio_dims = proprio_dims
        self.proprio_mean = proprio_mean  # (D,) numpy
        self.proprio_std = np.clip(proprio_std, 1e-6, np.inf) if proprio_std is not None else None

        self.rgb_normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        # Get episode lengths
        self.episode_lengths = []
        for eid in self.episode_ids:
            path = os.path.join(self.dataset_dir, f"episode_{eid}.hdf5")
            with h5py.File(path, "r", swmr=True) as f:
                T = f["observations"]["images"][camera_names[0]].shape[0]
                self.episode_lengths.append(T)

    def __len__(self):
        return len(self.episode_ids)

    def _sample_timesteps(self, T):
        """Sample n_samples timesteps with min_distance spacing."""
        n = self.n_samples
        segment_len = T / n
        timesteps = []
        for i in range(n):
            lo = int(i * segment_len)
            hi = min(int((i + 1) * segment_len), T)
            timesteps.append(int(np.random.randint(lo, hi)))
        return sorted(timesteps)

    def __getitem__(self, index):
        eid = self.episode_ids[index]
        T = self.episode_lengths[index]
        timesteps = self._sample_timesteps(T)

        path = os.path.join(self.dataset_dir, f"episode_{eid}.hdf5")
        with h5py.File(path, "r", swmr=True) as f:
            obs = f["observations"]

            all_cam_images = []  # (N, n_cams, 3, H, W)
            all_markers = []    # (N, 9, 9, 2)
            all_pos = []        # (N, proprio_dims)

            for t in timesteps:
                # Vision: all cameras
                cam_imgs = []
                for cam_name in self.camera_names:
                    img = obs["images"][cam_name][t]  # (H, W, 3) uint8
                    img = torch.tensor(img, dtype=torch.float32) / 255.0
                    img = img.permute(2, 0, 1)  # (3, H, W)
                    img = self.rgb_normalize(img)
                    cam_imgs.append(img)
                all_cam_images.append(torch.stack(cam_imgs))  # (n_cams, 3, H, W)

                # Tactile: marker_offset
                marker = obs["tac"][self.tac_side]["marker_offset"][t]  # (9, 9, 2)
                all_markers.append(torch.tensor(marker, dtype=torch.float32))

                # Proprio: eef position (x, y, z)
                pos = obs[self.proprio_key][t].astype(np.float32)[:self.proprio_dims]
                if self.proprio_mean is not None:
                    pos = (pos - self.proprio_mean[:self.proprio_dims]) / self.proprio_std[:self.proprio_dims]
                all_pos.append(torch.tensor(pos, dtype=torch.float32))

        # (N, n_cams, 3, H, W), (N, 9, 9, 2), (N, proprio_dims)
        return torch.stack(all_cam_images), torch.stack(all_markers), torch.stack(all_pos)


# ---- CLIP loss ----

def clip_loss(vision_emb, tactile_emb, temperature=0.07):
    """
    Args:
        vision_emb: (B, N, n_cams, D) — per-camera vision embeddings
        tactile_emb: (B, N, D) — tactile embeddings
    Returns:
        loss_per_cam: (n_cams,) — per-camera loss
        sim_map: (n_cams, N, N) — similarity maps (first sample)
    """
    B, N, n_cams, D = vision_emb.shape
    logit_scale = 1.0 / temperature

    vision_emb = vision_emb.permute(0, 2, 1, 3)  # (B, n_cams, N, D)
    tactile_emb = tactile_emb.unsqueeze(1)        # (B, 1, N, D)

    # Cosine similarity
    image_logits = logit_scale * vision_emb @ tactile_emb.permute(0, 1, 3, 2)  # (B, n_cams, N, N)
    tac_logits = logit_scale * tactile_emb @ vision_emb.permute(0, 1, 3, 2)    # (B, n_cams, N, N)

    sim_map = image_logits[0].detach().cpu().numpy() / logit_scale  # (n_cams, N, N)

    target = torch.eye(N, device=vision_emb.device)

    image_logits = image_logits.flatten(0, 1)  # (B*n_cams, N, N)
    tac_logits = tac_logits.flatten(0, 1)

    image_loss = F.cross_entropy(
        image_logits, target.unsqueeze(0).expand(image_logits.shape[0], -1, -1),
        reduction="none").mean(dim=1)
    tac_loss = F.cross_entropy(
        tac_logits, target.T.unsqueeze(0).expand(tac_logits.shape[0], -1, -1),
        reduction="none").mean(dim=1)

    combined = ((image_loss + tac_loss) / 2.0).view(B, n_cams)
    loss_per_cam = combined.mean(dim=0)  # (n_cams,)

    return loss_per_cam, sim_map


# ---- Training ----

def train_clip(args):
    save_dir = args['save_dir']
    model_name = args['name']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    seed = args.get('seed', 1)
    gpu = args.get('gpu', -1)
    clip_dim = args.get('clip_dim', 512)
    n_samples = args.get('n_clip_samples', 5)
    min_distance = args.get('clip_min_distance', 10)
    temperature = args.get('clip_temperature', 0.07)

    ckpt_dir = os.path.join(save_dir, model_name)
    dataset_dir = os.path.join(save_dir, 'data')
    assert os.path.exists(save_dir), f'{save_dir} does not exist.'

    meta_data = load_meta_data(dataset_dir, save_dir=save_dir, config_overrides=args)
    num_episodes = meta_data['num_episodes']
    # Only use vision cameras for CLIP (exclude gelsight)
    camera_names = [c for c in meta_data['camera_names'] if c != 'gelsight']
    tac_side = meta_data['tac_side']

    np.random.seed(seed)
    if gpu != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- Ckpt dir ---
    if os.path.exists(ckpt_dir):
        n = 0
        while os.path.exists(ckpt_dir + f'_{n}'):
            n += 1
        print(f'Warning: {ckpt_dir} exists. Using {ckpt_dir}_{n}')
        ckpt_dir = ckpt_dir + f'_{n}'
    os.makedirs(ckpt_dir)
    os.makedirs(os.path.join(ckpt_dir, 'graphs'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump({**args, **meta_data}, f, indent=4)

    # --- Compute proprio stats ---
    proprio_key = args.get('proprio_key', 'proprio_eef')
    proprio_dims = args.get('proprio_dims', 3)
    all_proprio = []
    for eid in range(num_episodes):
        path = os.path.join(dataset_dir, f"episode_{eid}.hdf5")
        if os.path.exists(path):
            with h5py.File(path, "r", swmr=True) as f:
                all_proprio.append(f["observations"][proprio_key][()].astype(np.float32))
    all_proprio = np.concatenate(all_proprio, axis=0)
    proprio_mean = all_proprio.mean(axis=0)
    proprio_std = all_proprio.std(axis=0)
    print(f"Proprio ({proprio_key}[:{proprio_dims}]): "
          f"mean={proprio_mean[:proprio_dims]}, std={proprio_std[:proprio_dims]}")

    # --- Dataset ---
    train_ratio = 0.8
    shuffled = np.random.permutation(num_episodes)
    train_ids = shuffled[:int(train_ratio * num_episodes)].tolist()
    val_ids = shuffled[int(train_ratio * num_episodes):].tolist()

    train_ds = VTClipDataset(train_ids, dataset_dir, camera_names,
                             tac_side=tac_side, n_samples=n_samples,
                             min_distance=min_distance,
                             proprio_key=proprio_key, proprio_dims=proprio_dims,
                             proprio_mean=proprio_mean, proprio_std=proprio_std)
    val_ds = VTClipDataset(val_ids, dataset_dir, camera_names,
                           tac_side=tac_side, n_samples=n_samples,
                           min_distance=min_distance,
                           proprio_key=proprio_key, proprio_dims=proprio_dims,
                           proprio_mean=proprio_mean, proprio_std=proprio_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # --- Models ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vision encoder
    vision_encoder = modified_resnet18().to(device)
    # Optionally load pretrained CLIP weights
    vision_pretrain = args.get('vision_backbone_path', None)
    if vision_pretrain and os.path.exists(vision_pretrain):
        vision_encoder.load_state_dict(torch.load(vision_pretrain))
        print(f"Loaded vision backbone from: {vision_pretrain}")

    vision_proj = VisionProjectionHead(
        num_channels=512, out_dim=clip_dim).to(device)

    # Tactile encoder (MarkerEncoderPointNet, same as TFAC) + projection with position conditioning
    tactile_encoder = MarkerEncoderPointNet(hidden_dim=512).to(device)
    tactile_proj = ProjectionHead(
        in_dim=512, out_dim=clip_dim, conditioning_dim=proprio_dims).to(device)

    # --- Optimizer ---
    resnet_lr = args.get('resnet_lr', 1e-5)
    proj_lr = args.get('projection_lr', 1e-4)
    pointnet_lr = args.get('pointnet_lr', 1e-4)

    optimizer = torch.optim.AdamW([
        {'params': vision_encoder.parameters(), 'lr': resnet_lr},
        {'params': vision_proj.parameters(), 'lr': proj_lr},
        {'params': tactile_encoder.parameters(), 'lr': pointnet_lr},
        {'params': tactile_proj.parameters(), 'lr': proj_lr},
    ], weight_decay=args.get('weight_decay', 1e-4))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7)

    n_cams = len(camera_names)
    trainable = sum(p.numel() for p in list(vision_encoder.parameters()) +
                    list(vision_proj.parameters()) +
                    list(tactile_encoder.parameters()) +
                    list(tactile_proj.parameters()))
    print(f"VT-CLIP: {trainable/1e6:.2f}M parameters")
    print(f"Cameras: {camera_names}, tactile: {tac_side}/marker_offset")
    print(f"clip_dim={clip_dim}, n_samples={n_samples}, temperature={temperature}")

    # --- Training loop ---
    train_losses = np.zeros((num_epochs, n_cams), dtype=np.float32)
    val_losses = np.zeros((num_epochs, n_cams), dtype=np.float32)
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in tqdm(range(num_epochs), desc="VT-CLIP"):
        # ---- Train ----
        vision_encoder.train()
        vision_proj.train()
        tactile_encoder.train()
        tactile_proj.train()

        epoch_loss = np.zeros(n_cams, dtype=np.float32)
        for images, markers, pos in train_loader:
            # images: (B, N, n_cams, 3, H, W), markers: (B, N, 9, 9, 2), pos: (B, N, 3)
            images = images.to(device)
            markers = markers.to(device)
            pos = pos.to(device)
            B, N = images.shape[0], images.shape[1]

            # Vision: (B*N*n_cams, 3, H, W) → (B, N, n_cams, D)
            imgs_flat = images.view(-1, *images.shape[3:])
            v_feat = vision_encoder(imgs_flat)          # feature maps
            v_emb = vision_proj(v_feat)                 # (B*N*n_cams, D)
            v_emb = v_emb.view(B, N, n_cams, clip_dim)

            # Tactile: (B*N, 9, 9, 2) + pos (B*N, 3) → (B, N, D)
            m_flat = markers.view(-1, 9, 9, 2)
            pos_flat = pos.view(-1, proprio_dims)
            t_feat = tactile_encoder(m_flat)            # (B*N, 512)
            t_emb = tactile_proj(t_feat, pos_flat)      # (B*N, D)
            t_emb = t_emb.view(B, N, clip_dim)

            loss_per_cam, _ = clip_loss(v_emb, t_emb, temperature)
            loss = loss_per_cam.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(vision_encoder.parameters()) +
                list(vision_proj.parameters()) +
                list(tactile_encoder.parameters()) +
                list(tactile_proj.parameters()),
                max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_per_cam.detach().cpu().numpy()

        train_losses[epoch] = epoch_loss / max(1, len(train_loader))

        # ---- Validate ----
        vision_encoder.eval()
        vision_proj.eval()
        tactile_encoder.eval()
        tactile_proj.eval()

        v_loss = np.zeros(n_cams, dtype=np.float32)
        sim_map_sample = None
        with torch.no_grad():
            for batch_idx, (images, markers, pos) in enumerate(val_loader):
                images = images.to(device)
                markers = markers.to(device)
                pos = pos.to(device)
                B, N = images.shape[0], images.shape[1]

                imgs_flat = images.view(-1, *images.shape[3:])
                v_feat = vision_encoder(imgs_flat)
                v_emb = vision_proj(v_feat).view(B, N, n_cams, clip_dim)

                m_flat = markers.view(-1, 9, 9, 2)
                pos_flat = pos.view(-1, proprio_dims)
                t_feat = tactile_encoder(m_flat)
                t_emb = tactile_proj(t_feat, pos_flat).view(B, N, clip_dim)

                loss_per_cam, sim_map = clip_loss(v_emb, t_emb, temperature)
                v_loss += loss_per_cam.detach().cpu().numpy()

                if batch_idx == 0:
                    sim_map_sample = sim_map

        val_losses[epoch] = v_loss / max(1, len(val_loader))

        scheduler.step()

        # ---- Best model ----
        avg_val = val_losses[epoch].mean()
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            torch.save(vision_encoder.state_dict(),
                       os.path.join(ckpt_dir, 'best_vision_encoder.pth'))
            torch.save(tactile_encoder.state_dict(),
                       os.path.join(ckpt_dir, 'best_tactile_encoder.pth'))
            torch.save(vision_proj.state_dict(),
                       os.path.join(ckpt_dir, 'best_vision_proj.pth'))
            torch.save(tactile_proj.state_dict(),
                       os.path.join(ckpt_dir, 'best_tactile_proj.pth'))

        # ---- Logging ----
        if epoch % 10 == 0:
            cam_str = ", ".join(f"{camera_names[i]}={val_losses[epoch][i]:.4f}"
                                for i in range(n_cams))
            tqdm.write(f"Epoch {epoch}: train={train_losses[epoch].mean():.4f}, "
                       f"val={avg_val:.4f} ({cam_str}) "
                       f"best: epoch {best_epoch}")

        # ---- Save & plot ----
        if epoch % 50 == 0 or epoch == num_epochs - 1:
            # Checkpoints
            torch.save(vision_encoder.state_dict(),
                       os.path.join(ckpt_dir, f'epoch_{epoch}_vision_encoder.pth'))
            torch.save(tactile_encoder.state_dict(),
                       os.path.join(ckpt_dir, f'epoch_{epoch}_tactile_encoder.pth'))

            # Loss curves
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(n_cams):
                ax.plot(train_losses[:epoch+1, i],
                        label=f"{camera_names[i]} train", color=f"C{i}")
                ax.plot(val_losses[:epoch+1, i],
                        label=f"{camera_names[i]} val",
                        linestyle="dashed", color=f"C{i}")
            ax.legend()
            ax.set_xlabel("Epoch")
            ax.set_ylabel("CLIP Loss")
            ax.set_title(f"VT-CLIP (best: epoch {best_epoch}, {best_val_loss:.4f})")
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, 'graphs', 'clip_loss.png'), dpi=100)
            plt.close()

            # Similarity map
            if sim_map_sample is not None:
                for i in range(n_cams):
                    fig, ax = plt.subplots(figsize=(5, 5))
                    im = ax.imshow(sim_map_sample[i], cmap='viridis', aspect='equal')
                    fig.colorbar(im, ax=ax)
                    ax.set_title(f"Sim Map - {camera_names[i]} (epoch {epoch})")
                    ax.set_xlabel("Tactile")
                    ax.set_ylabel("Vision")
                    plt.tight_layout()
                    plt.savefig(os.path.join(ckpt_dir, 'graphs',
                                f'sim_map_{camera_names[i]}_epoch_{epoch}.png'), dpi=100)
                    plt.close()

    # ---- Final save ----
    torch.save(vision_encoder.state_dict(),
               os.path.join(ckpt_dir, 'last_vision_encoder.pth'))
    torch.save(tactile_encoder.state_dict(),
               os.path.join(ckpt_dir, 'last_tactile_encoder.pth'))

    np.save(os.path.join(ckpt_dir, 'graphs', 'train_losses.npy'), train_losses)
    np.save(os.path.join(ckpt_dir, 'graphs', 'val_losses.npy'), val_losses)

    print(f"\nDone! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    cli = parser.parse_args()

    with open(cli.config, 'r') as f:
        config = json.load(f)
    train_clip(config)
