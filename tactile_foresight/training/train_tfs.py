"""
Train the Tactile Feasibility Score (TFS).

Stage 2: Contrastive learning scorer that maps (observation, action) pairs
to a feasibility score. Trained with InfoNCE loss and noisy action augmentation.

Usage:
    cd /path/to/TactileACT-cs
    python -m tactile_foresight.training.train_tfs \
        --dataset_dir /home/chenshuai/data/dataset/260309_0310 \
        --alignment_ckpt /path/to/alignment_best.pth \
        --save_dir /path/to/output/tfs \
        --num_episodes 337 \
        --epochs 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from tactile_foresight.models.tfs import TactileFeasibilityScore
from tactile_foresight.datasets.foresight_dataset import (
    ForesightDataset,
    collate_foresight,
)


def parse_args():
    p = argparse.ArgumentParser(description="Train TFS")
    p.add_argument("--dataset_dir", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--num_episodes", type=int, required=True)
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--camera_names", type=str, default="global,wrist")
    p.add_argument("--horizons", type=str, default="4,8,12")
    p.add_argument("--chunk_size", type=int, default=20)
    p.add_argument("--proprio_key", type=str, default="proprio_eef")
    p.add_argument("--action_key", type=str, default="actions/joint_abs")
    p.add_argument("--tac_side", type=str, default="left")
    p.add_argument("--tac_img_key", type=str, default="img")
    p.add_argument("--tac_mask_ratio", type=float, default=0.0,
                    help="Tactile mask ratio (0 for TFS, we always use real tactile)")

    # Alignment
    p.add_argument("--alignment_ckpt", type=str, required=True,
                    help="Path to pre-trained VT alignment checkpoint")
    p.add_argument("--aligned_dim", type=int, default=256)

    # Model
    p.add_argument("--score_dim", type=int, default=128)
    p.add_argument("--obs_hidden_dim", type=int, default=256)
    p.add_argument("--obs_num_layers", type=int, default=2)
    p.add_argument("--obs_nheads", type=int, default=4)
    p.add_argument("--action_dim", type=int, default=7)
    p.add_argument("--action_hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # Noisy action training
    p.add_argument("--noisy_action", action="store_true",
                    help="Add DDPM-schedule noise to actions during training")
    p.add_argument("--noise_steps", type=int, default=100,
                    help="Max diffusion noise steps (matches DP training)")
    p.add_argument("--noise_ratio", type=float, default=0.5,
                    help="Fraction of batch with noisy actions")

    # Training
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--save_freq", type=int, default=100)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--plot_freq", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_ddpm_betas(num_steps: int = 100):
    """Squared cosine beta schedule (matching DP training)."""
    steps = np.arange(num_steps + 1, dtype=np.float64)
    alpha_bar = np.cos(((steps / num_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = np.clip(betas, 0, 0.999)
    alphas = 1.0 - betas
    alpha_cumprod = np.cumprod(alphas)
    return torch.from_numpy(alpha_cumprod).float()


def add_ddpm_noise(actions, alpha_cumprod, device):
    """Add DDPM noise to actions at random timesteps.

    Args:
        actions: (B, T, D) clean action chunks
        alpha_cumprod: (num_steps,) cumulative product of alphas

    Returns:
        noisy_actions: (B, T, D)
        noise: (B, T, D)
        timesteps: (B,)
    """
    B = actions.shape[0]
    num_steps = alpha_cumprod.shape[0]
    alpha_cumprod = alpha_cumprod.to(device)

    # Sample random timesteps
    timesteps = torch.randint(0, num_steps, (B,), device=device)
    sqrt_alpha = alpha_cumprod[timesteps].sqrt().view(B, 1, 1)
    sqrt_one_minus = (1 - alpha_cumprod[timesteps]).sqrt().view(B, 1, 1)

    noise = torch.randn_like(actions)
    noisy_actions = sqrt_alpha * actions + sqrt_one_minus * noise
    return noisy_actions, noise, timesteps


def plot_curves(history, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("InfoNCE Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Top-1 Matching Accuracy")
    axes[1].legend()

    axes[2].plot(history["val_pos_score"], label="pos")
    axes[2].plot(history["val_neg_score"], label="neg")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Pos vs Neg Score")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "graphs", "tfs_curves.png"), dpi=100)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "graphs"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_tfs] device={device}")

    camera_names = args.camera_names.split(",")
    horizons = [int(h) for h in args.horizons.split(",")]

    # Save config
    with open(os.path.join(args.save_dir, "tfs_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # --- Dataset ---
    all_ids = list(range(args.start_episode, args.start_episode + args.num_episodes))
    full_dataset = ForesightDataset(
        episode_ids=all_ids,
        dataset_dir=args.dataset_dir,
        camera_names=camera_names,
        horizons=horizons,
        chunk_size=args.chunk_size,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        tac_side=args.tac_side,
        tac_img_key=args.tac_img_key,
        tac_mask_ratio=args.tac_mask_ratio,
    )

    val_size = int(len(full_dataset) * args.val_ratio)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"[train_tfs] train={train_size}, val={val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_foresight,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_foresight,
        pin_memory=True, drop_last=True,
    )

    # --- DDPM noise schedule ---
    alpha_cumprod = None
    if args.noisy_action:
        alpha_cumprod = make_ddpm_betas(args.noise_steps)
        print(f"[train_tfs] Noisy action training enabled (steps={args.noise_steps}, ratio={args.noise_ratio})")

    # --- Model ---
    print("[train_tfs] Building TFS...")
    model = TactileFeasibilityScore(
        alignment_ckpt=args.alignment_ckpt,
        aligned_dim=args.aligned_dim,
        score_dim=args.score_dim,
        obs_hidden_dim=args.obs_hidden_dim,
        obs_num_layers=args.obs_num_layers,
        obs_nheads=args.obs_nheads,
        action_dim=args.action_dim,
        chunk_size=args.chunk_size,
        action_hidden_dim=args.action_hidden_dim,
        dropout=args.dropout,
        device=device,
    )
    print(f"[train_tfs] Trainable params: {model.num_trainable_params():,}")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_warmup_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # --- Training loop ---
    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_pos_score": [], "val_neg_score": [],
    }

    for epoch in range(args.epochs):
        t0 = time.time()

        # --- Train ---
        model.obs_encoder.train()
        model.action_encoder.train()
        epoch_loss, epoch_acc, n_batches = 0.0, 0.0, 0

        for batch in train_loader:
            vis_img = batch["vision_images"][camera_names[0]].to(device)
            tac_fut = batch["tac_future"].to(device)
            actions = batch["actions"].to(device)

            # Optionally add noise to actions
            action_noise = None
            if alpha_cumprod is not None:
                B = actions.shape[0]
                n_noisy = int(B * args.noise_ratio)
                if n_noisy > 0:
                    noisy, noise, _ = add_ddpm_noise(
                        actions[:n_noisy], alpha_cumprod, device
                    )
                    # Replace a fraction of the batch with noisy actions
                    noisy_actions = actions.clone()
                    noisy_actions[:n_noisy] = noisy
                    actions = noisy_actions

            result = model.compute_loss(vis_img, tac_fut, actions)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += result["accuracy"]
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        avg_train_acc = epoch_acc / max(n_batches, 1)

        # --- Validate ---
        model.obs_encoder.eval()
        model.action_encoder.eval()
        val_loss, val_acc, val_pos, val_neg, n_val = 0.0, 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for batch in val_loader:
                vis_img = batch["vision_images"][camera_names[0]].to(device)
                tac_fut = batch["tac_future"].to(device)
                actions = batch["actions"].to(device)

                result = model.compute_loss(vis_img, tac_fut, actions)
                val_loss += result["loss"].item()
                val_acc += result["accuracy"]
                val_pos += result["pos_score"]
                val_neg += result["neg_score"]
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_acc = val_acc / max(n_val, 1)
        avg_val_pos = val_pos / max(n_val, 1)
        avg_val_neg = val_neg / max(n_val, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)
        history["val_pos_score"].append(avg_val_pos)
        history["val_neg_score"].append(avg_val_neg)

        dt = time.time() - t0

        if (epoch + 1) % args.log_freq == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            temp = model.temperature.item()
            print(
                f"[epoch {epoch+1:4d}/{args.epochs}] "
                f"loss={avg_train_loss:.4f}/{avg_val_loss:.4f}  "
                f"acc={avg_train_acc:.3f}/{avg_val_acc:.3f}  "
                f"pos={avg_val_pos:.3f} neg={avg_val_neg:.3f}  "
                f"temp={temp:.2f}  lr={lr:.2e}  {dt:.1f}s"
            )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save(os.path.join(args.save_dir, "tfs_best.pth"))

        # Periodic save
        if (epoch + 1) % args.save_freq == 0:
            model.save(os.path.join(args.save_dir, f"tfs_epoch_{epoch+1}.pth"))

        # Plot
        if (epoch + 1) % args.plot_freq == 0:
            plot_curves(history, args.save_dir)

    # Final saves
    model.save(os.path.join(args.save_dir, "tfs_final.pth"))
    with open(os.path.join(args.save_dir, "tfs_history.json"), "w") as f:
        json.dump(history, f)
    plot_curves(history, args.save_dir)

    print(f"\n[train_tfs] Done. Best val loss: {best_val_loss:.4f}")
    print(f"[train_tfs] Saved to {args.save_dir}")


if __name__ == "__main__":
    main()
