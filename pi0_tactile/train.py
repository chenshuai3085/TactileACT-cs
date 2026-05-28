"""
Pi0-TacForesight Training Script.

Usage:
    python -m pi0_tactile.train \
        --dataset_dir /path/to/episodes \
        --pi0_weights /path/to/pi0_base \
        --vae_checkpoint /path/to/tactile_vae.pth \
        --foresight_checkpoint /path/to/foresight.pth \
        --output_dir /home/chenshuai/Project/output/pi0_tactile_run1

Multi-GPU:
    torchrun --standalone --nproc_per_node=2 -m pi0_tactile.train ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pi0_tactile.config import Pi0TactileConfig
from pi0_tactile.model import Pi0Tactile
from pi0_tactile.dataset import (
    Pi0TactileDataset,
    build_datasets,
    collate_pi0_batch,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Pi0-TacForesight Training")

    # Data
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/home/chenshuai/Project/output/pi0_tactile")

    # Pretrained weights
    parser.add_argument("--pi0_weights", type=str, default="",
                        help="Path to pi0_base checkpoint dir (with model.safetensors)")
    parser.add_argument("--vae_checkpoint", type=str, default="")
    parser.add_argument("--foresight_checkpoint", type=str, default="")

    # Model
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--action_horizon", type=int, default=20)
    parser.add_argument("--tac_history", type=int, default=8)
    parser.add_argument("--foresight_horizon", type=int, default=10)
    parser.add_argument("--lambda_foresight", type=float, default=0.1)
    parser.add_argument("--foresight_t_threshold", type=float, default=0.3)
    parser.add_argument("--foresight_warmup_steps", type=int, default=1000)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_train_steps", type=int, default=30000)
    parser.add_argument("--save_interval", type=int, default=2000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # Resume
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint path")

    return parser.parse_args()


def setup_ddp():
    """Initialize distributed training if available."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    return use_ddp, local_rank, device


def cosine_lr_schedule(step, warmup_steps, peak_lr, total_steps, end_lr=1e-6):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cos = 0.5 * (1 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cos


def save_checkpoint(model, optimizer, step, config, output_dir, val_loss=None):
    """Save training checkpoint."""
    ckpt_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model

    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "val_loss": val_loss,
    }, os.path.join(ckpt_dir, "checkpoint.pth"))

    # Save config
    import dataclasses
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2, default=str)

    logger.info(f"Saved checkpoint at step {step} → {ckpt_dir}")


@torch.no_grad()
def validate(model, val_loader, device, max_batches=20):
    """Run validation and return average losses."""
    model.eval()
    total_flow = 0.0
    total_foresight = 0.0
    n = 0

    for i, (obs, actions) in enumerate(val_loader):
        if i >= max_batches:
            break

        obs = obs.to(device)
        actions = actions.to(device)

        # Unpack observation
        images_list = list(obs.images.values())
        img_masks_list = list(obs.image_masks.values())

        raw_model = model.module if hasattr(model, "module") else model
        losses = raw_model(
            images=images_list,
            img_masks=img_masks_list,
            lang_tokens=obs.tokenized_prompt,
            lang_masks=obs.tokenized_prompt_mask,
            state=obs.state,
            marker_offset=obs.marker_offset,
            actions=actions,
            future_marker_offset=obs.future_marker_offset,
        )

        total_flow += losses["flow_loss"].mean().item()
        total_foresight += losses["foresight_loss"].item()
        n += 1

    model.train()
    return {
        "val_flow_loss": total_flow / max(n, 1),
        "val_foresight_loss": total_foresight / max(n, 1),
    }


def train(args):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    # === Config ===
    config = Pi0TactileConfig(
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        tac_history=args.tac_history,
        foresight_horizon=args.foresight_horizon,
        lambda_foresight=args.lambda_foresight,
        foresight_t_threshold=args.foresight_t_threshold,
        foresight_warmup_steps=args.foresight_warmup_steps,
        vae_checkpoint=args.vae_checkpoint,
        foresight_checkpoint=args.foresight_checkpoint,
        dataset_dir=args.dataset_dir,
    )

    # === Dataset ===
    train_ds, val_ds, norm_stats = build_datasets(
        args.dataset_dir, config, seed=args.seed
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pi0_batch,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_pi0_batch,
        pin_memory=True,
    )

    if is_main:
        logger.info(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
        logger.info(f"Batch size: {args.batch_size}, Steps: {args.num_train_steps}")

    # === Model ===
    model = Pi0Tactile(config).to(device)

    # Load pi0 pretrained weights
    if args.pi0_weights:
        model.load_pi0_weights(args.pi0_weights)

    # Freeze PaliGemma, keep action expert + tactile + foresight trainable
    if config.freeze_paligemma:
        model.freeze_paligemma()

    # Gradient checkpointing
    if args.gradient_checkpointing:
        model.pi0.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")

    # DDP
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )

    # === Optimizer ===
    raw_model = model.module if use_ddp else model
    param_groups = raw_model.get_trainable_params()

    # Filter out empty groups and set LR
    optimizer_groups = []
    for pg in param_groups:
        params = [p for p in pg["params"] if p.requires_grad]
        if params:
            optimizer_groups.append({
                "params": params,
                "lr": args.lr * pg.get("lr_scale", 1.0),
                "name": pg.get("name", "default"),
            })

    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )

    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in raw_model.parameters())
    if is_main:
        logger.info(f"Trainable: {n_trainable/1e6:.1f}M / Total: {n_total/1e6:.1f}M params")
        for pg in optimizer_groups:
            n_pg = sum(p.numel() for p in pg["params"])
            logger.info(f"  {pg['name']}: {n_pg/1e6:.2f}M params, lr={pg['lr']:.2e}")

    # === Resume ===
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["step"]
        logger.info(f"Resumed from step {global_step}")

    # === Training Loop ===
    os.makedirs(args.output_dir, exist_ok=True)
    model.train()
    start_time = time.time()
    running_losses = {"flow": [], "foresight": [], "total": []}
    best_val_loss = float("inf")

    if is_main:
        logger.info("=" * 60)
        logger.info("Starting Pi0-TacForesight training")
        logger.info("=" * 60)

    while global_step < args.num_train_steps:
        for obs, actions in train_loader:
            if global_step >= args.num_train_steps:
                break

            # Move to device
            obs = obs.to(device)
            actions = actions.to(device)

            # LR schedule
            lr = cosine_lr_schedule(
                global_step, args.warmup_steps, args.lr, args.num_train_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr * pg.get("lr_scale", 1.0) if "lr_scale" in pg else lr

            # Update foresight step counter
            raw_model.set_step(global_step)

            # Forward
            images_list = list(obs.images.values())
            img_masks_list = list(obs.image_masks.values())

            losses = raw_model(
                images=images_list,
                img_masks=img_masks_list,
                lang_tokens=obs.tokenized_prompt,
                lang_masks=obs.tokenized_prompt_mask,
                state=obs.state,
                marker_offset=obs.marker_offset,
                actions=actions,
                future_marker_offset=obs.future_marker_offset,
            ) if not use_ddp else model.module(
                images=images_list,
                img_masks=img_masks_list,
                lang_tokens=obs.tokenized_prompt,
                lang_masks=obs.tokenized_prompt_mask,
                state=obs.state,
                marker_offset=obs.marker_offset,
                actions=actions,
                future_marker_offset=obs.future_marker_offset,
            )

            loss = losses["total_loss"]

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                max_norm=args.grad_clip,
            )
            optimizer.step()

            # Logging
            running_losses["flow"].append(losses["flow_loss"].mean().item())
            running_losses["foresight"].append(losses["foresight_loss"].item())
            running_losses["total"].append(loss.item())

            if is_main and global_step % args.log_interval == 0 and global_step > 0:
                elapsed = time.time() - start_time
                avg_flow = np.mean(running_losses["flow"])
                avg_fore = np.mean(running_losses["foresight"])
                avg_total = np.mean(running_losses["total"])

                logger.info(
                    f"step={global_step} | "
                    f"loss={avg_total:.4f} (flow={avg_flow:.4f}, fore={avg_fore:.4f}) | "
                    f"lr={lr:.2e} | grad={grad_norm:.2f} | "
                    f"time={elapsed:.1f}s"
                )

                running_losses = {"flow": [], "foresight": [], "total": []}
                start_time = time.time()

            # Validation
            if is_main and global_step % args.val_interval == 0 and global_step > 0:
                val_metrics = validate(raw_model, val_loader, device)
                logger.info(
                    f"  [VAL] flow={val_metrics['val_flow_loss']:.4f}, "
                    f"fore={val_metrics['val_foresight_loss']:.4f}"
                )

                if val_metrics["val_flow_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_flow_loss"]
                    save_checkpoint(
                        model, optimizer, global_step, config,
                        args.output_dir, val_loss=best_val_loss
                    )

            # Periodic save
            if is_main and global_step % args.save_interval == 0 and global_step > 0:
                save_checkpoint(model, optimizer, global_step, config, args.output_dir)

            global_step += 1

    # Final save
    if is_main:
        save_checkpoint(model, optimizer, global_step, config, args.output_dir)
        logger.info("Training complete!")

    if use_ddp:
        dist.destroy_process_group()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
