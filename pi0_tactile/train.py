"""Train pi0/pi0.5 with TactileVAE tokens, foresight loss, and flow guidance config."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pi0_tactile.config import Pi0TactileConfig


LOGGER = logging.getLogger("pi0_tactile.train")


def parse_csv(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_shape(value: Optional[str]) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    parts = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected H,W shape, got {value}")
    return parts[0], parts[1]


def load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def setup_logging(output_dir: Path, is_main: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO if is_main else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def setup_distributed(gpu: Optional[int]) -> tuple[bool, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    use_ddp = world_size > 1
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if torch.cuda.is_available():
        cuda_index = local_rank if use_ddp else int(gpu or 0)
        torch.cuda.set_device(cuda_index)
        device = torch.device(f"cuda:{cuda_index}")
    else:
        device = torch.device("cpu")
    return use_ddp, rank, local_rank, device


def set_seed(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float, end_lr: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * float(step + 1) / float(warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(end_lr + (peak_lr - end_lr) * cos)


def config_from_args(args: argparse.Namespace) -> Pi0TactileConfig:
    cfg: dict[str, Any] = {}
    if args.config:
        cfg.update({
            k: v for k, v in load_json(args.config).items()
            if k in Pi0TactileConfig.__dataclass_fields__
        })

    overrides = {
        "dataset_dir": args.dataset_dir,
        "vae_checkpoint": args.vae_checkpoint,
        "foresight_checkpoint": args.foresight_checkpoint,
        "action_dim": args.action_dim,
        "robot_action_dim": args.robot_action_dim,
        "action_horizon": args.action_horizon,
        "max_token_len": args.max_token_len,
        "tokenizer_backend": args.tokenizer_backend,
        "camera_names": parse_csv(args.camera_names),
        "image_size": parse_shape(args.image_size),
        "tac_history": args.tac_history,
        "foresight_horizon": args.foresight_horizon,
        "foresight_predict_horizon": args.foresight_predict_horizon,
        "lambda_foresight": args.lambda_foresight,
        "foresight_t_threshold": args.foresight_t_threshold,
        "foresight_warmup_steps": args.foresight_warmup_steps,
        "fixed_prompt": args.fixed_prompt,
        "action_key": args.action_key,
        "proprio_key": args.proprio_key,
        "tactile_key": args.tactile_key,
        "samples_per_episode": args.samples_per_episode,
        "preload_dataset": args.preload_dataset,
        "flow_guidance_steps": args.flow_guidance_steps,
        "flow_guidance_scale": args.flow_guidance_scale,
        "flow_guidance_max_total_delta": args.flow_guidance_max_total_delta,
        "flow_guidance_lambda_smooth": args.flow_guidance_lambda_smooth,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value

    if args.pi05:
        cfg["pi05"] = True
    if bool(cfg.get("pi05", False)) and args.action_dim is None and int(cfg.get("action_dim", 7)) == 7:
        cfg["action_dim"] = 32
    if args.no_freeze_paligemma:
        cfg["freeze_paligemma"] = False

    config = Pi0TactileConfig(**cfg)
    if not config.dataset_dir:
        raise ValueError("--dataset_dir is required unless set in --config")
    return config


def build_optimizer(model: Pi0Tactile, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    optimizer_groups = []
    for group in model.get_trainable_params():
        params = [p for p in group["params"] if p.requires_grad]
        if not params:
            continue
        scale = float(group.get("lr_scale", 1.0))
        optimizer_groups.append({
            "params": params,
            "lr": lr * scale,
            "lr_scale": scale,
            "name": group.get("name", "default"),
        })
    return torch.optim.AdamW(
        optimizer_groups,
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, base_lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = base_lr * float(group.get("lr_scale", 1.0))


def model_state(model: torch.nn.Module) -> Pi0Tactile:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Pi0TactileConfig,
    step: int,
    best_val: float,
    metrics: Optional[dict[str, float]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model_state(model)
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": dataclasses.asdict(config),
            "step": int(step),
            "best_val": float(best_val),
            "metrics": metrics or {},
        },
        path,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: Optional[DataLoader],
    device: torch.device,
    max_batches: int,
    step: int,
) -> dict[str, float]:
    if val_loader is None:
        return {"val_flow_loss": float("nan"), "val_foresight_loss": float("nan"), "val_total_loss": float("nan")}
    raw_model = model_state(model)
    was_training = raw_model.training
    raw_model.eval()
    raw_model.set_step(step)

    sums = {"flow": 0.0, "foresight": 0.0, "total": 0.0}
    count = 0
    for batch_idx, (obs, actions) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        obs = obs.to(device)
        actions = actions.to(device)
        losses = raw_model(
            images=list(obs.images.values()),
            img_masks=list(obs.image_masks.values()),
            lang_tokens=obs.tokenized_prompt,
            lang_masks=obs.tokenized_prompt_mask,
            state=obs.state,
            marker_offset=obs.marker_offset,
            actions=actions,
            future_marker_offset=obs.future_marker_offset,
            compute_foresight_loss=True,
        )
        sums["flow"] += float(losses["flow_loss"].mean().detach().cpu())
        sums["foresight"] += float(losses["foresight_loss"].detach().cpu())
        sums["total"] += float(losses["total_loss"].detach().cpu())
        count += 1

    raw_model.train(was_training)
    denom = max(1, count)
    return {
        "val_flow_loss": sums["flow"] / denom,
        "val_foresight_loss": sums["foresight"] / denom,
        "val_total_loss": sums["total"] / denom,
    }


def train(args: argparse.Namespace) -> None:
    from pi0_tactile.dataset import build_datasets, collate_pi0_batch
    from pi0_tactile.model import Pi0Tactile

    use_ddp, rank, local_rank, device = setup_distributed(args.gpu)
    is_main = rank == 0
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, is_main)
    set_seed(args.seed, rank)

    config = config_from_args(args)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(dataclasses.asdict(config), output_dir / "config.json")
        LOGGER.info("Output: %s", output_dir)
        LOGGER.info(
            "Config: pi05=%s action_dim=%d robot_action_dim=%d horizon=%d tokenizer=%s",
            config.pi05,
            config.action_dim,
            config.robot_action_dim,
            config.action_horizon,
            config.tokenizer_backend,
        )

    train_ds, val_ds, norm_stats = build_datasets(
        config.dataset_dir,
        config,
        seed=args.seed,
        val_ratio=args.val_ratio,
    )
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {config.dataset_dir}")
    if is_main:
        with open(output_dir / "dataset_stats.pkl", "wb") as f:
            pickle.dump(norm_stats, f)
        LOGGER.info("Train samples=%d, Val samples=%d", len(train_ds), len(val_ds))

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if use_ddp and len(val_ds) > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_pi0_batch,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=max(0, min(2, args.num_workers)),
            collate_fn=collate_pi0_batch,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    model = Pi0Tactile(config).to(device)
    if args.pi0_weights:
        model.load_pi0_weights(args.pi0_weights)
    if config.freeze_paligemma:
        model.freeze_paligemma()
    if args.gradient_checkpointing:
        model.pi0.gradient_checkpointing_enable()

    raw_model = model
    optimizer = build_optimizer(raw_model, args.lr, args.weight_decay)

    n_total = sum(p.numel() for p in raw_model.parameters())
    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main:
        LOGGER.info("Params trainable %.2fM / total %.2fM", n_trainable / 1e6, n_total / 1e6)
        for group in optimizer.param_groups:
            n_group = sum(p.numel() for p in group["params"])
            LOGGER.info(
                "  group=%s params=%.2fM lr_scale=%.3f",
                group.get("name", "default"),
                n_group / 1e6,
                float(group.get("lr_scale", 1.0)),
            )

    start_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        missing, unexpected = raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val", best_val))
        LOGGER.info(
            "Resumed %s at step=%d missing=%d unexpected=%d",
            args.resume,
            start_step,
            len(missing),
            len(unexpected),
        )

    if use_ddp:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    total_steps = int(args.num_train_steps)
    if args.epochs > 0 and args.num_train_steps <= 0:
        total_steps = int(args.epochs) * max(1, len(train_loader))
    if total_steps <= 0:
        raise ValueError("Set --num_train_steps > 0 or --epochs > 0")

    LOGGER.info("Start training for %d optimizer steps", total_steps)
    model.train()
    step = start_step
    epoch = 0
    running = {"flow": [], "foresight": [], "total": []}
    last_log_time = time.time()

    while step < total_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for obs, actions in train_loader:
            if step >= total_steps:
                break
            obs = obs.to(device)
            actions = actions.to(device)

            lr_now = cosine_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
            set_optimizer_lr(optimizer, lr_now)
            model_state(model).set_step(step)

            losses = model(
                images=list(obs.images.values()),
                img_masks=list(obs.image_masks.values()),
                lang_tokens=obs.tokenized_prompt,
                lang_masks=obs.tokenized_prompt_mask,
                state=obs.state,
                marker_offset=obs.marker_offset,
                actions=actions,
                future_marker_offset=obs.future_marker_offset,
            )
            loss = losses["total_loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model_state(model).parameters() if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()

            running["flow"].append(float(losses["flow_loss"].mean().detach().cpu()))
            running["foresight"].append(float(losses["foresight_loss"].detach().cpu()))
            running["total"].append(float(loss.detach().cpu()))

            if is_main and step > start_step and args.log_every > 0 and step % args.log_every == 0:
                dt = max(1e-6, time.time() - last_log_time)
                LOGGER.info(
                    "step=%d lr=%.3e loss=%.5f flow=%.5f foresight=%.5f grad=%.3f %.2fs/%dstep",
                    step,
                    lr_now,
                    float(np.mean(running["total"])),
                    float(np.mean(running["flow"])),
                    float(np.mean(running["foresight"])),
                    float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                    dt,
                    max(1, len(running["total"])),
                )
                running = {"flow": [], "foresight": [], "total": []}
                last_log_time = time.time()

            if is_main and step > start_step and args.val_every > 0 and step % args.val_every == 0:
                metrics = validate(model_state(model), val_loader, device, args.max_val_batches, step)
                LOGGER.info(
                    "val step=%d total=%.5f flow=%.5f foresight=%.5f",
                    step,
                    metrics["val_total_loss"],
                    metrics["val_flow_loss"],
                    metrics["val_foresight_loss"],
                )
                if metrics["val_total_loss"] < best_val:
                    best_val = metrics["val_total_loss"]
                    save_checkpoint(output_dir / "checkpoint_best.pth", model, optimizer, config, step, best_val, metrics)

            if is_main and step > start_step and args.save_every > 0 and step % args.save_every == 0:
                metrics = {
                    "train_total_loss": float(np.mean(running["total"])) if running["total"] else float("nan"),
                    "best_val": best_val,
                }
                save_checkpoint(output_dir / "checkpoint.pth", model, optimizer, config, step, best_val, metrics)
                save_checkpoint(output_dir / f"checkpoint_step_{step}.pth", model, optimizer, config, step, best_val, metrics)
                LOGGER.info("Saved checkpoint at step=%d", step)

            step += 1
        epoch += 1

    if is_main:
        metrics = validate(model_state(model), val_loader, device, args.max_val_batches, step)
        save_checkpoint(output_dir / "checkpoint.pth", model, optimizer, config, step, best_val, metrics)
        save_checkpoint(output_dir / f"checkpoint_step_{step}.pth", model, optimizer, config, step, best_val, metrics)
        LOGGER.info("Training complete at step=%d", step)

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pi0/pi0.5 tactile foresight policy")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pi0_weights", type=str, default="")
    parser.add_argument("--vae_checkpoint", type=str, default=None)
    parser.add_argument("--foresight_checkpoint", type=str, default=None)

    parser.add_argument("--pi05", action="store_true")
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--robot_action_dim", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--max_token_len", type=int, default=None)
    parser.add_argument("--tokenizer_backend", type=str, default=None, choices=["ascii", "paligemma", "official", "openpi"])
    parser.add_argument("--camera_names", type=str, default=None)
    parser.add_argument("--image_size", type=str, default=None)
    parser.add_argument("--tac_history", type=int, default=None)
    parser.add_argument("--foresight_horizon", type=int, default=None)
    parser.add_argument("--foresight_predict_horizon", type=int, default=None)
    parser.add_argument("--lambda_foresight", type=float, default=None)
    parser.add_argument("--foresight_t_threshold", type=float, default=None)
    parser.add_argument("--foresight_warmup_steps", type=int, default=None)
    parser.add_argument("--fixed_prompt", type=str, default=None)
    parser.add_argument("--action_key", type=str, default=None)
    parser.add_argument("--proprio_key", type=str, default=None)
    parser.add_argument("--tactile_key", type=str, default=None)
    parser.add_argument("--samples_per_episode", type=int, default=None)
    parser.add_argument("--preload_dataset", dest="preload_dataset", action="store_true", default=None)
    parser.add_argument("--no_preload_dataset", dest="preload_dataset", action="store_false")

    parser.add_argument("--flow_guidance_steps", type=int, default=None)
    parser.add_argument("--flow_guidance_scale", type=float, default=None)
    parser.add_argument("--flow_guidance_max_total_delta", type=float, default=None)
    parser.add_argument("--flow_guidance_lambda_smooth", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_train_steps", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_val_batches", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_freeze_paligemma", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
