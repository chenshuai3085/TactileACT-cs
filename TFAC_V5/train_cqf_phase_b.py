"""
CQF Phase B/C Training: mixed GT + Foresight prediction.

Phase B: gt_ratio linearly decays from 0.9 to 0.1 over training
Phase C: gt_ratio = 0.0 (pure Foresight)

Requires:
  - Trained LightweightForesight checkpoint
  - CQF Phase A checkpoint (for warm-start)

Usage:
  # Phase B (30 epochs, gt_ratio 0.9→0.1)
  python TFAC_V5/train_cqf_phase_b.py \
    --foresight_ckpt /path/to/foresight_best.pt \
    --cqf_ckpt /path/to/cqf_best.pt \
    --phase B --epochs 30

  # Phase C (10 epochs, pure Foresight)
  python TFAC_V5/train_cqf_phase_b.py \
    --foresight_ckpt /path/to/foresight_best.pt \
    --cqf_ckpt /path/to/phase_b_best.pt \
    --phase C --epochs 10
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cqf_model import ContactQualityScorer, CQFLoss
from cqf_phase_b_dataset import CQFPhaseBDataset
from lightweight_foresight import LightweightForesight


def load_foresight(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = LightweightForesight(
        hidden=args.get("hidden", 512),
        n_layers=args.get("n_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Loaded foresight from {ckpt_path}")
    return model


def load_cqf(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    model = ContactQualityScorer(
        tac_dim=args.get("tac_dim", 162),
        hidden=args.get("hidden", 256),
        action_dropout=args.get("action_dropout", 0.5),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded CQF from {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_metrics = {}
    n = 0
    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        mcur = batch["marker_cur"].to(device).flatten(1)
        mfut = batch["marker_future"].to(device).flatten(1)
        labels = batch["label"].to(device)

        scores = model(qpos, eef, action, mcur, mfut)
        loss, metrics = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n += 1
    return {k: v / n for k, v in total_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_metrics = {}
    n = 0
    all_scores, all_labels = [], []

    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        mcur = batch["marker_cur"].to(device).flatten(1)
        mfut = batch["marker_future"].to(device).flatten(1)
        labels = batch["label"].to(device)

        scores = model(qpos, eef, action, mcur, mfut)
        loss, metrics = criterion(scores, labels)

        all_scores.append(scores.squeeze(-1).cpu())
        all_labels.append(labels.cpu())

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n += 1

    avg = {k: v / n for k, v in total_metrics.items()}
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    pos = all_scores[all_labels > 0.5]
    neg = all_scores[all_labels <= 0.5]
    if len(pos) > 0 and len(neg) > 0:
        avg["spread"] = (pos.mean() - neg.mean()).item()
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--foresight_ckpt", type=str, required=True)
    parser.add_argument("--cqf_ckpt", type=str, default=None,
                        help="CQF checkpoint for warm-start")
    parser.add_argument("--phase", type=str, default="B", choices=["B", "C"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf_phase_b")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    phase = args.phase
    print("=" * 60)
    print(f"CQF Phase {phase} Training")
    print("=" * 60)

    foresight = load_foresight(args.foresight_ckpt, device)

    if args.cqf_ckpt:
        model = load_cqf(args.cqf_ckpt, device)
    else:
        model = ContactQualityScorer(tac_dim=162, hidden=256, action_dropout=0.5).to(device)

    criterion = CQFLoss(margin=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_spread = -float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()

        if phase == "B":
            gt_ratio = 0.9 - 0.8 * (epoch / max(args.epochs - 1, 1))
        else:
            gt_ratio = 0.0

        print(f"\nBuilding dataset (gt_ratio={gt_ratio:.2f})...")
        train_ds = CQFPhaseBDataset(
            foresight, gt_ratio=gt_ratio,
            n_perturb_per_pos=1, device=args.device, split="train")
        val_ds = CQFPhaseBDataset(
            foresight, gt_ratio=0.0, device=args.device, split="val")

        train_sampler = train_ds.get_weighted_sampler()
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        spread = val_m.get("spread", 0)

        print(f"Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e} gt_ratio={gt_ratio:.2f}")
        print(f"  Train: loss={train_m['loss']:.4f} acc={train_m['accuracy']:.3f} "
              f"spread={train_m['score_spread']:.3f}")
        print(f"  Val:   loss={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
              f"spread={spread:.3f}")

        if spread > best_spread:
            best_spread = spread
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "phase": phase,
                "gt_ratio": gt_ratio,
            }, os.path.join(args.output_dir, f"cqf_phase_{phase.lower()}_best.pt"))
            print(f"  ★ Best model saved (spread={spread:.4f})")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_m,
        "phase": phase,
    }, os.path.join(args.output_dir, f"cqf_phase_{phase.lower()}_final.pt"))

    print(f"\nPhase {phase} complete. Best spread: {best_spread:.4f}")


if __name__ == "__main__":
    main()
