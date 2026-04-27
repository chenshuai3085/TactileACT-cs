"""
CQF Phase A Training: GT tactile.

使用raw marker_offset作为触觉表示 (不需要TactileVAE)。
正样本: insertion帧, 负样本: pre_bounce帧 (过滤后t+h>=lift_start)。

Usage:
  python TFAC_V5/train_cqf.py --config TFAC_V5/config_cqf.json
  python TFAC_V5/train_cqf.py  # 使用默认参数
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cqf_dataset import CQFDataset, build_cqf_dataloaders
from cqf_model import ContactQualityScorer, CQFLoss


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_metrics = {}
    n_batches = 0

    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        marker_cur = batch["marker_cur"].to(device).flatten(1)     # (B, 162)
        marker_future = batch["marker_future"].to(device).flatten(1)  # (B, 162)
        labels = batch["label"].to(device)

        scores = model(qpos, eef, action, marker_cur, marker_future)
        loss, metrics = criterion(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_metrics.items()}
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_metrics = {}
    n_batches = 0

    all_scores = []
    all_labels = []

    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        marker_cur = batch["marker_cur"].to(device).flatten(1)
        marker_future = batch["marker_future"].to(device).flatten(1)
        labels = batch["label"].to(device)

        scores = model(qpos, eef, action, marker_cur, marker_future)
        loss, metrics = criterion(scores, labels)

        all_scores.append(scores.squeeze(-1).cpu())
        all_labels.append(labels.cpu())

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n_batches += 1

    avg = {k: v / n_batches for k, v in total_metrics.items()}

    # ranking accuracy: 从16个样本(1 pos + 15 neg)中选出pos排第一的比例
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    pos_scores = all_scores[all_labels > 0.5]
    neg_scores = all_scores[all_labels <= 0.5]

    if len(pos_scores) > 0 and len(neg_scores) > 0:
        avg["global_pos_mean"] = pos_scores.mean().item()
        avg["global_neg_mean"] = neg_scores.mean().item()
        avg["global_spread"] = (pos_scores.mean() - neg_scores.mean()).item()
        # 有多少比例的正样本分数 > 负样本均值
        avg["pos_above_neg_mean"] = (pos_scores > neg_scores.mean()).float().mean().item()

    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--action_dropout", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    cli_args = parser.parse_args()

    # config file overrides
    args = vars(cli_args)
    if cli_args.config and os.path.exists(cli_args.config):
        with open(cli_args.config) as f:
            config = json.load(f)
        args.update(config)

    device = torch.device(args["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(args["output_dir"], exist_ok=True)

    print("=" * 60)
    print("CQF Phase A Training (GT tactile, raw marker space)")
    print("=" * 60)
    for k, v in sorted(args.items()):
        print(f"  {k}: {v}")
    print()

    # Data
    print("Loading data...")
    t0 = time.time()
    train_loader, val_loader = build_cqf_dataloaders(
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
    )
    print(f"Data loaded in {time.time()-t0:.1f}s\n")

    # Model
    model = ContactQualityScorer(
        tac_dim=162,  # raw marker: 9*9*2
        hidden=args["hidden"],
        action_dropout=args["action_dropout"],
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss & Optimizer
    criterion = CQFLoss(margin=args["margin"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args["epochs"], eta_min=1e-6)

    # Training loop
    best_spread = -float("inf")
    for epoch in range(args["epochs"]):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args['epochs']} ({elapsed:.1f}s)  lr={lr:.2e}")
        print(f"  Train: loss={train_metrics['loss']:.4f}  "
              f"rank={train_metrics['loss_rank']:.4f}  "
              f"bce={train_metrics['loss_bce']:.4f}  "
              f"acc={train_metrics['accuracy']:.3f}  "
              f"spread={train_metrics['score_spread']:.3f}")
        print(f"  Val:   loss={val_metrics['loss']:.4f}  "
              f"acc={val_metrics['accuracy']:.3f}  "
              f"spread={val_metrics.get('global_spread', 0):.3f}  "
              f"pos>{val_metrics.get('pos_above_neg_mean', 0):.3f}")

        # Save best
        spread = val_metrics.get("global_spread", 0)
        if spread > best_spread:
            best_spread = spread
            save_path = os.path.join(args["output_dir"], "cqf_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "args": args,
            }, save_path)
            print(f"  ★ Best model saved (spread={spread:.4f})")

        # Save periodic
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(args["output_dir"], f"cqf_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": args,
            }, save_path)

    # Final save
    save_path = os.path.join(args["output_dir"], "cqf_final.pt")
    torch.save({
        "epoch": args["epochs"],
        "model_state_dict": model.state_dict(),
        "val_metrics": val_metrics,
        "args": args,
    }, save_path)
    print(f"\nTraining complete. Best spread: {best_spread:.4f}")
    print(f"Model saved to {args['output_dir']}")


if __name__ == "__main__":
    main()
