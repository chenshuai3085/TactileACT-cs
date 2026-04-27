"""
CQF Training with Latent Foresight (Phase A/B/C unified).

Uses precomputed 144-dim TactileVAE latent vectors from precompute_cqf_latents.py.

Phase A: GT latent only (z_cur + z_future_gt)
Phase B: Mixed GT + Foresight (gt_ratio decays from 0.9 to 0.1)
Phase C: Pure Foresight (z_cur + z_pred)

Usage:
  # Phase A (from scratch)
  python TFAC_V5/train_cqf_latent.py --phase A --epochs 50

  # Phase B (warm-start from Phase A)
  python TFAC_V5/train_cqf_latent.py --phase B --epochs 30 \
    --cqf_ckpt /path/to/cqf_latent_best.pt

  # Phase C (warm-start from Phase B)
  python TFAC_V5/train_cqf_latent.py --phase C --epochs 10 \
    --cqf_ckpt /path/to/phase_b_best.pt
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, os.path.dirname(__file__))

from cqf_model import ContactQualityScorer, CQFLoss


class CQFLatentDataset(Dataset):
    """CQF dataset using precomputed 144-dim latent vectors."""

    def __init__(self, samples, gt_ratio=1.0, include_perturb=True):
        """
        Args:
            samples: list of dicts from precompute_cqf_latents.py
            gt_ratio: probability of using GT future vs foresight prediction
            include_perturb: whether to include perturbation negatives
        """
        if include_perturb:
            self.samples = samples
        else:
            self.samples = [s for s in samples if not s.get("is_perturb")]
        self.gt_ratio = gt_ratio

        n_pos = sum(1 for s in self.samples if s["label"] > 0.5)
        n_neg = len(self.samples) - n_pos
        print(f"CQFLatentDataset: {len(self.samples)} samples "
              f"(pos={n_pos}, neg={n_neg}, gt_ratio={gt_ratio:.2f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Choose between GT and foresight prediction
        if np.random.random() < self.gt_ratio:
            z_future = torch.from_numpy(s["z_future_gt"])
        else:
            z_future = torch.from_numpy(s["z_pred"])

        return {
            "qpos": torch.from_numpy(s["qpos"]),
            "eef": torch.from_numpy(s["eef"]),
            "action_chunk": torch.from_numpy(s["action_chunk"]),
            "z_cur": torch.from_numpy(s["z_cur"]),
            "z_future": z_future,
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }

    def get_weighted_sampler(self):
        labels = np.array([s["label"] for s in self.samples])
        n_pos = (labels > 0.5).sum()
        n_neg = (labels <= 0.5).sum()
        weight_pos = 1.0 / max(n_pos, 1)
        weight_neg = 1.0 / max(n_neg, 1)
        weights = np.where(labels > 0.5, weight_pos, weight_neg)
        return WeightedRandomSampler(weights, num_samples=len(self), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_metrics = {}
    n = 0
    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        z_cur = batch["z_cur"].to(device)
        z_future = batch["z_future"].to(device)
        labels = batch["label"].to(device)

        scores = model(qpos, eef, action, z_cur, z_future)
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
        z_cur = batch["z_cur"].to(device)
        z_future = batch["z_future"].to(device)
        labels = batch["label"].to(device)

        scores = model(qpos, eef, action, z_cur, z_future)
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
        avg["pos_above_neg_mean"] = (pos > neg.mean()).float().mean().item()

        # Ranking accuracy: sample K=16 groups, check if top-1 is positive
        rank_correct = 0
        rank_total = 0
        n_pos_available = len(pos)
        n_neg_available = len(neg)
        K = 16
        for _ in range(min(200, n_pos_available)):
            p_idx = torch.randint(n_pos_available, (1,)).item()
            n_indices = torch.randint(n_neg_available, (K - 1,))
            group_scores = torch.cat([pos[p_idx:p_idx+1], neg[n_indices]])
            if group_scores.argmax() == 0:
                rank_correct += 1
            rank_total += 1
        if rank_total > 0:
            avg["rank1_acc"] = rank_correct / rank_total
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent_data")
    parser.add_argument("--phase", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--action_dropout", type=float, default=0.5)
    parser.add_argument("--cqf_ckpt", type=str, default=None,
                        help="CQF checkpoint for warm-start (Phase B/C)")
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf_latent")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    phase = args.phase
    print("=" * 60)
    print(f"CQF Phase {phase} Training (Latent Space, tac_dim=144)")
    print("=" * 60)

    # Load precomputed data
    print("Loading precomputed latent data...")
    t0 = time.time()
    train_samples = torch.load(
        os.path.join(args.data_dir, "train_samples.pt"), weights_only=False)
    val_samples = torch.load(
        os.path.join(args.data_dir, "val_samples.pt"), weights_only=False)
    print(f"Loaded {len(train_samples)} train + {len(val_samples)} val "
          f"in {time.time()-t0:.1f}s")

    # Model
    tac_dim = 144  # TactileVAE latent: 16 * 3 * 3
    if args.cqf_ckpt:
        ckpt = torch.load(args.cqf_ckpt, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        model = ContactQualityScorer(
            tac_dim=ckpt_args.get("tac_dim", tac_dim),
            hidden=ckpt_args.get("hidden", args.hidden),
            action_dropout=ckpt_args.get("action_dropout", args.action_dropout),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded CQF from {args.cqf_ckpt} (epoch {ckpt.get('epoch', '?')})")
    else:
        model = ContactQualityScorer(
            tac_dim=tac_dim,
            hidden=args.hidden,
            action_dropout=args.action_dropout,
        ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CQFLoss(margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Save args
    save_args = vars(args).copy()
    save_args["tac_dim"] = tac_dim
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(save_args, f, indent=2)

    best_spread = -float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        # Determine gt_ratio for this epoch
        if phase == "A":
            gt_ratio = 1.0
            include_perturb = False
        elif phase == "B":
            gt_ratio = 0.9 - 0.8 * (epoch / max(args.epochs - 1, 1))
            include_perturb = True
        else:  # C
            gt_ratio = 0.0
            include_perturb = True

        # Build datasets (lightweight — just sets gt_ratio)
        train_ds = CQFLatentDataset(
            train_samples, gt_ratio=gt_ratio, include_perturb=include_perturb)
        val_ds = CQFLatentDataset(
            val_samples, gt_ratio=0.0, include_perturb=include_perturb)

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
        rank1 = val_m.get("rank1_acc", 0)

        print(f"Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e} "
              f"gt_ratio={gt_ratio:.2f}")
        print(f"  Train: loss={train_m['loss']:.4f} acc={train_m['accuracy']:.3f} "
              f"spread={train_m['score_spread']:.3f}")
        print(f"  Val:   loss={val_m['loss']:.4f} acc={val_m['accuracy']:.3f} "
              f"spread={spread:.3f} rank1={rank1:.3f}")

        if spread > best_spread:
            best_spread = spread
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "phase": phase,
                "args": save_args,
            }, os.path.join(args.output_dir, f"cqf_latent_best.pt"))
            print(f"  -> Best (spread={spread:.4f}, rank1={rank1:.3f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "phase": phase,
                "args": save_args,
            }, os.path.join(args.output_dir, f"cqf_latent_epoch{epoch+1}.pt"))

    # Final save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_m,
        "phase": phase,
        "args": save_args,
    }, os.path.join(args.output_dir, f"cqf_latent_final.pt"))

    print(f"\nPhase {phase} complete. Best: epoch {best_epoch}, spread={best_spread:.4f}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
