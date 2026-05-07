"""
CQF Training with DP-generated candidates (group-wise ranking).

Uses precomputed data from precompute_cqf_dp_candidates.py:
  - Each group = K=16 candidates from same frame
  - Labels are continuous [0, 1] (normalized L1 to expert)
  - Training: CQFLoss per group, then average

Usage:
  python TFAC_V5/train_cqf_dp.py \
    --data_dir /home/chenshuai/Project/output/cqf_dp_candidates \
    --output_dir /home/chenshuai/Project/output/cqf_dp_trained \
    --epochs 50 --batch_groups 16 --K 16 \
    --lr 1e-4 --margin 0.5 --hidden 256
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader, Sampler

import sys
sys.path.insert(0, os.path.dirname(__file__))

from cqf_model import ContactQualityScorer, CQFLoss


class CQFDPDataset(Dataset):
    """CQF dataset from DP-generated candidates. Returns individual samples."""

    def __init__(self, samples, K=16):
        self.samples = samples
        self.K = K
        n_groups = len(samples) // K
        n_pos = sum(1 for s in samples if s["label"] > 0.5)
        print(f"CQFDPDataset: {len(samples)} samples, {n_groups} groups, "
              f"K={K}, pos(>0.5)={n_pos}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "qpos": torch.from_numpy(s["qpos"]),
            "action_chunk": torch.from_numpy(s["action_chunk"]),
            "z_cur": torch.from_numpy(s["z_cur"]),
            "z_pred": torch.from_numpy(s["z_pred"]),
            "label": torch.tensor(s["label"], dtype=torch.float32),
            "group_id": s["group_id"],
        }


class GroupBatchSampler(Sampler):
    """Samples complete groups (K samples each). Each batch = N groups × K samples.

    Ensures all K candidates from the same frame are in the same batch
    so CQFLoss can compute intra-group ranking.
    """

    def __init__(self, dataset, K, batch_groups, shuffle=True, drop_last=True):
        self.K = K
        self.batch_groups = batch_groups
        self.drop_last = drop_last
        self.shuffle = shuffle

        n_samples = len(dataset)
        self.n_groups = n_samples // K
        self.group_starts = [i * K for i in range(self.n_groups)]

    def __iter__(self):
        group_order = list(range(self.n_groups))
        if self.shuffle:
            np.random.shuffle(group_order)

        batch = []
        for g_idx in group_order:
            start = self.group_starts[g_idx]
            batch.extend(range(start, start + self.K))
            if len(batch) == self.batch_groups * self.K:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.n_groups // self.batch_groups
        return (self.n_groups + self.batch_groups - 1) // self.batch_groups


def train_one_epoch(model, loader, criterion, optimizer, device, K, batch_groups):
    model.train()
    total_loss = 0
    total_rank_acc = 0
    total_spread = 0
    n_batches = 0

    for batch in loader:
        qpos = batch["qpos"].to(device)
        action = batch["action_chunk"].to(device)
        z_cur = batch["z_cur"].to(device)
        z_pred = batch["z_pred"].to(device)
        labels = batch["label"].to(device)

        B = qpos.shape[0]
        N = B // K

        scores = model(qpos, action, z_cur, z_pred)

        scores_groups = scores.view(N, K, 1)
        labels_groups = labels.view(N, K)

        loss_sum = torch.tensor(0.0, device=device)
        batch_rank_acc = 0
        batch_spread = 0

        for g in range(N):
            loss_g, metrics_g = criterion(scores_groups[g], labels_groups[g])
            loss_sum = loss_sum + loss_g
            batch_rank_acc += metrics_g["pairwise_acc"]
            batch_spread += metrics_g["score_spread"]

        loss = loss_sum / N

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_rank_acc += batch_rank_acc / N
        total_spread += batch_spread / N
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "pairwise_acc": total_rank_acc / max(n_batches, 1),
        "spread": total_spread / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, K):
    model.eval()
    total_loss = 0
    n_batches = 0

    all_group_scores = []
    all_group_labels = []

    for batch in loader:
        qpos = batch["qpos"].to(device)
        action = batch["action_chunk"].to(device)
        z_cur = batch["z_cur"].to(device)
        z_pred = batch["z_pred"].to(device)
        labels = batch["label"].to(device)

        B = qpos.shape[0]
        N = B // K

        scores = model(qpos, action, z_cur, z_pred)

        scores_groups = scores.view(N, K, 1)
        labels_groups = labels.view(N, K)

        loss_sum = 0
        for g in range(N):
            loss_g, _ = criterion(scores_groups[g], labels_groups[g])
            loss_sum += loss_g.item()

            all_group_scores.append(scores_groups[g].squeeze(-1).cpu().numpy())
            all_group_labels.append(labels_groups[g].cpu().numpy())

        total_loss += loss_sum / N
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Group-level metrics
    rank1_correct = 0
    top3_correct = 0
    correlations = []
    spreads = []

    for scores_g, labels_g in zip(all_group_scores, all_group_labels):
        score_order = np.argsort(-scores_g)
        label_order = np.argsort(-labels_g)

        oracle_idx = label_order[0]
        cqf_top1 = score_order[0]
        cqf_top3 = set(score_order[:3].tolist())

        if cqf_top1 == oracle_idx:
            rank1_correct += 1
        if oracle_idx in cqf_top3:
            top3_correct += 1

        if labels_g.std() > 1e-6 and scores_g.std() > 1e-6:
            corr, _ = spearmanr(scores_g, labels_g)
            if not np.isnan(corr):
                correlations.append(corr)

        spreads.append(scores_g.max() - scores_g.min())

    n_groups = len(all_group_scores)
    metrics = {
        "loss": avg_loss,
        "rank1_acc": rank1_correct / max(n_groups, 1),
        "top3_acc": top3_correct / max(n_groups, 1),
        "spearman_corr": np.mean(correlations) if correlations else 0.0,
        "spread": np.mean(spreads) if spreads else 0.0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_groups", type=int, default=16,
                        help="Number of groups per batch (batch_size = batch_groups × K)")
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--action_dropout", type=float, default=0.5)
    parser.add_argument("--cqf_ckpt", type=str, default=None,
                        help="Warm-start from existing CQF checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CQF Training with DP Candidates (Group-wise Ranking)")
    print("=" * 60)
    print(f"  K={args.K}, batch_groups={args.batch_groups}, "
          f"batch_size={args.batch_groups * args.K}")

    # Load data
    print("\nLoading precomputed DP candidate data...")
    t0 = time.time()
    train_samples = torch.load(
        os.path.join(args.data_dir, "train_dp_samples.pt"), weights_only=False)
    val_samples = torch.load(
        os.path.join(args.data_dir, "val_dp_samples.pt"), weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s: "
          f"train={len(train_samples)}, val={len(val_samples)}")

    # Load meta for chunk_size
    meta_path = os.path.join(args.data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        chunk_size = meta.get("cqf_chunk_size", 20)
        tac_dim = meta.get("tac_dim", 144)
    else:
        chunk_size = train_samples[0]["action_chunk"].shape[0]
        tac_dim = train_samples[0]["z_cur"].shape[0]

    print(f"  chunk_size={chunk_size}, tac_dim={tac_dim}")

    # Model
    if args.cqf_ckpt:
        ckpt = torch.load(args.cqf_ckpt, map_location=device, weights_only=False)
        ckpt_args = ckpt.get("args", {})
        model = ContactQualityScorer(
            tac_dim=ckpt_args.get("tac_dim", tac_dim),
            chunk_size=ckpt_args.get("cqf_chunk_size", chunk_size),
            hidden=ckpt_args.get("hidden", args.hidden),
            action_dropout=ckpt_args.get("action_dropout", args.action_dropout),
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Warm-start from {args.cqf_ckpt} (epoch {ckpt.get('epoch', '?')})")
    else:
        model = ContactQualityScorer(
            tac_dim=tac_dim,
            chunk_size=chunk_size,
            hidden=args.hidden,
            action_dropout=args.action_dropout,
        ).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = CQFLoss(base_margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # Datasets + samplers
    train_ds = CQFDPDataset(train_samples, K=args.K)
    val_ds = CQFDPDataset(val_samples, K=args.K)

    train_sampler = GroupBatchSampler(
        train_ds, K=args.K, batch_groups=args.batch_groups,
        shuffle=True, drop_last=True)
    val_sampler = GroupBatchSampler(
        val_ds, K=args.K, batch_groups=args.batch_groups,
        shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True)

    # Save args
    save_args = vars(args).copy()
    save_args["tac_dim"] = tac_dim
    save_args["chunk_size"] = chunk_size
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(save_args, f, indent=2)

    best_rank1 = -1.0
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  {len(train_sampler)} train batches, {len(val_sampler)} val batches per epoch")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            K=args.K, batch_groups=args.batch_groups)
        val_m = validate(model, val_loader, criterion, device, K=args.K)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e}")
        print(f"  Train: loss={train_m['loss']:.4f} "
              f"pair_acc={train_m['pairwise_acc']:.3f} "
              f"spread={train_m['spread']:.3f}")
        print(f"  Val:   loss={val_m['loss']:.4f} "
              f"rank1={val_m['rank1_acc']:.3f} top3={val_m['top3_acc']:.3f} "
              f"corr={val_m['spearman_corr']:.3f} spread={val_m['spread']:.3f}")

        rank1 = val_m["rank1_acc"]
        if rank1 > best_rank1:
            best_rank1 = rank1
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "args": save_args,
            }, os.path.join(args.output_dir, "cqf_latent_best.pt"))
            print(f"  -> Best (rank1={rank1:.3f}, corr={val_m['spearman_corr']:.3f})")

        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "args": save_args,
            }, os.path.join(args.output_dir, f"cqf_epoch{epoch+1}.pt"))

    # Final save
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_m,
        "args": save_args,
    }, os.path.join(args.output_dir, "cqf_final.pt"))

    print(f"\nDone! Best: epoch {best_epoch}, rank1={best_rank1:.3f}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
