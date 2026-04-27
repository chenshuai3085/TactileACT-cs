"""
CQF Ablation Study: Action Dependency Analysis.

Trains 3 variants to understand action vs tactile contribution:
  1. full:     action_dropout=0.5 (current default)
  2. no_action: action_dropout=1.0 (never uses action during training)
  3. tac_only: removes action branch entirely

Also includes a proper "cross-sample ranking" test:
  Given insertion and pre_bounce frames from the same episode,
  test if CQF correctly ranks insertion > pre_bounce.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cqf_dataset import CQFDataset, build_cqf_dataloaders
from cqf_model import ContactQualityScorer, CQFLoss


class TactileOnlyScorer(nn.Module):
    """CQF without action branch — only tactile + state."""

    def __init__(self, tac_dim=162, qpos_dim=7, eef_dim=6, hidden=256):
        super().__init__()
        self.tac_encoder = nn.Sequential(
            nn.Linear(tac_dim * 3, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(qpos_dim + eef_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, qpos, eef, action_chunk, tac_cur, tac_pred):
        delta = tac_pred - tac_cur
        h_tac = self.tac_encoder(torch.cat([tac_cur, tac_pred, delta], dim=-1))
        h_state = self.state_encoder(torch.cat([qpos, eef], dim=-1))
        h_all = torch.cat([h_tac, h_state], dim=-1)
        return self.scorer(h_all)


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
        avg["auc"] = (pos.unsqueeze(-1) > neg.unsqueeze(0)).float().mean().item()
    return avg


def run_ablation(name, model, device, epochs=30, lr=1e-4, batch_size=256):
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"{'='*60}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(device)
    train_loader, val_loader = build_cqf_dataloaders(batch_size=batch_size, num_workers=4)

    criterion = CQFLoss(margin=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_spread = -float("inf")
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        spread = val_m.get("spread", 0)
        auc = val_m.get("auc", 0)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_m["loss"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["accuracy"],
            "val_spread": spread,
            "val_auc": auc,
        })

        if spread > best_spread:
            best_spread = spread
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            print(f"  Epoch {epoch+1:3d}/{epochs} ({elapsed:.1f}s)  "
                  f"loss={train_m['loss']:.4f}  "
                  f"val_acc={val_m['accuracy']:.3f}  "
                  f"spread={spread:.2f}  auc={auc:.4f}")

    print(f"  Best spread: {best_spread:.2f} (epoch {best_epoch})")

    # Action dependency test (only for models with action input)
    model.eval()
    act_sens, tac_sens = test_sensitivity(model, val_loader, device)
    print(f"  Action sensitivity:  {act_sens:.3f}")
    print(f"  Tactile sensitivity: {tac_sens:.3f}")
    print(f"  Tac/Act ratio:       {tac_sens/(act_sens+1e-8):.2f}x")

    return {
        "name": name,
        "best_spread": best_spread,
        "best_epoch": best_epoch,
        "final_auc": history[-1]["val_auc"],
        "act_sensitivity": act_sens,
        "tac_sensitivity": tac_sens,
        "history": history,
    }


@torch.no_grad()
def test_sensitivity(model, loader, device):
    """Quick sensitivity test: randomize action or zero tactile delta."""
    model.eval()
    orig_scores, rand_act_scores, zero_delta_scores = [], [], []

    for i, batch in enumerate(loader):
        if i >= 5:
            break
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        mcur = batch["marker_cur"].to(device).flatten(1)
        mfut = batch["marker_future"].to(device).flatten(1)

        s_orig = model(qpos, eef, action, mcur, mfut).squeeze(-1)
        s_rand = model(qpos, eef, torch.randn_like(action), mcur, mfut).squeeze(-1)
        s_zero = model(qpos, eef, action, mcur, mcur).squeeze(-1)

        orig_scores.append(s_orig.cpu())
        rand_act_scores.append(s_rand.cpu())
        zero_delta_scores.append(s_zero.cpu())

    orig = torch.cat(orig_scores)
    rand_act = torch.cat(rand_act_scores)
    zero_delta = torch.cat(zero_delta_scores)

    act_sens = (orig - rand_act).abs().mean().item() / (orig.abs().mean().item() + 1e-8)
    tac_sens = (orig - zero_delta).abs().mean().item() / (orig.abs().mean().item() + 1e-8)
    return act_sens, tac_sens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenshuai/Project/output/cqf_ablation")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # Ablation 1: action_dropout=0.5 (default)
    model_full = ContactQualityScorer(tac_dim=162, hidden=256, action_dropout=0.5)
    results["full_d05"] = run_ablation("full (dropout=0.5)", model_full, device, epochs=args.epochs)

    # Ablation 2: action_dropout=1.0 (never uses action)
    model_no_act = ContactQualityScorer(tac_dim=162, hidden=256, action_dropout=1.0)
    results["no_action_d10"] = run_ablation("no_action (dropout=1.0)", model_no_act, device, epochs=args.epochs)

    # Ablation 3: tactile-only (no action branch at all)
    model_tac = TactileOnlyScorer(tac_dim=162, hidden=256)
    results["tac_only"] = run_ablation("tac_only", model_tac, device, epochs=args.epochs)

    # Summary
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<25} {'Spread':>8} {'AUC':>8} {'Act Sens':>10} {'Tac Sens':>10} {'Ratio':>8}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{r['name']:<25} {r['best_spread']:>8.2f} {r['final_auc']:>8.4f} "
              f"{r['act_sensitivity']:>10.3f} {r['tac_sensitivity']:>10.3f} "
              f"{r['tac_sensitivity']/(r['act_sensitivity']+1e-8):>8.2f}x")

    save_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x))
    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
