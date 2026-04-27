"""
Train Lightweight Foresight V2: FiLM-conditioned action + contrastive loss.

V2 improvements over V1:
  1. FiLM conditioning: action modulates features instead of concatenation
  2. Action-contrastive loss: different actions → different predictions
  3. Periodic action sensitivity test during training

Usage:
  python TFAC_V5/train_lightweight_foresight_v2.py --epochs 100 --device cuda:0
"""

import argparse
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lightweight_foresight_v2 import LightweightForesightV2, ForesightV2Loss


DATA_ROOT = "/home/chenshuai/data/dataset"
ALL_DATASETS = [
    "260309", "260310", "260401", "260402", "260403", "260407",
    "0414", "260401_0402", "260408_0409", "260417",
    "0209-0210", "0331", "260309_0310",
]
FORESIGHT_HORIZON = 10
CHUNK_SIZE = 20


class ForesightDatasetV2(Dataset):
    """Dataset that also provides perturbed actions for contrastive training."""

    def __init__(self, data_root=DATA_ROOT, dataset_names=None,
                 foresight_horizon=FORESIGHT_HORIZON, chunk_size=CHUNK_SIZE,
                 split="train", val_ratio=0.15, seed=42):
        super().__init__()
        if dataset_names is None:
            dataset_names = ALL_DATASETS

        h = foresight_horizon
        cs = chunk_size
        samples = []
        action_stds = []

        for ds_name in dataset_names:
            ds_dir = os.path.join(data_root, ds_name)
            hdf5_files = self._find_all_hdf5(ds_dir)

            for hdf5_path in hdf5_files:
                try:
                    with h5py.File(hdf5_path, "r") as f:
                        qpos_all = f["observations/proprio_joint"][:].astype(np.float32)
                        eef_all = f["observations/proprio_eef"][:].astype(np.float32)
                        action_all = f["actions/joint_abs"][:].astype(np.float32)
                        marker_all = f["observations/tac/left/marker_offset"][:].astype(np.float32)
                except Exception:
                    continue

                T = qpos_all.shape[0]
                if len(action_stds) < 100:
                    action_stds.append(action_all.std(axis=0))

                for t in range(0, T - max(cs, h), 2):
                    if t + cs > T or t + h >= T:
                        continue
                    samples.append({
                        "qpos": qpos_all[t],
                        "eef": eef_all[t],
                        "action_chunk": action_all[t:t+cs],
                        "marker_cur": marker_all[t].flatten(),
                        "marker_future": marker_all[t+h].flatten(),
                    })

        self.action_std = np.mean(action_stds, axis=0) if action_stds else np.ones(7)

        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(samples))
        n_val = int(len(samples) * val_ratio)
        if split == "val":
            indices = indices[:n_val]
        else:
            indices = indices[n_val:]

        self.samples = [samples[i] for i in indices]
        self.split = split
        print(f"ForesightDatasetV2 ({split}): {len(self.samples)} samples")

    def _find_all_hdf5(self, ds_dir):
        import glob
        files = []
        for pattern in [os.path.join(ds_dir, "*.hdf5"),
                        os.path.join(ds_dir, "success", "*.hdf5"),
                        os.path.join(ds_dir, "bounce", "*.hdf5")]:
            files.extend(glob.glob(pattern))
        return sorted(set(files))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        action = s["action_chunk"].copy()

        if self.split == "train":
            noise_scale = np.random.exponential(0.5)
            noise = np.random.randn(*action.shape).astype(np.float32)
            perturbed = action + noise * self.action_std * noise_scale
            action_diff = np.linalg.norm((perturbed - action).flatten())
        else:
            perturbed = action.copy()
            action_diff = 0.0

        return {
            "qpos": torch.from_numpy(s["qpos"]),
            "eef": torch.from_numpy(s["eef"]),
            "action_chunk": torch.from_numpy(action),
            "action_perturbed": torch.from_numpy(perturbed),
            "action_diff_norm": torch.tensor(action_diff, dtype=torch.float32),
            "marker_cur": torch.from_numpy(s["marker_cur"]),
            "marker_future": torch.from_numpy(s["marker_future"]),
        }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_metrics = {}
    n = 0
    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        action_pert = batch["action_perturbed"].to(device)
        action_diff = batch["action_diff_norm"].to(device)
        mcur = batch["marker_cur"].to(device)
        mfut = batch["marker_future"].to(device)

        pred = model(qpos, eef, action, mcur)
        pred_pert = model(qpos, eef, action_pert, mcur)
        loss, metrics = criterion(pred, mfut, mcur, pred_pert, action_diff)

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
    for batch in loader:
        qpos = batch["qpos"].to(device)
        eef = batch["eef"].to(device)
        action = batch["action_chunk"].to(device)
        mcur = batch["marker_cur"].to(device)
        mfut = batch["marker_future"].to(device)

        pred = model(qpos, eef, action, mcur)
        loss, metrics = criterion(pred, mfut, mcur)

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0) + v
        n += 1
    return {k: v / n for k, v in total_metrics.items()}


@torch.no_grad()
def action_sensitivity_test(model, val_ds, device, n_samples=200, n_variants=4):
    """Quick action sensitivity test during training."""
    model.eval()
    diffs = []
    rng = np.random.RandomState(0)
    indices = rng.choice(len(val_ds), min(n_samples, len(val_ds)), replace=False)

    for idx in indices:
        s = val_ds.samples[idx]
        qpos = torch.from_numpy(s["qpos"][None]).to(device)
        eef = torch.from_numpy(s["eef"][None]).to(device)
        action = torch.from_numpy(s["action_chunk"][None]).to(device)
        mcur = torch.from_numpy(s["marker_cur"][None]).to(device)

        pred_gt = model(qpos, eef, action, mcur)

        for _ in range(n_variants):
            noise = torch.randn_like(action) * action.std() * 1.0
            pred_noisy = model(qpos, eef, action + noise, mcur)
            diff = (pred_gt - pred_noisy).norm().item()
            diffs.append(diff)

    return np.mean(diffs), np.std(diffs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--contrastive_weight", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str,
                        default="/home/chenshuai/Project/output/lightweight_foresight_v2")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Lightweight Foresight V2 Training (FiLM + Contrastive)")
    print("=" * 60)

    print("\nLoading data...")
    t0 = time.time()
    train_ds = ForesightDatasetV2(split="train")
    val_ds = ForesightDatasetV2(split="val")
    print(f"Data loaded in {time.time()-t0:.1f}s\n")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    model = LightweightForesightV2(
        hidden=args.hidden, n_layers=args.n_layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    criterion = ForesightV2Loss(
        delta_weight=0.3, contrastive_weight=args.contrastive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} ({elapsed:.1f}s) lr={lr:.2e}")
            print(f"  Train: loss={train_m['loss']:.4f} mse={train_m['mse']:.6f} "
                  f"delta_cos={train_m['delta_cosine']:.3f} "
                  f"L_ctr={train_m['loss_contrastive']:.4f}")
            print(f"  Val:   loss={val_m['loss']:.4f} mse={val_m['mse']:.6f} "
                  f"delta_cos={val_m['delta_cosine']:.3f}")

        if (epoch + 1) % 10 == 0:
            mean_diff, std_diff = action_sensitivity_test(model, val_ds, device)
            print(f"  Action Sensitivity: pred_diff={mean_diff:.4f}±{std_diff:.4f}")

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }, os.path.join(args.output_dir, "foresight_v2_best.pt"))
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  ★ Best model saved (loss={best_val_loss:.4f})")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_m,
                "args": vars(args),
            }, os.path.join(args.output_dir, f"foresight_v2_epoch{epoch+1}.pt"))

    mean_diff, std_diff = action_sensitivity_test(model, val_ds, device)
    print(f"\nFinal Action Sensitivity: pred_diff={mean_diff:.4f}±{std_diff:.4f}")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_metrics": val_m,
        "args": vars(args),
    }, os.path.join(args.output_dir, "foresight_v2_final.pt"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
