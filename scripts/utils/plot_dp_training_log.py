#!/usr/bin/env python3
"""Plot train/val loss curves from DP training logs."""
import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_RE = re.compile(r"Ep\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+)")
VAL_RE = re.compile(r"\|\s+val=([0-9.eE+-]+)")
BEST_RE = re.compile(r"\|\s+best=([a-zA-Z_]+)=([0-9.eE+-]+|pending)")


def parse_log(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_m = EPOCH_RE.search(line)
            best_m = BEST_RE.search(line)
            if not epoch_m or not best_m:
                continue
            val_m = VAL_RE.search(line)
            epoch = int(epoch_m.group(1))
            total = int(epoch_m.group(2))
            train = float(epoch_m.group(3))
            val = float(val_m.group(1)) if val_m is not None else None
            best_name = best_m.group(1)
            best = None if best_m.group(2) == "pending" else float(best_m.group(2))
            rows.append({
                "epoch": epoch,
                "total_epochs": total,
                "train_loss": train,
                "val_loss": val,
                "best_metric_name": best_name,
                "best_metric": best,
            })
    return rows


def moving_average(values, window):
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--smooth", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    rows = parse_log(args.log)
    if not rows:
        raise RuntimeError(f"No epoch summary lines found in {args.log}")

    csv_path = os.path.join(out_dir, "loss_curve.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    epochs = [r["epoch"] for r in rows]
    train = [r["train_loss"] for r in rows]
    val_epochs = [r["epoch"] for r in rows if r["val_loss"] is not None]
    val = [r["val_loss"] for r in rows if r["val_loss"] is not None]
    best = [r["best_metric"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=160)

    axes[0].plot(epochs, train, color="#2563eb", alpha=0.35, linewidth=1.0, label="train")
    axes[0].plot(epochs, moving_average(train, args.smooth), color="#1d4ed8", linewidth=2.0,
                 label=f"train smooth{args.smooth}")
    if val:
        axes[0].plot(val_epochs, val, color="#dc2626", alpha=0.5, linewidth=1.0, label="val")
        axes[0].plot(val_epochs, moving_average(val, args.smooth), color="#991b1b", linewidth=2.0,
                     label=f"val smooth{args.smooth}")
    axes[0].set_title("DP Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE noise prediction loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train, color="#2563eb", alpha=0.35, linewidth=1.0, label="train")
    if val:
        axes[1].plot(val_epochs, val, color="#dc2626", alpha=0.55, linewidth=1.0, label="val")
    valid_best = [(e, b) for e, b in zip(epochs, best) if b is not None]
    if valid_best:
        axes[1].plot([x[0] for x in valid_best], [x[1] for x in valid_best],
                     color="#16a34a", linewidth=1.6, label="best val")
    axes[1].set_yscale("log")
    axes[1].set_title("Loss Log Scale")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("loss")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    latest = rows[-1]
    best_val = min((r["val_loss"] for r in rows if r["val_loss"] is not None), default=None)
    subtitle = f"latest epoch {latest['epoch']}/{latest['total_epochs']}, train={latest['train_loss']:.6f}"
    if latest["val_loss"] is not None:
        subtitle += f", val={latest['val_loss']:.6f}"
    if best_val is not None:
        subtitle += f", best_val={best_val:.6f}"
    fig.suptitle(subtitle, fontsize=10)
    fig.tight_layout()

    png_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(png_path)
    print(f"Parsed {len(rows)} epochs")
    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
