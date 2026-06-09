"""Train a marker-field tactile quality scorer for DP guidance.

This experiment is the next step after the 18-D marker proxy scorer.  The proxy
features are interpretable, but they compress away spatial/temporal details that
matter for contact quality.  Here the scorer consumes both:

  1. raw marker_offset window (T, 9, 9, 2), through a small CNN/temporal encoder;
  2. differentiable physical proxy features, for calibration and interpretability;
  3. task id, so insertion and board wiping can share a body but keep calibrated
     decision boundaries.

The deployed objective remains multi-head:
  - binary good/bad: main classifier-guidance target;
  - T4 contact reason: auxiliary interpretability target;
  - continuous quality score: ranking/calibration target.

For board wiping, too_light is treated as bad in the binary head because the user
defined both too-small and too-large force as bad.  For socket insertion,
approach/no-contact remains neutral for the binary head.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.evaluate_marker_proxy_scorer import (  # noqa: E402
    T4_NAMES,
    assign_board_t4,
    build_insertion_samples,
    board_windows,
    marker_proxy_features,
)


INSERTION_DIR = Path("/home/chenshuai/data/dataset/0414")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/marker_field_scorer")
WINDOW = 8
TASK_TO_ID = {"insertion": 0, "board": 1}


def _pad_or_crop_window(marker_seq: np.ndarray, window: int = WINDOW) -> np.ndarray:
    marker_seq = np.asarray(marker_seq, dtype=np.float32)
    if len(marker_seq) >= window:
        return marker_seq[-window:]
    pad = np.repeat(marker_seq[:1], window - len(marker_seq), axis=0)
    return np.concatenate([pad, marker_seq], axis=0)


def build_insertion_field_samples(max_per_episode=220):
    annotations = pickle.load(open(INSERTION_DIR / "annotations.pkl", "rb"))
    rows = []
    proxy_rows = build_insertion_samples(max_per_episode=max_per_episode)
    lookup = {r["sample_id"]: r for r in proxy_rows}
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        path = INSERTION_DIR / f"{ep_name}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            marker = f["observations/tac/left/marker_offset"][:]
        labels = np.asarray(info["labels"])
        n = min(len(marker), len(labels))
        indices = np.arange(7, n)
        if max_per_episode and len(indices) > max_per_episode:
            rng = np.random.default_rng(abs(hash(ep_name)) % (2**32))
            indices = np.sort(rng.choice(indices, max_per_episode, replace=False))
        for frame_idx in indices:
            sample_id = f"insertion/{ep_name}/{frame_idx}"
            base = lookup.get(sample_id)
            if base is None:
                continue
            window = _pad_or_crop_window(marker[max(0, frame_idx - WINDOW + 1) : frame_idx + 1])
            rows.append(
                {
                    **base,
                    "marker": window,
                }
            )
    return rows


def build_board_field_samples():
    rows = board_windows(window=32, stride=16)
    labels, thresholds = assign_board_t4(rows)
    samples = []
    for row, (t4, raw) in zip(rows, labels):
        marker_seq = np.asarray(row["marker_seq"], dtype=np.float32)
        short_window = _pad_or_crop_window(marker_seq)
        binary = 1 if t4 == 1 else 0
        # Continuous score follows the user-defined quality: good highest;
        # too_light is bad but less dangerous than too_heavy/rough.
        score = {0: 0.15, 1: 1.0, 2: 0.02, 3: 0.0}[t4]
        samples.append(
            {
                "sample_id": f"board/{row['episode']}/{row['start']}_{row['end']}",
                "task": "board",
                "episode": row["episode"],
                "t4": t4,
                "binary": binary,
                "score": score,
                "raw_label": raw,
                "feature": marker_proxy_features(marker_seq),
                "marker": short_window,
                "force_mean": row["force_mean"],
                "force_p95": row["force_p95"],
                "force_delta": row["force_delta"],
                "action_delta": row["action_delta"],
            }
        )
    return samples, thresholds


def build_or_load_dataset(force_rebuild=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "marker_field_features.npz"
    meta_path = OUT_DIR / "marker_field_metadata.json"
    sample_path = OUT_DIR / "marker_field_samples.csv"
    if data_path.exists() and not force_rebuild:
        data = np.load(data_path, allow_pickle=True)
        meta = json.load(open(meta_path, encoding="utf-8"))
        return data, meta

    insertion = build_insertion_field_samples()
    board, thresholds = build_board_field_samples()
    samples = insertion + board
    markers = np.stack([s["marker"] for s in samples]).astype(np.float32)
    proxies = np.stack([s["feature"] for s in samples]).astype(np.float32)
    y_t4 = np.array([s["t4"] for s in samples], dtype=np.int64)
    y_bin = np.array([s["binary"] for s in samples], dtype=np.int64)
    score = np.array([s["score"] for s in samples], dtype=np.float32)
    task = np.array([s["task"] for s in samples])
    task_id = np.array([TASK_TO_ID[s["task"]] for s in samples], dtype=np.int64)
    groups = np.array([f"{s['task']}::{s['episode']}" for s in samples])
    sample_ids = np.array([s["sample_id"] for s in samples])

    np.savez_compressed(
        data_path,
        marker=markers,
        proxy=proxies,
        y_t4=y_t4,
        y_binary=y_bin,
        score=score,
        task=task,
        task_id=task_id,
        groups=groups,
        sample_ids=sample_ids,
    )
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "task",
                "episode",
                "t4",
                "t4_name",
                "binary",
                "score",
                "raw_label",
            ],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "task": s["task"],
                    "episode": s["episode"],
                    "t4": s["t4"],
                    "t4_name": T4_NAMES[s["t4"]],
                    "binary": s["binary"],
                    "score": s["score"],
                    "raw_label": s["raw_label"],
                }
            )
    meta = {
        "window": WINDOW,
        "task_to_id": TASK_TO_ID,
        "t4_names": T4_NAMES,
        "board_thresholds": thresholds,
        "binary_definition": {
            "insertion": "good=success insert, bad=bounce pre-bounce/bounce/recovery, approach neutral/excluded",
            "board": "good=force suitable and smooth, bad=too_light/too_heavy/rough",
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return np.load(data_path, allow_pickle=True), meta


def balanced_indices(y_t4, task, max_per_task_class=1400, seed=42):
    rng = np.random.default_rng(seed)
    keep = []
    for task_name in np.unique(task):
        for cls in np.unique(y_t4):
            idx = np.flatnonzero((task == task_name) & (y_t4 == cls))
            if len(idx):
                keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


class MarkerFieldScorer(nn.Module):
    def __init__(self, proxy_dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.field_encoder = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
        )
        field_dim = 48 * 2 * 3 * 3
        self.encoder = nn.Sequential(
            nn.Linear(field_dim + proxy_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.t4_head = nn.Linear(hidden // 2, 4)
        self.score_head = nn.Linear(hidden // 2, 1)

    def forward(self, marker, proxy, task_id):
        # marker: (B, T, 9, 9, 2) -> (B, 2, T, 9, 9)
        marker = marker.permute(0, 4, 1, 2, 3)
        field = self.field_encoder(marker)
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([field, proxy, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "t4_logits": self.t4_head(h),
            "score": self.score_head(h).squeeze(-1),
        }


def make_loader(marker, proxy, task_id, y_bin, y_t4, score, idx, batch_size=256, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(marker[idx]).float(),
        torch.from_numpy(proxy[idx]).float(),
        torch.from_numpy(task_id[idx]).long(),
        torch.from_numpy(y_bin[idx]).long(),
        torch.from_numpy(y_t4[idx]).long(),
        torch.from_numpy(score[idx]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def standardize_marker(train_marker, all_marker):
    mean = train_marker.reshape(-1, 2).mean(axis=0).astype(np.float32)
    std = train_marker.reshape(-1, 2).std(axis=0).astype(np.float32) + 1e-6
    out = (all_marker - mean.reshape(1, 1, 1, 1, 2)) / std.reshape(1, 1, 1, 1, 2)
    return out.astype(np.float32), mean, std


def standardize_proxy(train_proxy, all_proxy):
    scaler = StandardScaler()
    scaler.fit(train_proxy)
    return scaler.transform(all_proxy).astype(np.float32), scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


def train_epoch(model, loader, opt, device, weights):
    model.train()
    total = 0.0
    n = 0
    for marker, proxy, task_id, y_bin, y_t4, score in loader:
        marker = marker.to(device)
        proxy = proxy.to(device)
        task_id = task_id.to(device)
        y_bin = y_bin.to(device)
        y_t4 = y_t4.to(device)
        score = score.to(device)
        out = model(marker, proxy, task_id)
        loss_bin = F.cross_entropy(out["binary_logits"], y_bin)
        loss_t4 = F.cross_entropy(out["t4_logits"], y_t4)
        loss_score = F.smooth_l1_loss(torch.sigmoid(out["score"]), score)
        loss = weights["binary"] * loss_bin + weights["t4"] * loss_t4 + weights["score"] * loss_score
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        bs = len(marker)
        total += float(loss.item()) * bs
        n += bs
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys_b, ys_t, qs, ps_b, ps_t, ps_q = [], [], [], [], [], []
    for marker, proxy, task_id, y_bin, y_t4, score in loader:
        out = model(marker.to(device), proxy.to(device), task_id.to(device))
        ys_b.append(y_bin.numpy())
        ys_t.append(y_t4.numpy())
        qs.append(score.numpy())
        ps_b.append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
        ps_t.append(torch.softmax(out["t4_logits"], dim=-1).cpu().numpy())
        ps_q.append(torch.sigmoid(out["score"]).cpu().numpy())
    yb = np.concatenate(ys_b)
    yt = np.concatenate(ys_t)
    q = np.concatenate(qs)
    pb = np.concatenate(ps_b)
    pt = np.concatenate(ps_t)
    pq = np.concatenate(ps_q)
    pred_b = (pb >= 0.5).astype(np.int64)
    pred_t = pt.argmax(axis=1)
    row = {
        "binary_balanced_accuracy": float(balanced_accuracy_score(yb, pred_b)),
        "binary_macro_f1": float(f1_score(yb, pred_b, average="macro")),
        "binary_auc": float(roc_auc_score(yb, pb)) if len(np.unique(yb)) == 2 else None,
        "t4_balanced_accuracy": float(balanced_accuracy_score(yt, pred_t)),
        "t4_macro_f1": float(f1_score(yt, pred_t, average="macro")),
        "score_corr": float(np.corrcoef(pq, q)[0, 1]) if np.std(pq) > 1e-8 and np.std(q) > 1e-8 else 0.0,
        "binary_confusion": confusion_matrix(yb, pred_b, labels=[0, 1]).tolist(),
        "t4_confusion": confusion_matrix(yt, pred_t, labels=[0, 1, 2, 3]).tolist(),
    }
    return row


def scalar_metric(row):
    auc = row["binary_auc"] or 0.0
    return auc + 0.25 * row["score_corr"] + 0.1 * row["t4_macro_f1"]


def train_eval_split(marker, proxy, task_id, y_bin, y_t4, score, train_idx, test_idx, device, epochs=70):
    marker_scaled, marker_mean, marker_std = standardize_marker(marker[train_idx], marker)
    proxy_scaled, proxy_mean, proxy_scale = standardize_proxy(proxy[train_idx], proxy)
    train_loader = make_loader(marker_scaled, proxy_scaled, task_id, y_bin, y_t4, score, train_idx, shuffle=True)
    test_loader = make_loader(marker_scaled, proxy_scaled, task_id, y_bin, y_t4, score, test_idx, shuffle=False)

    model = MarkerFieldScorer(proxy_dim=proxy.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    weights = {"binary": 1.0, "t4": 0.45, "score": 0.55}
    best = None
    best_state = None
    best_metric = -math.inf
    patience = 12
    bad = 0
    for _ in range(epochs):
        train_epoch(model, train_loader, opt, device, weights)
        sched.step()
        metrics = evaluate(model, test_loader, device)
        m = scalar_metric(metrics)
        if m > best_metric:
            best_metric = m
            best = metrics
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best, best_state, {
        "marker_mean": marker_mean,
        "marker_std": marker_std,
        "proxy_mean": proxy_mean,
        "proxy_scale": proxy_scale,
    }


def aggregate(rows):
    out = {}
    keys = [
        "binary_balanced_accuracy",
        "binary_macro_f1",
        "binary_auc",
        "t4_balanced_accuracy",
        "t4_macro_f1",
        "score_corr",
    ]
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
    return out


def run_experiment(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, meta = build_or_load_dataset(force_rebuild=args.force_rebuild)
    marker = data["marker"].astype(np.float32)
    proxy = data["proxy"].astype(np.float32)
    y_t4 = data["y_t4"].astype(np.int64)
    y_bin = data["y_binary"].astype(np.int64)
    score = data["score"].astype(np.float32)
    task = data["task"]
    task_id = data["task_id"].astype(np.int64)
    groups = data["groups"]

    valid = y_bin >= 0
    keep_local = balanced_indices(y_t4[valid], task[valid], max_per_task_class=args.max_per_task_class, seed=args.seed)
    keep = np.flatnonzero(valid)[keep_local]
    marker = marker[keep]
    proxy = proxy[keep]
    y_t4 = y_t4[keep]
    y_bin = y_bin[keep]
    score = score[keep]
    task = task[keep]
    task_id = task_id[keep]
    groups = groups[keep]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results = {
        "data": {
            "n": int(len(marker)),
            "marker_shape": list(marker.shape[1:]),
            "proxy_dim": int(proxy.shape[1]),
            "task_counts": {k: int(v) for k, v in Counter(task.tolist()).items()},
            "binary_counts": {str(k): int(v) for k, v in Counter(y_bin.tolist()).items()},
            "t4_counts": {str(k): int(v) for k, v in Counter(y_t4.tolist()).items()},
            "max_per_task_class": args.max_per_task_class,
        },
        "meta": meta,
        "mixed_group_cv": {},
        "cross_task": {},
    }

    rows = []
    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(cv.split(marker, y_bin, groups)):
        print(f"fold {fold}", flush=True)
        row, _, _ = train_eval_split(marker, proxy, task_id, y_bin, y_t4, score, tr, te, device, epochs=args.epochs)
        row["fold"] = fold
        rows.append(row)
        print(row, flush=True)
    results["mixed_group_cv"] = aggregate(rows)
    results["mixed_group_cv"]["folds"] = rows

    for train_task, test_task in [("insertion", "board"), ("board", "insertion")]:
        tr = np.flatnonzero(task == train_task)
        te = np.flatnonzero(task == test_task)
        row, _, _ = train_eval_split(
            marker, proxy, task_id, y_bin, y_t4, score, tr, te, device, epochs=args.epochs
        )
        results["cross_task"][f"{train_task}_to_{test_task}"] = row
        print(f"cross {train_task}->{test_task}: {row}", flush=True)

    # Final deployable mixed-task model.
    all_idx = np.arange(len(marker))
    marker_scaled, marker_mean, marker_std = standardize_marker(marker, marker)
    proxy_scaled, proxy_mean, proxy_scale = standardize_proxy(proxy, proxy)
    loader = make_loader(marker_scaled, proxy_scaled, task_id, y_bin, y_t4, score, all_idx, shuffle=True)
    final_model = MarkerFieldScorer(proxy_dim=proxy.shape[1]).to(device)
    opt = torch.optim.AdamW(final_model.parameters(), lr=2e-3, weight_decay=1e-4)
    weights = {"binary": 1.0, "t4": 0.45, "score": 0.55}
    for _ in range(args.final_epochs):
        train_epoch(final_model, loader, opt, device, weights)
    train_metrics = evaluate(final_model, make_loader(marker_scaled, proxy_scaled, task_id, y_bin, y_t4, score, all_idx, shuffle=False), device)
    results["final_train_metrics_capacity_check"] = train_metrics

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "marker_field_scorer_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "proxy_dim": int(proxy.shape[1]),
            "marker_mean": marker_mean,
            "marker_std": marker_std,
            "proxy_mean": proxy_mean,
            "proxy_scale": proxy_scale,
            "task_to_id": TASK_TO_ID,
            "window": WINDOW,
            "metrics_train": train_metrics,
            "heads": ["binary_logits", "t4_logits", "score"],
        },
        OUT_DIR / "marker_field_scorer_final.pt",
    )
    with open(OUT_DIR / "marker_field_scorer_final_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": train_metrics,
                "n_train": int(len(marker)),
                "task_counts": results["data"]["task_counts"],
                "binary_counts": results["data"]["binary_counts"],
                "t4_counts": results["data"]["t4_counts"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved {OUT_DIR / 'marker_field_scorer_eval.json'}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--final_epochs", type=int, default=55)
    parser.add_argument("--max_per_task_class", type=int, default=1400)
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
