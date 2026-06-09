"""Train an insertion-specific tactile risk scorer for DP guidance.

The unified PTG-v2 scorer is useful for board wiping but weak on insertion DP
candidate ranking.  This script trains a task-specialized scorer with a sharper
target:

  good insertion contact vs pre-bounce / bounce-risk contact.

Inputs are deliberately aligned with the current insertion DP/Foresight path:
  left marker_offset window + joint_abs action window + differentiable proxies.

Labels:
  0 weak/approach       -> neutral for binary loss, quality 0.30
  1 good_insert         -> binary good, quality 1.00
  2 pre_bounce_risk     -> binary bad, quality 0.05
  3 impact_or_recovery  -> binary bad, quality 0.00
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import zlib
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.evaluate_marker_proxy_scorer import insertion_t4, marker_proxy_features  # noqa: E402
from TFAC_V5.train_action_aware_marker_scorer import action_proxy_features  # noqa: E402


DATA_DIR = Path("/home/chenshuai/data/dataset/0414")
OUT_DIR = Path("/home/chenshuai/Project/output/insertion_risk_scorer")
WINDOW = 8
ACTION_KEY = "joint_abs"
ACTION_DIM = 7
REASON_NAMES = {
    0: "weak_approach",
    1: "good_insert",
    2: "pre_bounce_risk",
    3: "impact_or_recovery",
}


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def pad_window(seq: np.ndarray, window: int = WINDOW) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) >= window:
        return seq[-window:]
    pad = np.repeat(seq[:1], window - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)


def reason_from_annotation(ep_type: str, raw_label: int, frame_idx: int, lifts, pre_window: int) -> int:
    t4 = insertion_t4(ep_type, raw_label, frame_idx, lifts, pre_window=pre_window)
    if ep_type != "bounce" and t4 in (2, 3):
        t4 = 1 if raw_label == 1 else 0
    if t4 == 0:
        return 0
    if t4 == 1:
        return 1
    if t4 == 2:
        return 2
    return 3


def build_samples(args):
    annotations = pickle.load(open(DATA_DIR / "annotations.pkl", "rb"))
    samples = []
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        path = DATA_DIR / f"{ep_name}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            marker = f["observations/tac/left/marker_offset"][:]
            action = f[f"actions/{args.action_key}"][:]
        labels = np.asarray(info["labels"])
        n = min(len(marker), len(action), len(labels))
        indices = np.arange(WINDOW - 1, n)
        if args.max_per_episode and len(indices) > args.max_per_episode:
            rng = np.random.default_rng(stable_seed(ep_name))
            indices = np.sort(rng.choice(indices, args.max_per_episode, replace=False))
        ep_type = info.get("type")
        lifts = info.get("lifts", [])
        for frame_idx in indices:
            raw = int(labels[frame_idx])
            reason = reason_from_annotation(ep_type, raw, int(frame_idx), lifts, args.pre_window)
            if reason == 1:
                binary = 1
                quality = 1.0
            elif reason == 0:
                binary = -1
                quality = 0.30
            elif reason == 2:
                binary = 0
                quality = 0.05
            else:
                binary = 0
                quality = 0.0
            start = max(0, int(frame_idx) - WINDOW + 1)
            marker_seq = pad_window(marker[start : frame_idx + 1], WINDOW)
            action_seq = pad_window(action[start : frame_idx + 1], WINDOW)
            samples.append(
                {
                    "sample_id": f"{ep_name}/{frame_idx}",
                    "episode": ep_name,
                    "ep_type": ep_type,
                    "raw_label": raw,
                    "reason": reason,
                    "binary": binary,
                    "quality": quality,
                    "marker": marker_seq,
                    "marker_proxy": marker_proxy_features(marker_seq),
                    "action": action_seq,
                    "action_proxy": action_proxy_features(action_seq),
                }
            )
    return samples


def build_or_load(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "insertion_risk_features.npz"
    meta_path = OUT_DIR / "insertion_risk_metadata.json"
    sample_path = OUT_DIR / "insertion_risk_samples.csv"
    if data_path.exists() and not args.force_rebuild:
        return np.load(data_path, allow_pickle=True), json.load(open(meta_path, encoding="utf-8"))

    samples = build_samples(args)
    marker = np.stack([s["marker"] for s in samples]).astype(np.float32)
    marker_proxy = np.stack([s["marker_proxy"] for s in samples]).astype(np.float32)
    action = np.stack([s["action"] for s in samples]).astype(np.float32)
    action_proxy = np.stack([s["action_proxy"] for s in samples]).astype(np.float32)
    reason = np.array([s["reason"] for s in samples], dtype=np.int64)
    binary = np.array([s["binary"] for s in samples], dtype=np.int64)
    quality = np.array([s["quality"] for s in samples], dtype=np.float32)
    groups = np.array([s["episode"] for s in samples])
    ep_type = np.array([s["ep_type"] for s in samples])
    sample_ids = np.array([s["sample_id"] for s in samples])
    np.savez_compressed(
        data_path,
        marker=marker,
        marker_proxy=marker_proxy,
        action=action,
        action_proxy=action_proxy,
        reason=reason,
        binary=binary,
        quality=quality,
        groups=groups,
        ep_type=ep_type,
        sample_ids=sample_ids,
    )
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "episode", "ep_type", "raw_label", "reason", "reason_name", "binary", "quality"],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "episode": s["episode"],
                    "ep_type": s["ep_type"],
                    "raw_label": s["raw_label"],
                    "reason": s["reason"],
                    "reason_name": REASON_NAMES[s["reason"]],
                    "binary": s["binary"],
                    "quality": s["quality"],
                }
            )
    meta = {
        "window": WINDOW,
        "action_key": args.action_key,
        "action_dim": ACTION_DIM,
        "pre_window": args.pre_window,
        "reason_names": REASON_NAMES,
        "binary_definition": "good_insert=1, pre_bounce_risk/impact=0, weak_approach=-1 neutral",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.load(data_path, allow_pickle=True), meta


def balanced_indices(reason, max_per_class, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(reason):
        idx = np.flatnonzero(reason == cls)
        if len(idx):
            keep.extend(rng.choice(idx, min(max_per_class, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


def standardize_field(train, all_arr, last_dim):
    mean = train.reshape(-1, last_dim).mean(axis=0).astype(np.float32)
    std = train.reshape(-1, last_dim).std(axis=0).astype(np.float32) + 1e-6
    shape = [1] * all_arr.ndim
    shape[-1] = last_dim
    return ((all_arr - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32), mean, std


def standardize_tabular(train, all_arr):
    scaler = StandardScaler().fit(train)
    return scaler.transform(all_arr).astype(np.float32), scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32)


class InsertionRiskScorer(nn.Module):
    def __init__(self, marker_proxy_dim: int, action_proxy_dim: int, action_dim: int = ACTION_DIM, hidden: int = 160, dropout: float = 0.12):
        super().__init__()
        self.action_dim = action_dim
        self.marker_encoder = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(action_dim, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        in_dim = 48 * 2 * 3 * 3 + 48 * 4 + marker_proxy_dim + action_proxy_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
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
        self.reason_head = nn.Linear(hidden // 2, 4)
        self.quality_head = nn.Linear(hidden // 2, 1)

    def forward(self, marker, marker_proxy, action, action_proxy):
        marker_feat = self.marker_encoder(marker.permute(0, 4, 1, 2, 3))
        action_feat = self.action_encoder(action.permute(0, 2, 1))
        h = self.encoder(torch.cat([marker_feat, action_feat, marker_proxy, action_proxy], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "reason_logits": self.reason_head(h),
            "quality": self.quality_head(h).squeeze(-1),
        }


def make_loader(marker, mp, action, ap, binary, reason, quality, idx, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(marker[idx]).float(),
        torch.from_numpy(mp[idx]).float(),
        torch.from_numpy(action[idx]).float(),
        torch.from_numpy(ap[idx]).float(),
        torch.from_numpy(binary[idx]).long(),
        torch.from_numpy(reason[idx]).long(),
        torch.from_numpy(quality[idx]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, opt, device, weights, reason_w, binary_w):
    model.train()
    total = 0.0
    n = 0
    for marker, mp, action, ap, y_bin, y_reason, quality in loader:
        marker = marker.to(device)
        mp = mp.to(device)
        action = action.to(device)
        ap = ap.to(device)
        y_bin = y_bin.to(device)
        y_reason = y_reason.to(device)
        quality = quality.to(device)
        out = model(marker, mp, action, ap)
        mask = y_bin >= 0
        loss_bin = F.cross_entropy(out["binary_logits"][mask], y_bin[mask], weight=binary_w.to(device))
        loss_reason = F.cross_entropy(out["reason_logits"], y_reason, weight=reason_w.to(device))
        loss_quality = F.smooth_l1_loss(torch.sigmoid(out["quality"]), quality)
        loss = weights["binary"] * loss_bin + weights["reason"] * loss_reason + weights["quality"] * loss_quality
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * len(marker)
        n += len(marker)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    yb, yr, q, pb, pr, pq = [], [], [], [], [], []
    for marker, mp, action, ap, y_bin, y_reason, quality in loader:
        out = model(marker.to(device), mp.to(device), action.to(device), ap.to(device))
        yb.append(y_bin.numpy())
        yr.append(y_reason.numpy())
        q.append(quality.numpy())
        pb.append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
        pr.append(torch.softmax(out["reason_logits"], dim=-1).cpu().numpy())
        pq.append(torch.sigmoid(out["quality"]).cpu().numpy())
    return {
        "binary": np.concatenate(yb),
        "reason": np.concatenate(yr),
        "quality_true": np.concatenate(q),
        "p_good": np.concatenate(pb),
        "reason_prob": np.concatenate(pr),
        "quality_pred": np.concatenate(pq),
    }


def evaluate_pred(pred):
    yb = pred["binary"]
    yr = pred["reason"]
    q = pred["quality_true"]
    pb = pred["p_good"]
    pr = pred["reason_prob"]
    pq = pred["quality_pred"]
    valid = yb >= 0
    binary_pred = (pb[valid] >= 0.5).astype(np.int64)
    reason_pred = pr.argmax(axis=1)
    return {
        "binary_balanced_accuracy": float(balanced_accuracy_score(yb[valid], binary_pred)),
        "binary_macro_f1": float(f1_score(yb[valid], binary_pred, average="macro")),
        "binary_auc": float(roc_auc_score(yb[valid], pb[valid])),
        "reason_balanced_accuracy": float(balanced_accuracy_score(yr, reason_pred)),
        "reason_macro_f1": float(f1_score(yr, reason_pred, average="macro")),
        "quality_corr": float(np.corrcoef(pq, q)[0, 1]) if np.std(pq) > 1e-8 and np.std(q) > 1e-8 else 0.0,
        "quality_r2": float(r2_score(q, pq)),
        "binary_confusion": confusion_matrix(yb[valid], binary_pred, labels=[0, 1]).tolist(),
        "reason_confusion": confusion_matrix(yr, reason_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def metric(row):
    return row["binary_auc"] + 0.15 * row["reason_macro_f1"] + 0.15 * row["quality_corr"]


def aggregate(rows):
    keys = ["binary_balanced_accuracy", "binary_macro_f1", "binary_auc", "reason_balanced_accuracy", "reason_macro_f1", "quality_corr", "quality_r2"]
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def prep_arrays(marker, mp, action, ap, train_idx):
    marker_s, marker_mean, marker_std = standardize_field(marker[train_idx], marker, 2)
    action_s, action_mean, action_std = standardize_field(action[train_idx], action, ACTION_DIM)
    mp_s, mp_mean, mp_scale = standardize_tabular(mp[train_idx], mp)
    ap_s, ap_mean, ap_scale = standardize_tabular(ap[train_idx], ap)
    stats = {
        "marker_mean": marker_mean,
        "marker_std": marker_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "marker_proxy_mean": mp_mean,
        "marker_proxy_scale": mp_scale,
        "action_proxy_mean": ap_mean,
        "action_proxy_scale": ap_scale,
    }
    return marker_s, mp_s, action_s, ap_s, stats


def run_split(marker, mp, action, ap, binary, reason, quality, tr, te, device, args):
    marker_s, mp_s, action_s, ap_s, _ = prep_arrays(marker, mp, action, ap, tr)
    train_loader = make_loader(marker_s, mp_s, action_s, ap_s, binary, reason, quality, tr, args.batch_size, True)
    test_loader = make_loader(marker_s, mp_s, action_s, ap_s, binary, reason, quality, te, args.batch_size, False)
    reason_w = class_weights(reason[tr], 4)
    valid = binary[tr] >= 0
    binary_w = class_weights(binary[tr][valid], 2)
    model = InsertionRiskScorer(mp.shape[1], ap.shape[1], action_dim=ACTION_DIM, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    weights = {"binary": args.binary_weight, "reason": args.reason_weight, "quality": args.quality_weight}
    best = None
    best_metric = -math.inf
    bad = 0
    for _ in range(args.epochs):
        train_epoch(model, train_loader, opt, device, weights, reason_w, binary_w)
        sched.step()
        row = evaluate_pred(predict(model, test_loader, device))
        m = metric(row)
        if m > best_metric:
            best_metric = m
            best = row
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best


def train_final(marker, mp, action, ap, binary, reason, quality, device, args):
    all_idx = np.arange(len(marker))
    marker_s, mp_s, action_s, ap_s, stats = prep_arrays(marker, mp, action, ap, all_idx)
    loader = make_loader(marker_s, mp_s, action_s, ap_s, binary, reason, quality, all_idx, args.batch_size, True)
    reason_w = class_weights(reason, 4)
    valid = binary >= 0
    binary_w = class_weights(binary[valid], 2)
    model = InsertionRiskScorer(mp.shape[1], ap.shape[1], action_dim=ACTION_DIM, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {"binary": args.binary_weight, "reason": args.reason_weight, "quality": args.quality_weight}
    for _ in range(args.final_epochs):
        train_epoch(model, loader, opt, device, weights, reason_w, binary_w)
    eval_loader = make_loader(marker_s, mp_s, action_s, ap_s, binary, reason, quality, all_idx, args.batch_size, False)
    return model, stats, evaluate_pred(predict(model, eval_loader, device))


def gradient_sanity(model, stats, marker, mp, action, ap, device):
    model.eval()
    idx = np.arange(min(16, len(marker)))
    marker_s = (marker[idx] - stats["marker_mean"].reshape(1, 1, 1, 1, 2)) / stats["marker_std"].reshape(1, 1, 1, 1, 2)
    action_s = (action[idx] - stats["action_mean"].reshape(1, 1, ACTION_DIM)) / stats["action_std"].reshape(1, 1, ACTION_DIM)
    mp_s = (mp[idx] - stats["marker_proxy_mean"].reshape(1, -1)) / stats["marker_proxy_scale"].reshape(1, -1)
    ap_s = (ap[idx] - stats["action_proxy_mean"].reshape(1, -1)) / stats["action_proxy_scale"].reshape(1, -1)
    mt = torch.tensor(marker_s, dtype=torch.float32, device=device, requires_grad=True)
    at = torch.tensor(action_s, dtype=torch.float32, device=device, requires_grad=True)
    mpt = torch.tensor(mp_s, dtype=torch.float32, device=device, requires_grad=True)
    apt = torch.tensor(ap_s, dtype=torch.float32, device=device, requires_grad=True)
    out = model(mt, mpt, at, apt)
    p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1]
    risk = torch.softmax(out["reason_logits"], dim=-1)[:, 2:].sum(dim=1)
    score = (p_good - risk + torch.sigmoid(out["quality"])).mean()
    grads = torch.autograd.grad(score, [mt, at, mpt, apt])
    return {
        "grad_marker_norm": float(grads[0].norm().detach().cpu()),
        "grad_action_norm": float(grads[1].norm().detach().cpu()),
        "grad_marker_proxy_norm": float(grads[2].norm().detach().cpu()),
        "grad_action_proxy_norm": float(grads[3].norm().detach().cpu()),
        "usable_for_guidance": bool(all(torch.isfinite(g).all().item() and g.norm().item() > 1e-8 for g in grads)),
    }


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, meta = build_or_load(args)
    marker = data["marker"].astype(np.float32)
    mp = data["marker_proxy"].astype(np.float32)
    action = data["action"].astype(np.float32)
    ap = data["action_proxy"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
    groups = data["groups"]

    keep = balanced_indices(reason, args.max_per_class, args.seed)
    marker, mp, action, ap = marker[keep], mp[keep], action[keep], ap[keep]
    reason, binary, quality, groups = reason[keep], binary[keep], quality[keep], groups[keep]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    rows = []
    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(cv.split(marker, reason, groups)):
        print(f"fold {fold}", flush=True)
        row = run_split(marker, mp, action, ap, binary, reason, quality, tr, te, device, args)
        row["fold"] = fold
        rows.append(row)
        print({k: row[k] for k in ["binary_auc", "reason_macro_f1", "quality_corr"]}, flush=True)

    final_model, stats, train_metrics = train_final(marker, mp, action, ap, binary, reason, quality, device, args)
    sanity = gradient_sanity(final_model, stats, marker, mp, action, ap, device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "data": {
            "n": int(len(marker)),
            "reason_counts": {str(k): int(v) for k, v in Counter(reason.tolist()).items()},
            "binary_counts_with_neutral": {str(k): int(v) for k, v in Counter(binary.tolist()).items()},
        },
        "meta": meta,
        "args": vars(args),
        "mixed_group_cv": aggregate(rows),
        "folds": rows,
        "final_train_capacity_check": train_metrics,
        "gradient_sanity": sanity,
    }
    (OUT_DIR / "insertion_risk_scorer_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    ckpt = {
        "model_state_dict": final_model.state_dict(),
        "marker_proxy_dim": int(mp.shape[1]),
        "action_proxy_dim": int(ap.shape[1]),
        "action_key": args.action_key,
        "action_dim": ACTION_DIM,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "reason_names": REASON_NAMES,
        **{k: v.astype(np.float32) for k, v in stats.items()},
    }
    torch.save(ckpt, OUT_DIR / "insertion_risk_scorer_final.pt")
    print(json.dumps({"mixed_group_cv": result["mixed_group_cv"], "gradient_sanity": sanity}, ensure_ascii=False, indent=2))
    print(f"Saved {OUT_DIR / 'insertion_risk_scorer_eval.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--final_epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--binary_weight", type=float, default=1.0)
    parser.add_argument("--reason_weight", type=float, default=0.55)
    parser.add_argument("--quality_weight", type=float, default=0.55)
    parser.add_argument("--max_per_class", type=int, default=1400)
    parser.add_argument("--max_per_episode", type=int, default=260)
    parser.add_argument("--pre_window", type=int, default=14)
    parser.add_argument("--action_key", default="joint_abs")
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
