"""Train an action-aware tactile quality scorer for DP guidance.

Why another scorer?
The marker-only scorer can tell whether a tactile outcome looks good, but DP
guidance needs a score whose gradient can flow to the action.  The useful
structure is closer to TouchGuide/CPM:

    current/future tactile + candidate action + task -> quality score

This script trains that deployable form offline with available labels:
  - marker window field: tactile consequence;
  - marker proxy: physical contact intensity/area/smoothness;
  - action proxy: motion magnitude and smoothness;
  - task id: insertion vs board calibration;
  - three heads: binary good/bad, T4 reason, continuous quality score.

Important label detail:
  - insertion approach/weak contact is neutral for binary loss but kept for T4;
  - board too_light is bad, because weak force is a bad wipe action.
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.evaluate_marker_proxy_scorer import (  # noqa: E402
    T4_NAMES,
    assign_board_t4,
    board_windows,
    insertion_t4,
    marker_proxy_features,
)


INSERTION_DIR = Path("/home/chenshuai/data/dataset/0414")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/action_aware_marker_scorer")
WINDOW = 8
ACTION_DIM = 6
TASK_TO_ID = {"insertion": 0, "board": 1}


def pad_last(seq: np.ndarray, length: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) >= length:
        return seq[-length:]
    pad = np.repeat(seq[:1], length - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)


def action_proxy_features(action_seq: np.ndarray) -> np.ndarray:
    action_seq = np.asarray(action_seq, dtype=np.float32)
    if action_seq.shape[-1] > ACTION_DIM:
        action_seq = action_seq[:, :ACTION_DIM]
    elif action_seq.shape[-1] < ACTION_DIM:
        pad = np.zeros((len(action_seq), ACTION_DIM - action_seq.shape[-1]), dtype=np.float32)
        action_seq = np.concatenate([action_seq, pad], axis=-1)
    delta = np.diff(action_seq, axis=0) if len(action_seq) > 1 else np.zeros((1, ACTION_DIM), dtype=np.float32)
    speed = np.linalg.norm(delta, axis=-1)
    accel = np.diff(delta, axis=0) if len(delta) > 1 else np.zeros((1, ACTION_DIM), dtype=np.float32)
    accel_norm = np.linalg.norm(accel, axis=-1)
    return np.array(
        [
            np.linalg.norm(action_seq, axis=-1).mean(),
            np.linalg.norm(action_seq, axis=-1).std(),
            speed.mean(),
            speed.std(),
            np.percentile(speed, 90),
            accel_norm.mean(),
            np.percentile(accel_norm, 90),
            np.linalg.norm(action_seq[-1] - action_seq[0]),
            np.abs(delta).mean(),
            np.abs(delta).max(),
        ],
        dtype=np.float32,
    )


def build_insertion_samples(max_per_episode=260):
    annotations = pickle.load(open(INSERTION_DIR / "annotations.pkl", "rb"))
    samples = []
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        path = INSERTION_DIR / f"{ep_name}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            marker = f["observations/tac/left/marker_offset"][:]
            action = f["actions/eef_abs"][:]
        labels = np.asarray(info["labels"])
        n = min(len(marker), len(labels), len(action))
        indices = np.arange(WINDOW - 1, n)
        if max_per_episode and len(indices) > max_per_episode:
            rng = np.random.default_rng(abs(hash(ep_name)) % (2**32))
            indices = np.sort(rng.choice(indices, max_per_episode, replace=False))
        ep_type = info.get("type")
        lifts = info.get("lifts", [])
        for frame_idx in indices:
            raw = int(labels[frame_idx])
            t4 = insertion_t4(ep_type, raw, frame_idx, lifts)
            if ep_type != "bounce" and t4 in (2, 3):
                t4 = 1 if raw == 1 else 0
            if t4 == 1:
                binary = 1
                score = 1.0
            elif t4 in (2, 3):
                binary = 0
                score = 0.05 if t4 == 2 else 0.0
            else:
                binary = -1
                score = 0.25
            mseq = marker[max(0, frame_idx - WINDOW + 1) : frame_idx + 1]
            aseq = action[max(0, frame_idx - WINDOW + 1) : frame_idx + 1]
            samples.append(
                {
                    "sample_id": f"insertion/{ep_name}/{frame_idx}",
                    "task": "insertion",
                    "episode": ep_name,
                    "t4": t4,
                    "binary": binary,
                    "score": score,
                    "raw_label": raw,
                    "marker": pad_last(mseq, WINDOW),
                    "marker_proxy": marker_proxy_features(mseq),
                    "action": pad_last(aseq, WINDOW),
                    "action_proxy": action_proxy_features(aseq),
                }
            )
    return samples


def build_board_samples():
    rows = board_windows(window=32, stride=16)
    labels, thresholds = assign_board_t4(rows)
    samples = []
    for row, (t4, raw) in zip(rows, labels):
        with h5py.File(row["path"], "r") as f:
            action = f["actions/eef_abs"][row["start"] : row["end"]]
        marker_seq = np.asarray(row["marker_seq"], dtype=np.float32)
        binary = 1 if t4 == 1 else 0
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
                "marker": pad_last(marker_seq, WINDOW),
                "marker_proxy": marker_proxy_features(marker_seq),
                "action": pad_last(action, WINDOW),
                "action_proxy": action_proxy_features(action),
            }
        )
    return samples, thresholds


def build_or_load(force_rebuild=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "action_aware_marker_features.npz"
    meta_path = OUT_DIR / "action_aware_marker_metadata.json"
    sample_path = OUT_DIR / "action_aware_marker_samples.csv"
    if path.exists() and not force_rebuild:
        return np.load(path, allow_pickle=True), json.load(open(meta_path, encoding="utf-8"))

    insertion = build_insertion_samples()
    board, thresholds = build_board_samples()
    samples = insertion + board
    marker = np.stack([s["marker"] for s in samples]).astype(np.float32)
    marker_proxy = np.stack([s["marker_proxy"] for s in samples]).astype(np.float32)
    action = np.stack([s["action"] for s in samples]).astype(np.float32)
    action_proxy = np.stack([s["action_proxy"] for s in samples]).astype(np.float32)
    y_t4 = np.array([s["t4"] for s in samples], dtype=np.int64)
    y_bin = np.array([s["binary"] for s in samples], dtype=np.int64)
    score = np.array([s["score"] for s in samples], dtype=np.float32)
    task = np.array([s["task"] for s in samples])
    task_id = np.array([TASK_TO_ID[s["task"]] for s in samples], dtype=np.int64)
    groups = np.array([f"{s['task']}::{s['episode']}" for s in samples])
    sample_ids = np.array([s["sample_id"] for s in samples])
    np.savez_compressed(
        path,
        marker=marker,
        marker_proxy=marker_proxy,
        action=action,
        action_proxy=action_proxy,
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
            fieldnames=["sample_id", "task", "episode", "t4", "t4_name", "binary", "score", "raw_label"],
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
            "insertion": "good=success insert, bad=bounce risk/impact, approach neutral masked from binary",
            "board": "good=force suitable and smooth, bad=too_light/too_heavy/rough",
        },
        "action_proxy_names": [
            "action_norm_mean",
            "action_norm_std",
            "speed_mean",
            "speed_std",
            "speed_p90",
            "accel_mean",
            "accel_p90",
            "first_last_l2",
            "abs_delta_mean",
            "abs_delta_max",
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return np.load(path, allow_pickle=True), meta


def balanced_indices(y_t4, task, max_per_task_class=1400, seed=42):
    rng = np.random.default_rng(seed)
    keep = []
    for task_name in np.unique(task):
        for cls in np.unique(y_t4):
            idx = np.flatnonzero((task == task_name) & (y_t4 == cls))
            if len(idx):
                keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
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


class ActionAwareMarkerScorer(nn.Module):
    def __init__(self, marker_proxy_dim, action_proxy_dim, hidden=160, dropout=0.15):
        super().__init__()
        self.marker_encoder = nn.Sequential(
            nn.Conv3d(2, 24, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.Conv3d(24, 48, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 3, 3)),
            nn.Flatten(),
        )
        self.action_encoder = nn.Sequential(
            nn.Conv1d(ACTION_DIM, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        in_dim = 48 * 2 * 3 * 3 + 48 * 4 + marker_proxy_dim + action_proxy_dim + 2
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
        self.t4_head = nn.Linear(hidden // 2, 4)
        self.score_head = nn.Linear(hidden // 2, 1)

    def forward(self, marker, marker_proxy, action, action_proxy, task_id):
        marker_feat = self.marker_encoder(marker.permute(0, 4, 1, 2, 3))
        action_feat = self.action_encoder(action.permute(0, 2, 1))
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([marker_feat, action_feat, marker_proxy, action_proxy, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "t4_logits": self.t4_head(h),
            "score": self.score_head(h).squeeze(-1),
        }


def make_loader(marker, marker_proxy, action, action_proxy, task_id, y_bin, y_t4, score, idx, batch=256, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(marker[idx]).float(),
        torch.from_numpy(marker_proxy[idx]).float(),
        torch.from_numpy(action[idx]).float(),
        torch.from_numpy(action_proxy[idx]).float(),
        torch.from_numpy(task_id[idx]).long(),
        torch.from_numpy(y_bin[idx]).long(),
        torch.from_numpy(y_t4[idx]).long(),
        torch.from_numpy(score[idx]).float(),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0)


def train_epoch(model, loader, opt, device, weights):
    model.train()
    total = 0.0
    n = 0
    for marker, mp, action, ap, task_id, y_bin, y_t4, score in loader:
        marker = marker.to(device)
        mp = mp.to(device)
        action = action.to(device)
        ap = ap.to(device)
        task_id = task_id.to(device)
        y_bin = y_bin.to(device)
        y_t4 = y_t4.to(device)
        score = score.to(device)
        out = model(marker, mp, action, ap, task_id)
        mask = y_bin >= 0
        loss_bin = F.cross_entropy(out["binary_logits"][mask], y_bin[mask]) if mask.any() else torch.zeros((), device=device)
        loss_t4 = F.cross_entropy(out["t4_logits"], y_t4)
        loss_score = F.smooth_l1_loss(torch.sigmoid(out["score"]), score)
        loss = weights["binary"] * loss_bin + weights["t4"] * loss_t4 + weights["score"] * loss_score
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * len(marker)
        n += len(marker)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yb_all, yt_all, q_all, pb_all, pt_all, pq_all = [], [], [], [], [], []
    for marker, mp, action, ap, task_id, y_bin, y_t4, score in loader:
        out = model(marker.to(device), mp.to(device), action.to(device), ap.to(device), task_id.to(device))
        yb_all.append(y_bin.numpy())
        yt_all.append(y_t4.numpy())
        q_all.append(score.numpy())
        pb_all.append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
        pt_all.append(torch.softmax(out["t4_logits"], dim=-1).cpu().numpy())
        pq_all.append(torch.sigmoid(out["score"]).cpu().numpy())
    yb = np.concatenate(yb_all)
    yt = np.concatenate(yt_all)
    q = np.concatenate(q_all)
    pb = np.concatenate(pb_all)
    pt = np.concatenate(pt_all)
    pq = np.concatenate(pq_all)
    valid_b = yb >= 0
    pred_b = (pb[valid_b] >= 0.5).astype(np.int64)
    pred_t = pt.argmax(axis=1)
    row = {
        "binary_n": int(valid_b.sum()),
        "binary_balanced_accuracy": float(balanced_accuracy_score(yb[valid_b], pred_b)) if valid_b.any() else None,
        "binary_macro_f1": float(f1_score(yb[valid_b], pred_b, average="macro")) if valid_b.any() else None,
        "binary_auc": float(roc_auc_score(yb[valid_b], pb[valid_b])) if len(np.unique(yb[valid_b])) == 2 else None,
        "t4_balanced_accuracy": float(balanced_accuracy_score(yt, pred_t)),
        "t4_macro_f1": float(f1_score(yt, pred_t, average="macro")),
        "score_corr": float(np.corrcoef(pq, q)[0, 1]) if np.std(pq) > 1e-8 and np.std(q) > 1e-8 else 0.0,
        "binary_confusion": confusion_matrix(yb[valid_b], pred_b, labels=[0, 1]).tolist() if valid_b.any() else None,
        "t4_confusion": confusion_matrix(yt, pred_t, labels=[0, 1, 2, 3]).tolist(),
    }
    return row


def metric(row):
    return (row.get("binary_auc") or 0.0) + 0.25 * row.get("score_corr", 0.0) + 0.12 * row.get("t4_macro_f1", 0.0)


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


def run_split(marker, mp, action, ap, task_id, y_bin, y_t4, score, tr, te, device, epochs):
    marker_s, mp_s, action_s, ap_s, _ = prep_arrays(marker, mp, action, ap, tr)
    train_loader = make_loader(marker_s, mp_s, action_s, ap_s, task_id, y_bin, y_t4, score, tr, shuffle=True)
    test_loader = make_loader(marker_s, mp_s, action_s, ap_s, task_id, y_bin, y_t4, score, te, shuffle=False)
    model = ActionAwareMarkerScorer(mp.shape[1], ap.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    weights = {"binary": 1.0, "t4": 0.45, "score": 0.55}
    best = None
    best_metric = -math.inf
    patience = 12
    bad = 0
    for _ in range(epochs):
        train_epoch(model, train_loader, opt, device, weights)
        sched.step()
        row = evaluate(model, test_loader, device)
        m = metric(row)
        if m > best_metric:
            best_metric = m
            best = row
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best


def aggregate(rows):
    keys = ["binary_balanced_accuracy", "binary_macro_f1", "binary_auc", "t4_balanced_accuracy", "t4_macro_f1", "score_corr"]
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))} if vals else None
    return out


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, meta = build_or_load(force_rebuild=args.force_rebuild)
    marker = data["marker"].astype(np.float32)
    mp = data["marker_proxy"].astype(np.float32)
    action = data["action"].astype(np.float32)
    ap = data["action_proxy"].astype(np.float32)
    y_t4 = data["y_t4"].astype(np.int64)
    y_bin = data["y_binary"].astype(np.int64)
    score = data["score"].astype(np.float32)
    task = data["task"]
    task_id = data["task_id"].astype(np.int64)
    groups = data["groups"]

    keep = balanced_indices(y_t4, task, args.max_per_task_class, args.seed)
    marker, mp, action, ap = marker[keep], mp[keep], action[keep], ap[keep]
    y_t4, y_bin, score, task, task_id, groups = y_t4[keep], y_bin[keep], score[keep], task[keep], task_id[keep], groups[keep]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results = {
        "data": {
            "n": int(len(marker)),
            "marker_shape": list(marker.shape[1:]),
            "action_shape": list(action.shape[1:]),
            "marker_proxy_dim": int(mp.shape[1]),
            "action_proxy_dim": int(ap.shape[1]),
            "task_counts": {k: int(v) for k, v in Counter(task.tolist()).items()},
            "binary_counts_with_neutral": {str(k): int(v) for k, v in Counter(y_bin.tolist()).items()},
            "t4_counts": {str(k): int(v) for k, v in Counter(y_t4.tolist()).items()},
        },
        "meta": meta,
        "mixed_group_cv": {},
        "cross_task": {},
    }
    rows = []
    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(cv.split(marker, y_t4, groups)):
        print(f"fold {fold}", flush=True)
        row = run_split(marker, mp, action, ap, task_id, y_bin, y_t4, score, tr, te, device, args.epochs)
        row["fold"] = fold
        rows.append(row)
        print(row, flush=True)
    results["mixed_group_cv"] = aggregate(rows)
    results["mixed_group_cv"]["folds"] = rows

    for train_task, test_task in [("insertion", "board"), ("board", "insertion")]:
        tr = np.flatnonzero(task == train_task)
        te = np.flatnonzero(task == test_task)
        row = run_split(marker, mp, action, ap, task_id, y_bin, y_t4, score, tr, te, device, args.epochs)
        results["cross_task"][f"{train_task}_to_{test_task}"] = row
        print(f"cross {train_task}->{test_task}: {row}", flush=True)

    all_idx = np.arange(len(marker))
    marker_s, mp_s, action_s, ap_s, stats = prep_arrays(marker, mp, action, ap, all_idx)
    final = ActionAwareMarkerScorer(mp.shape[1], ap.shape[1]).to(device)
    opt = torch.optim.AdamW(final.parameters(), lr=2e-3, weight_decay=1e-4)
    weights = {"binary": 1.0, "t4": 0.45, "score": 0.55}
    loader = make_loader(marker_s, mp_s, action_s, ap_s, task_id, y_bin, y_t4, score, all_idx, shuffle=True)
    for _ in range(args.final_epochs):
        train_epoch(final, loader, opt, device, weights)
    train_metrics = evaluate(final, make_loader(marker_s, mp_s, action_s, ap_s, task_id, y_bin, y_t4, score, all_idx, shuffle=False), device)
    results["final_train_metrics_capacity_check"] = train_metrics

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "action_aware_marker_scorer_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    ckpt = {
        "model_state_dict": final.state_dict(),
        "marker_proxy_dim": int(mp.shape[1]),
        "action_proxy_dim": int(ap.shape[1]),
        "task_to_id": TASK_TO_ID,
        "window": WINDOW,
        "metrics_train": train_metrics,
        "heads": ["binary_logits", "t4_logits", "score"],
    }
    ckpt.update({k: v for k, v in stats.items()})
    torch.save(ckpt, OUT_DIR / "action_aware_marker_scorer_final.pt")
    with open(OUT_DIR / "action_aware_marker_scorer_final_summary.json", "w", encoding="utf-8") as f:
        json.dump({"train_metrics": train_metrics, "data": results["data"]}, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUT_DIR / 'action_aware_marker_scorer_eval.json'}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--final_epochs", type=int, default=60)
    parser.add_argument("--max_per_task_class", type=int, default=1400)
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
