"""Train a differentiable proxy scorer for PTG guidance.

This is the deployable counterpart of the strong board RF/GBM teachers.
It uses only marker/action proxy features, not force, and trains a small
PyTorch multi-head scorer:

  binary good/bad      -> safety gate / classifier guidance
  5-way reason class   -> interpretable failure mode
  continuous quality   -> ranking / gradient guidance score

Unified reason taxonomy:
  0 weak_or_no_contact / too_light
  1 good_stable_smooth
  2 excessive_or_risk / too_heavy
  3 impact_or_rough_force
  4 rough_motion

Insertion labels come from existing annotations.  Board labels come from the
best weak-label scheme found in eval_board_quality_label_schemes.py:
left_force + t5_scoreband.
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

from TFAC_V5.eval_board_quality_label_schemes import (  # noqa: E402
    assign_labels as assign_board_labels,
    load_or_build_windows,
    marker_proxy_features,
    action_proxy_features,
)
from TFAC_V5.evaluate_marker_proxy_scorer import insertion_t4  # noqa: E402


INSERTION_DIR = Path("/home/chenshuai/data/dataset/0414")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2")

TASK_TO_ID = {"insertion": 0, "board": 1}
REASON_NAMES = {
    0: "weak_or_no_contact_too_light",
    1: "good_stable_smooth",
    2: "excessive_or_risk_too_heavy",
    3: "impact_or_rough_force",
    4: "rough_motion",
}
FEATURE_BLOCKS = ["left_marker", "right_marker", "lr_absdiff", "eef_action", "joint_action"]


def deterministic_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def pad_window(seq: np.ndarray, window: int) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) >= window:
        return seq[-window:]
    pad = np.repeat(seq[:1], window - len(seq), axis=0)
    return np.concatenate([pad, seq], axis=0)


def feature_vector(left_seq: np.ndarray, right_seq: np.ndarray, eef_seq: np.ndarray, joint_seq: np.ndarray) -> np.ndarray:
    left = marker_proxy_features(left_seq)
    right = marker_proxy_features(right_seq)
    diff = np.abs(left - right)
    eef = action_proxy_features(eef_seq)
    joint = action_proxy_features(joint_seq)
    return np.concatenate([left, right, diff, eef, joint]).astype(np.float32)


def insertion_reason(ep_type: str, raw_label: int, frame_idx: int, lifts) -> int:
    t4 = insertion_t4(ep_type, raw_label, frame_idx, lifts)
    if ep_type != "bounce" and t4 in (2, 3):
        t4 = 1 if raw_label == 1 else 0
    if t4 == 0:
        return 0
    if t4 == 1:
        return 1
    if t4 == 2:
        return 2
    return 3


def build_insertion_samples(window: int, max_per_episode: int) -> list[dict]:
    annotations = pickle.load(open(INSERTION_DIR / "annotations.pkl", "rb"))
    rows = []
    for ep_name, info in sorted(annotations.items()):
        if not isinstance(info, dict) or info.get("type") not in {"success", "bounce"}:
            continue
        path = INSERTION_DIR / f"{ep_name}.hdf5"
        if not path.exists():
            continue
        with h5py.File(path, "r") as f:
            left = f["observations/tac/left/marker_offset"][:]
            right = f["observations/tac/right/marker_offset"][:] if "observations/tac/right/marker_offset" in f else left
            eef = f["actions/eef_abs"][:]
            joint = f["actions/joint_abs"][:]
        labels = np.asarray(info["labels"])
        n = min(len(left), len(right), len(eef), len(joint), len(labels))
        indices = np.arange(window - 1, n)
        if max_per_episode and len(indices) > max_per_episode:
            rng = np.random.default_rng(deterministic_seed(ep_name))
            indices = np.sort(rng.choice(indices, max_per_episode, replace=False))
        ep_type = info.get("type")
        lifts = info.get("lifts", [])
        for frame_idx in indices:
            raw = int(labels[frame_idx])
            reason = insertion_reason(ep_type, raw, int(frame_idx), lifts)
            if reason == 1:
                binary = 1
                quality = 1.0
            elif reason == 0:
                binary = -1
                quality = 0.25
            elif reason == 2:
                binary = 0
                quality = 0.05
            else:
                binary = 0
                quality = 0.0
            start = max(0, int(frame_idx) - window + 1)
            rows.append(
                {
                    "sample_id": f"insertion/{ep_name}/{frame_idx}",
                    "task": "insertion",
                    "episode": ep_name,
                    "reason": reason,
                    "binary": binary,
                    "quality": quality,
                    "raw_label": raw,
                    "feature": feature_vector(
                        pad_window(left[start : frame_idx + 1], window),
                        pad_window(right[start : frame_idx + 1], window),
                        pad_window(eef[start : frame_idx + 1], window),
                        pad_window(joint[start : frame_idx + 1], window),
                    ),
                }
            )
    return rows


def build_board_samples(args) -> tuple[list[dict], dict]:
    board_args = argparse.Namespace(window=args.board_window, stride=args.board_stride, force_rebuild=args.force_rebuild)
    data, meta = load_or_build_windows(board_args)
    y, quality, thresholds = assign_board_labels(data, args.board_force_source, args.board_scheme)
    X = np.concatenate(
        [
            data["X_left"].astype(np.float32),
            data["X_right"].astype(np.float32),
            data["X_lr_absdiff"].astype(np.float32),
            data["X_eef"].astype(np.float32),
            data["X_joint"].astype(np.float32),
        ],
        axis=1,
    )
    rows = []
    for i in range(len(y)):
        reason = int(y[i])
        rows.append(
            {
                "sample_id": f"board/{str(data['sample_id'][i])}",
                "task": "board",
                "episode": str(data["episode"][i]),
                "reason": reason,
                "binary": 1 if reason == 1 else 0,
                "quality": float(quality[i]),
                "raw_label": REASON_NAMES[reason],
                "feature": X[i].astype(np.float32),
            }
        )
    meta = {**meta, "board_force_source": args.board_force_source, "board_scheme": args.board_scheme, "thresholds": thresholds}
    return rows, meta


def build_or_load_dataset(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = OUT_DIR / "ptg_proxy_scorer_v2_features.npz"
    meta_path = OUT_DIR / "ptg_proxy_scorer_v2_metadata.json"
    sample_path = OUT_DIR / "ptg_proxy_scorer_v2_samples.csv"
    if data_path.exists() and not args.force_rebuild:
        return np.load(data_path, allow_pickle=True), json.load(open(meta_path, encoding="utf-8"))

    insertion = build_insertion_samples(args.insertion_window, args.max_insertion_per_episode)
    board, board_meta = build_board_samples(args)
    samples = insertion + board
    X = np.stack([s["feature"] for s in samples]).astype(np.float32)
    reason = np.array([s["reason"] for s in samples], dtype=np.int64)
    binary = np.array([s["binary"] for s in samples], dtype=np.int64)
    quality = np.array([s["quality"] for s in samples], dtype=np.float32)
    task = np.array([s["task"] for s in samples])
    task_id = np.array([TASK_TO_ID[s["task"]] for s in samples], dtype=np.int64)
    groups = np.array([f"{s['task']}::{s['episode']}" for s in samples])
    sample_ids = np.array([s["sample_id"] for s in samples])
    np.savez_compressed(
        data_path,
        X=X,
        reason=reason,
        binary=binary,
        quality=quality,
        task=task,
        task_id=task_id,
        groups=groups,
        sample_ids=sample_ids,
    )
    with open(sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "task", "episode", "reason", "reason_name", "binary", "quality", "raw_label"],
        )
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "sample_id": s["sample_id"],
                    "task": s["task"],
                    "episode": s["episode"],
                    "reason": s["reason"],
                    "reason_name": REASON_NAMES[s["reason"]],
                    "binary": s["binary"],
                    "quality": s["quality"],
                    "raw_label": s["raw_label"],
                }
            )
    meta = {
        "feature_blocks": FEATURE_BLOCKS,
        "feature_dim": int(X.shape[1]),
        "task_to_id": TASK_TO_ID,
        "reason_names": REASON_NAMES,
        "insertion_window": args.insertion_window,
        "board_window": args.board_window,
        "board_stride": args.board_stride,
        "board_meta": board_meta,
        "binary_definition": {
            "insertion": "good=success insertion; bad=bounce risk/impact; weak/no-contact masked from binary loss",
            "board": "good=force-band and smooth; bad=too_light/too_heavy/rough_force/rough_motion",
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return np.load(data_path, allow_pickle=True), meta


def balanced_indices(reason: np.ndarray, task: np.ndarray, max_per_task_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for task_name in np.unique(task):
        for cls in np.unique(reason):
            idx = np.flatnonzero((task == task_name) & (reason == cls))
            if len(idx):
                keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


class PTGProxyScorerV2(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.reason_head = nn.Linear(hidden // 2, 5)
        self.quality_head = nn.Linear(hidden // 2, 1)

    def forward(self, x, task_id):
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([x, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "reason_logits": self.reason_head(h),
            "quality": self.quality_head(h).squeeze(-1),
        }


def make_loader(X, task_id, binary, reason, quality, idx, batch_size=256, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(X[idx]).float(),
        torch.from_numpy(task_id[idx]).long(),
        torch.from_numpy(binary[idx]).long(),
        torch.from_numpy(reason[idx]).long(),
        torch.from_numpy(quality[idx]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, opt, device, weights, reason_w, binary_w):
    model.train()
    total = 0.0
    n = 0
    for x, task_id, y_bin, y_reason, q in loader:
        x = x.to(device)
        task_id = task_id.to(device)
        y_bin = y_bin.to(device)
        y_reason = y_reason.to(device)
        q = q.to(device)
        out = model(x, task_id)
        mask = y_bin >= 0
        loss_bin = (
            F.cross_entropy(out["binary_logits"][mask], y_bin[mask], weight=binary_w.to(device))
            if mask.any()
            else torch.zeros((), device=device)
        )
        loss_reason = F.cross_entropy(out["reason_logits"], y_reason, weight=reason_w.to(device))
        loss_quality = F.smooth_l1_loss(torch.sigmoid(out["quality"]), q)
        loss = weights["binary"] * loss_bin + weights["reason"] * loss_reason + weights["quality"] * loss_quality
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * len(x)
        n += len(x)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    yb_all, yr_all, q_all, task_all, pb_all, pr_all, pq_all = [], [], [], [], [], [], []
    for x, task_id, y_bin, y_reason, q in loader:
        out = model(x.to(device), task_id.to(device))
        yb_all.append(y_bin.numpy())
        yr_all.append(y_reason.numpy())
        q_all.append(q.numpy())
        task_all.append(task_id.numpy())
        pb_all.append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
        pr_all.append(torch.softmax(out["reason_logits"], dim=-1).cpu().numpy())
        pq_all.append(torch.sigmoid(out["quality"]).cpu().numpy())
    return {
        "binary": np.concatenate(yb_all),
        "reason": np.concatenate(yr_all),
        "quality_true": np.concatenate(q_all),
        "task_id": np.concatenate(task_all),
        "p_good": np.concatenate(pb_all),
        "reason_prob": np.concatenate(pr_all),
        "quality_pred": np.concatenate(pq_all),
    }


def eval_from_pred(pred: dict, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(pred["reason"]), dtype=bool)
    yb = pred["binary"][mask]
    yr = pred["reason"][mask]
    q = pred["quality_true"][mask]
    pb = pred["p_good"][mask]
    pr = pred["reason_prob"][mask]
    pq = pred["quality_pred"][mask]
    valid = yb >= 0
    out = {}
    if valid.any() and len(np.unique(yb[valid])) == 2:
        out["binary_balanced_accuracy"] = float(balanced_accuracy_score(yb[valid], (pb[valid] >= 0.5).astype(np.int64)))
        out["binary_macro_f1"] = float(f1_score(yb[valid], (pb[valid] >= 0.5).astype(np.int64), average="macro"))
        out["binary_auc"] = float(roc_auc_score(yb[valid], pb[valid]))
        out["binary_confusion"] = confusion_matrix(yb[valid], (pb[valid] >= 0.5).astype(np.int64), labels=[0, 1]).tolist()
    else:
        out["binary_balanced_accuracy"] = None
        out["binary_macro_f1"] = None
        out["binary_auc"] = None
        out["binary_confusion"] = None
    reason_pred = pr.argmax(axis=1)
    out["reason_balanced_accuracy"] = float(balanced_accuracy_score(yr, reason_pred))
    out["reason_macro_f1"] = float(f1_score(yr, reason_pred, average="macro"))
    out["reason_confusion"] = confusion_matrix(yr, reason_pred, labels=[0, 1, 2, 3, 4]).tolist()
    out["quality_corr"] = float(np.corrcoef(pq, q)[0, 1]) if np.std(pq) > 1e-8 and np.std(q) > 1e-8 else 0.0
    out["quality_r2"] = float(r2_score(q, pq))
    return out


def metric(row: dict) -> float:
    return (
        (row.get("binary_auc") or 0.0)
        + 0.20 * row.get("quality_corr", 0.0)
        + 0.12 * row.get("reason_macro_f1", 0.0)
    )


def aggregate(rows: list[dict]) -> dict:
    keys = [
        "binary_balanced_accuracy",
        "binary_macro_f1",
        "binary_auc",
        "reason_balanced_accuracy",
        "reason_macro_f1",
        "quality_corr",
        "quality_r2",
    ]
    out = {}
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def standardize(train_idx, X):
    scaler = StandardScaler().fit(X[train_idx])
    return scaler.transform(X).astype(np.float32), scaler


def run_split(X, task_id, binary, reason, quality, tr, te, device, args):
    Xs, _ = standardize(tr, X)
    reason_w = class_weights(reason[tr], 5)
    valid_bin = binary[tr] >= 0
    binary_w = class_weights(binary[tr][valid_bin], 2)
    model = PTGProxyScorerV2(X.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    train_loader = make_loader(Xs, task_id, binary, reason, quality, tr, args.batch_size, shuffle=True)
    test_loader = make_loader(Xs, task_id, binary, reason, quality, te, args.batch_size, shuffle=False)
    weights = {"binary": args.binary_weight, "reason": args.reason_weight, "quality": args.quality_weight}
    best = None
    best_metric = -math.inf
    bad = 0
    for _ in range(args.epochs):
        train_epoch(model, train_loader, opt, device, weights, reason_w, binary_w)
        sched.step()
        pred = predict(model, test_loader, device)
        row = eval_from_pred(pred)
        m = metric(row)
        if m > best_metric:
            best_metric = m
            best = row
            best["per_task"] = {
                "insertion": eval_from_pred(pred, pred["task_id"] == TASK_TO_ID["insertion"]),
                "board": eval_from_pred(pred, pred["task_id"] == TASK_TO_ID["board"]),
            }
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    return best


def train_final(X, task_id, binary, reason, quality, device, args):
    all_idx = np.arange(len(X))
    Xs, scaler = standardize(all_idx, X)
    reason_w = class_weights(reason, 5)
    valid_bin = binary >= 0
    binary_w = class_weights(binary[valid_bin], 2)
    model = PTGProxyScorerV2(X.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(Xs, task_id, binary, reason, quality, all_idx, args.batch_size, shuffle=True)
    weights = {"binary": args.binary_weight, "reason": args.reason_weight, "quality": args.quality_weight}
    for _ in range(args.final_epochs):
        train_epoch(model, loader, opt, device, weights, reason_w, binary_w)
    pred = predict(model, make_loader(Xs, task_id, binary, reason, quality, all_idx, args.batch_size, shuffle=False), device)
    return model, scaler, eval_from_pred(pred)


def gradient_sanity(model, scaler, X, task_id, device):
    model.eval()
    idx = np.arange(min(32, len(X)))
    xs = ((X[idx] - scaler.mean_) / scaler.scale_).astype(np.float32)
    x = torch.tensor(xs, dtype=torch.float32, device=device, requires_grad=True)
    tid = torch.tensor(task_id[idx], dtype=torch.long, device=device)
    out = model(x, tid)
    score = torch.sigmoid(out["quality"]).mean() + 0.25 * torch.softmax(out["binary_logits"], dim=-1)[:, 1].mean()
    grad = torch.autograd.grad(score, x, retain_graph=False)[0]
    return {
        "input_grad_norm": float(grad.norm().detach().cpu()),
        "score_value": float(score.detach().cpu()),
        "usable_for_feature_guidance": bool(torch.isfinite(grad).all().item() and grad.norm().item() > 1e-8),
    }


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, meta = build_or_load_dataset(args)
    X = data["X"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
    task = data["task"]
    task_id = data["task_id"].astype(np.int64)
    groups = data["groups"]

    keep = balanced_indices(reason, task, args.max_per_task_class, args.seed)
    X, reason, binary, quality = X[keep], reason[keep], binary[keep], quality[keep]
    task, task_id, groups = task[keep], task_id[keep], groups[keep]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    rows = []
    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(cv.split(X, reason, groups)):
        print(f"fold {fold}", flush=True)
        row = run_split(X, task_id, binary, reason, quality, tr, te, device, args)
        row["fold"] = fold
        rows.append(row)
        print(
            {
                "fold": fold,
                "binary_auc": row.get("binary_auc"),
                "reason_macro_f1": row.get("reason_macro_f1"),
                "quality_corr": row.get("quality_corr"),
            },
            flush=True,
        )

    final_model, scaler, train_metrics = train_final(X, task_id, binary, reason, quality, device, args)
    sanity = gradient_sanity(final_model, scaler, X, task_id, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "data": {
            "n": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "task_counts": {k: int(v) for k, v in Counter(task.tolist()).items()},
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
    (OUT_DIR / "ptg_proxy_scorer_v2_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "model_class": "PTGProxyScorerV2",
            "feature_dim": int(X.shape[1]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "task_to_id": TASK_TO_ID,
            "reason_names": REASON_NAMES,
            "feature_blocks": FEATURE_BLOCKS,
            "meta": meta,
        },
        OUT_DIR / "ptg_proxy_scorer_v2_final.pt",
    )
    print(json.dumps({"mixed_group_cv": result["mixed_group_cv"], "gradient_sanity": sanity}, ensure_ascii=False, indent=2))
    print(f"Saved {OUT_DIR / 'ptg_proxy_scorer_v2_eval.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--final_epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--binary_weight", type=float, default=1.0)
    parser.add_argument("--reason_weight", type=float, default=0.55)
    parser.add_argument("--quality_weight", type=float, default=0.75)
    parser.add_argument("--max_per_task_class", type=int, default=1200)
    parser.add_argument("--max_insertion_per_episode", type=int, default=220)
    parser.add_argument("--insertion_window", type=int, default=8)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=16)
    parser.add_argument("--board_force_source", default="left_force")
    parser.add_argument("--board_scheme", default="t5_scoreband")
    parser.add_argument("--force_rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
