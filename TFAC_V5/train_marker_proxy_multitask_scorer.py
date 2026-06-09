"""Train a differentiable multi-head marker-proxy tactile quality scorer.

This is the deployable direction after the marker-proxy GBM experiment:
  - binary head: main classifier-guidance objective, P(good)
  - T4 head: auxiliary interpretable contact-quality reason
  - score head: continuous quality score for ranking/calibration

The model is intentionally small MLP so it can later be placed inside the DP
denoising loop and differentiated with respect to predicted marker/latent.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATA = Path("/home/chenshuai/Project/output/marker_proxy_scorer/marker_proxy_features.npz")
OUT_DIR = Path("/home/chenshuai/Project/output/marker_proxy_multitask_scorer")


class MultiHeadMarkerScorer(nn.Module):
    def __init__(self, in_dim, hidden=128, dropout=0.15):
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
            nn.SiLU(),
        )
        self.binary_head = nn.Linear(hidden // 2, 2)
        self.t4_head = nn.Linear(hidden // 2, 4)
        self.score_head = nn.Linear(hidden // 2, 1)

    def forward(self, x, task_id):
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([x, task_oh], dim=-1))
        return {
            "binary_logits": self.binary_head(h),
            "t4_logits": self.t4_head(h),
            "score": self.score_head(h).squeeze(-1),
        }


def sample_balanced_indices(y_t4, task, max_per_task_class=1400, seed=42):
    rng = np.random.default_rng(seed)
    keep = []
    for task_name in np.unique(task):
        for cls in np.unique(y_t4):
            idx = np.flatnonzero((task == task_name) & (y_t4 == cls))
            if len(idx):
                keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


def make_loader(X, task_id, y_bin, y_t4, score, indices, batch_size=256, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(X[indices]).float(),
        torch.from_numpy(task_id[indices]).long(),
        torch.from_numpy(y_bin[indices]).long(),
        torch.from_numpy(y_t4[indices]).long(),
        torch.from_numpy(score[indices]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_one(model, loader, opt, device, loss_weights):
    model.train()
    total = Counter()
    n = 0
    for x, task_id, y_bin, y_t4, score in loader:
        x = x.to(device)
        task_id = task_id.to(device)
        y_bin = y_bin.to(device)
        y_t4 = y_t4.to(device)
        score = score.to(device)
        out = model(x, task_id)
        loss_bin = F.cross_entropy(out["binary_logits"], y_bin)
        loss_t4 = F.cross_entropy(out["t4_logits"], y_t4)
        loss_score = F.smooth_l1_loss(torch.sigmoid(out["score"]), score)
        loss = (
            loss_weights["binary"] * loss_bin
            + loss_weights["t4"] * loss_t4
            + loss_weights["score"] * loss_score
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        bs = len(x)
        total["loss"] += float(loss.item()) * bs
        total["loss_bin"] += float(loss_bin.item()) * bs
        total["loss_t4"] += float(loss_t4.item()) * bs
        total["loss_score"] += float(loss_score.item()) * bs
        n += bs
    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    yb, yt, qs, pb, pt, ps = [], [], [], [], [], []
    for x, task_id, y_bin, y_t4, score in loader:
        out = model(x.to(device), task_id.to(device))
        yb.append(y_bin.numpy())
        yt.append(y_t4.numpy())
        qs.append(score.numpy())
        pb.append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
        pt.append(torch.softmax(out["t4_logits"], dim=-1).cpu().numpy())
        ps.append(torch.sigmoid(out["score"]).cpu().numpy())
    yb = np.concatenate(yb)
    yt = np.concatenate(yt)
    qs = np.concatenate(qs)
    pb = np.concatenate(pb)
    pt = np.concatenate(pt)
    ps = np.concatenate(ps)
    pred_b = (pb >= 0.5).astype(np.int64)
    pred_t = pt.argmax(axis=1)
    out = {
        "binary_balanced_accuracy": float(balanced_accuracy_score(yb, pred_b)),
        "binary_macro_f1": float(f1_score(yb, pred_b, average="macro")),
        "binary_auc": float(roc_auc_score(yb, pb)) if len(np.unique(yb)) == 2 else None,
        "t4_balanced_accuracy": float(balanced_accuracy_score(yt, pred_t)),
        "t4_macro_f1": float(f1_score(yt, pred_t, average="macro")),
        "score_corr": float(np.corrcoef(ps, qs)[0, 1]) if np.std(ps) > 1e-8 and np.std(qs) > 1e-8 else 0.0,
    }
    return out


def run_fold(X, task_id, y_bin, y_t4, score, train_idx, test_idx, device, epochs=80):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    X_scaled = X.copy()
    X_scaled[train_idx] = X_train
    X_scaled[test_idx] = X_test

    model = MultiHeadMarkerScorer(X.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    train_loader = make_loader(X_scaled, task_id, y_bin, y_t4, score, train_idx, shuffle=True)
    test_loader = make_loader(X_scaled, task_id, y_bin, y_t4, score, test_idx, shuffle=False)

    best = None
    best_metric = -1
    patience = 15
    bad = 0
    weights = {"binary": 1.0, "t4": 0.35, "score": 0.5}
    for _ in range(epochs):
        train_one(model, train_loader, opt, device, weights)
        metrics = evaluate(model, test_loader, device)
        metric = metrics["binary_auc"] + 0.25 * metrics["score_corr"]
        if metric > best_metric:
            best_metric = metric
            best = {k: float(v) if v is not None else None for k, v in metrics.items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best


def aggregate(rows):
    out = {}
    for key in rows[0]:
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_t4 = data["y_t4"].astype(np.int64)
    y_bin_raw = data["y_binary"].astype(np.int64)
    score = data["score"].astype(np.float32)
    task = data["task"]
    groups = data["groups"]

    valid = y_bin_raw >= 0
    keep = sample_balanced_indices(y_t4[valid], task[valid])
    valid_idx = np.flatnonzero(valid)[keep]
    X = X[valid_idx]
    y_t4 = y_t4[valid_idx]
    y_bin = y_bin_raw[valid_idx]
    score = score[valid_idx]
    task = task[valid_idx]
    groups = groups[valid_idx]
    task_id = (task == "board").astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {
        "data": {
            "n": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "task_counts": {k: int(v) for k, v in Counter(task.tolist()).items()},
            "binary_counts": {str(k): int(v) for k, v in Counter(y_bin.tolist()).items()},
            "t4_counts": {str(k): int(v) for k, v in Counter(y_t4.tolist()).items()},
        },
        "mixed_group_cv": {},
        "cross_task": {},
    }

    fold_rows = []
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(cv.split(X, y_bin, groups)):
        print(f"fold {fold}", flush=True)
        row = run_fold(X, task_id, y_bin, y_t4, score, tr, te, device)
        row["fold"] = fold
        fold_rows.append(row)
        print(row, flush=True)
    results["mixed_group_cv"] = aggregate(fold_rows)
    results["mixed_group_cv"]["folds"] = fold_rows

    for train_task, test_task in [("insertion", "board"), ("board", "insertion")]:
        tr = np.flatnonzero(task == train_task)
        te = np.flatnonzero(task == test_task)
        row = run_fold(X, task_id, y_bin, y_t4, score, tr, te, device, epochs=100)
        results["cross_task"][f"{train_task}_to_{test_task}"] = row
        print(f"cross {train_task}->{test_task}: {row}", flush=True)

    with open(OUT_DIR / "marker_proxy_multitask_eval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    torch.save(
        {
            "model_class": "MultiHeadMarkerScorer",
            "feature_dim": int(X.shape[1]),
            "note": "Evaluation folds train transient models; retrain final deployment scorer separately.",
        },
        OUT_DIR / "marker_proxy_multitask_meta.pt",
    )
    print(f"Saved {OUT_DIR / 'marker_proxy_multitask_eval.json'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
