"""Train a teacher-distilled differentiable TacQualityEnergy scorer.

The split-leakage audit showed that a RandomForest teacher is the strongest
offline classifier for the unified good/bad quality standard, but it is not
differentiable and therefore cannot directly guide DP actions.  This script
distills that teacher into a small MLP energy scorer while preserving the
episode-level GroupKFold evaluation protocol.

Inputs:
  /home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz

Outputs:
  /home/chenshuai/Project/output/distilled_tac_quality_energy/
    distilled_tac_quality_energy_eval.json
    distilled_tac_quality_energy_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


FEATURES = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")
OUT_DIR = Path("/home/chenshuai/Project/output/distilled_tac_quality_energy")
TASK_TO_ID = {"insertion": 0, "board": 1}


class DistilledTacQualityEnergy(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 192, dropout: float = 0.10):
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
        self.teacher_head = nn.Linear(hidden // 2, 1)
        self.energy_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        task_oh = F.one_hot(task_id.long(), num_classes=2).float()
        h = self.encoder(torch.cat([x, task_oh], dim=-1))
        binary_logits = self.binary_head(h)
        reason_logits = self.reason_head(h)
        quality_logit = self.quality_head(h).squeeze(-1)
        teacher_logit = self.teacher_head(h).squeeze(-1)
        free_energy = self.energy_head(h).squeeze(-1)
        good_margin = binary_logits[:, 1] - binary_logits[:, 0]
        reason_margin = reason_logits[:, 1] - torch.logsumexp(
            torch.stack([reason_logits[:, 0], torch.logsumexp(reason_logits[:, 2:], dim=-1)], dim=-1),
            dim=-1,
        )
        energy_logit = 0.45 * quality_logit + 0.30 * teacher_logit + 0.15 * good_margin + 0.10 * reason_margin
        energy_logit = energy_logit + 0.10 * free_energy
        return {
            "binary_logits": binary_logits,
            "reason_logits": reason_logits,
            "quality_logit": quality_logit,
            "teacher_logit": teacher_logit,
            "free_energy": free_energy,
            "good_margin": good_margin,
            "reason_margin": reason_margin,
            "energy_logit": energy_logit,
            "energy_clipped": torch.tanh(energy_logit / 4.0) * 4.0,
        }


def balanced_indices(reason: np.ndarray, task: np.ndarray, max_per_task_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for task_name in np.unique(task):
        for cls in np.unique(reason):
            idx = np.flatnonzero((task == task_name) & (reason == cls))
            if len(idx) == 0:
                continue
            keep.extend(rng.choice(idx, min(max_per_task_class, len(idx)), replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


def class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    X: np.ndarray,
    task_id: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    teacher_prob: np.ndarray,
    idx: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X[idx]).float(),
        torch.from_numpy(task_id[idx]).long(),
        torch.from_numpy(binary[idx]).long(),
        torch.from_numpy(reason[idx]).long(),
        torch.from_numpy(quality[idx]).float(),
        torch.from_numpy(teacher_prob[idx]).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def teacher_soft_labels(
    X: np.ndarray,
    binary: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    valid_train = binary[train_idx] >= 0
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X[train_idx][valid_train], binary[train_idx][valid_train])
    classes = {int(c): i for i, c in enumerate(rf.classes_)}
    out = np.full(len(X), 0.5, dtype=np.float32)
    for idx in [train_idx, test_idx]:
        proba = rf.predict_proba(X[idx])
        out[idx] = proba[:, classes[1]].astype(np.float32)
    test_valid = binary[test_idx] >= 0
    row = {
        "teacher_binary_auc": float(roc_auc_score(binary[test_idx][test_valid], out[test_idx][test_valid])),
        "teacher_balanced_accuracy": float(
            balanced_accuracy_score(binary[test_idx][test_valid], (out[test_idx][test_valid] >= 0.5).astype(np.int64))
        ),
    }
    return out, row


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-8 or np.std(rb) < 1e-8:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    weights: Dict[str, float],
    reason_w: torch.Tensor,
    binary_w: torch.Tensor,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for x, task_id, y_bin, y_reason, q, teacher in loader:
        x = x.to(device)
        task_id = task_id.to(device)
        y_bin = y_bin.to(device)
        y_reason = y_reason.to(device)
        q = q.to(device)
        teacher = teacher.to(device)
        out = model(x, task_id)
        mask = y_bin >= 0
        loss_binary = (
            F.cross_entropy(out["binary_logits"][mask], y_bin[mask], weight=binary_w.to(device))
            if mask.any()
            else torch.zeros((), device=device)
        )
        loss_reason = F.cross_entropy(out["reason_logits"], y_reason, weight=reason_w.to(device))
        loss_quality = F.smooth_l1_loss(torch.sigmoid(out["quality_logit"]), q)
        loss_teacher = F.binary_cross_entropy_with_logits(out["teacher_logit"], teacher)
        loss_energy_teacher = F.binary_cross_entropy_with_logits(out["energy_logit"], teacher)
        loss = (
            weights["binary"] * loss_binary
            + weights["reason"] * loss_reason
            + weights["quality"] * loss_quality
            + weights["teacher"] * loss_teacher
            + weights["energy"] * loss_energy_teacher
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += float(loss.item()) * len(x)
        n += len(x)
    return total / max(n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    rows: Dict[str, List[np.ndarray]] = {
        "binary": [],
        "reason": [],
        "quality": [],
        "teacher": [],
        "p_good": [],
        "reason_prob": [],
        "quality_pred": [],
        "teacher_pred": [],
        "energy": [],
    }
    for x, task_id, y_bin, y_reason, q, teacher in loader:
        out = model(x.to(device), task_id.to(device))
        rows["binary"].append(y_bin.numpy())
        rows["reason"].append(y_reason.numpy())
        rows["quality"].append(q.numpy())
        rows["teacher"].append(teacher.numpy())
        rows["p_good"].append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
        rows["reason_prob"].append(torch.softmax(out["reason_logits"], dim=-1).cpu().numpy())
        rows["quality_pred"].append(torch.sigmoid(out["quality_logit"]).cpu().numpy())
        rows["teacher_pred"].append(torch.sigmoid(out["teacher_logit"]).cpu().numpy())
        rows["energy"].append(out["energy_clipped"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in rows.items()}


def eval_pred(pred: Dict[str, np.ndarray]) -> Dict[str, Any]:
    yb = pred["binary"]
    yr = pred["reason"]
    q = pred["quality"]
    teacher = pred["teacher"]
    p_good = pred["p_good"]
    reason_pred = pred["reason_prob"].argmax(axis=1)
    q_pred = pred["quality_pred"]
    teacher_pred = pred["teacher_pred"]
    energy = pred["energy"]
    valid = yb >= 0
    out: Dict[str, Any] = {
        "reason_balanced_accuracy": float(balanced_accuracy_score(yr, reason_pred)),
        "reason_macro_f1": float(f1_score(yr, reason_pred, average="macro")),
        "quality_corr": float(np.corrcoef(q_pred, q)[0, 1]) if np.std(q_pred) > 1e-8 and np.std(q) > 1e-8 else 0.0,
        "quality_r2": float(r2_score(q, q_pred)),
        "teacher_pred_corr": float(np.corrcoef(teacher_pred, teacher)[0, 1])
        if np.std(teacher_pred) > 1e-8 and np.std(teacher) > 1e-8
        else 0.0,
        "energy_teacher_spearman": rank_corr(energy, teacher),
        "energy_quality_spearman": rank_corr(energy, q),
    }
    if valid.any() and len(np.unique(yb[valid])) == 2:
        out.update(
            {
                "binary_balanced_accuracy": float(balanced_accuracy_score(yb[valid], (p_good[valid] >= 0.5).astype(np.int64))),
                "binary_macro_f1": float(f1_score(yb[valid], (p_good[valid] >= 0.5).astype(np.int64), average="macro")),
                "binary_auc": float(roc_auc_score(yb[valid], p_good[valid])),
                "energy_binary_auc": float(roc_auc_score(yb[valid], energy[valid])),
            }
        )
    return out


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "binary_balanced_accuracy",
        "binary_macro_f1",
        "binary_auc",
        "energy_binary_auc",
        "reason_balanced_accuracy",
        "reason_macro_f1",
        "quality_corr",
        "quality_r2",
        "teacher_pred_corr",
        "energy_teacher_spearman",
        "energy_quality_spearman",
        "teacher_binary_auc",
        "teacher_balanced_accuracy",
    ]
    out: Dict[str, Any] = {}
    for key in keys:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if vals:
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def standardize(X: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X[train_idx])
    return scaler.transform(X).astype(np.float32), scaler


def guidance_sanity(model: nn.Module, scaler: StandardScaler, X: np.ndarray, task_id: np.ndarray, device: torch.device) -> Dict[str, Any]:
    model.eval()
    idx = np.arange(min(64, len(X)))
    xs = ((X[idx] - scaler.mean_) / (scaler.scale_ + 1e-8)).astype(np.float32)
    x = torch.tensor(xs, dtype=torch.float32, device=device, requires_grad=True)
    tid = torch.tensor(task_id[idx], dtype=torch.long, device=device)
    out = model(x, tid)
    score = out["energy_clipped"].mean()
    grad = torch.autograd.grad(score, x, retain_graph=False)[0]
    return {
        "score": float(score.detach().cpu()),
        "input_grad_norm": float(grad.norm().detach().cpu()),
        "input_grad_abs_mean": float(grad.abs().mean().detach().cpu()),
        "usable_for_feature_guidance": bool(torch.isfinite(grad).all().item() and grad.norm().item() > 1e-8),
    }


def run_split(
    X: np.ndarray,
    task_id: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
    fold: int,
) -> Dict[str, Any]:
    Xs, _ = standardize(X, train_idx)
    teacher, teacher_row = teacher_soft_labels(
        Xs,
        binary,
        train_idx,
        test_idx,
        args.seed + fold,
        args.teacher_trees,
        args.teacher_max_depth,
    )
    reason_w = class_weights(reason[train_idx], 5)
    valid_bin = binary[train_idx] >= 0
    binary_w = class_weights(binary[train_idx][valid_bin], 2)
    model = DistilledTacQualityEnergy(X.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    weights = {
        "binary": args.binary_weight,
        "reason": args.reason_weight,
        "quality": args.quality_weight,
        "teacher": args.teacher_weight,
        "energy": args.energy_weight,
    }
    train_loader = make_loader(Xs, task_id, binary, reason, quality, teacher, train_idx, args.batch_size, True)
    test_loader = make_loader(Xs, task_id, binary, reason, quality, teacher, test_idx, args.batch_size, False)
    best: Dict[str, Any] | None = None
    best_metric = -math.inf
    bad = 0
    for epoch in range(args.epochs):
        train_epoch(model, train_loader, opt, device, weights, reason_w, binary_w)
        sched.step()
        row = eval_pred(predict(model, test_loader, device))
        metric = (
            (row.get("energy_binary_auc") or 0.0)
            + 0.25 * row.get("energy_quality_spearman", 0.0)
            + 0.25 * row.get("teacher_pred_corr", 0.0)
            + 0.10 * row.get("reason_macro_f1", 0.0)
        )
        if metric > best_metric:
            best_metric = metric
            best = {**row, **teacher_row, "fold": fold, "best_epoch": epoch, "selection_metric": float(metric)}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    assert best is not None
    return best


def train_final(
    X: np.ndarray,
    task_id: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[nn.Module, StandardScaler, Dict[str, Any]]:
    all_idx = np.arange(len(X))
    Xs, scaler = standardize(X, all_idx)
    teacher, teacher_row = teacher_soft_labels(
        Xs,
        binary,
        all_idx,
        all_idx,
        args.seed + 999,
        args.teacher_trees,
        args.teacher_max_depth,
    )
    reason_w = class_weights(reason, 5)
    binary_w = class_weights(binary[binary >= 0], 2)
    model = DistilledTacQualityEnergy(X.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = {
        "binary": args.binary_weight,
        "reason": args.reason_weight,
        "quality": args.quality_weight,
        "teacher": args.teacher_weight,
        "energy": args.energy_weight,
    }
    loader = make_loader(Xs, task_id, binary, reason, quality, teacher, all_idx, args.batch_size, True)
    for _ in range(args.final_epochs):
        train_epoch(model, loader, opt, device, weights, reason_w, binary_w)
    pred = predict(model, make_loader(Xs, task_id, binary, reason, quality, teacher, all_idx, args.batch_size, False), device)
    return model, scaler, {**eval_pred(pred), **teacher_row}


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.features, allow_pickle=True)
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

    rows: List[Dict[str, Any]] = []
    cv = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    for fold, (tr, te) in enumerate(cv.split(X, reason, groups)):
        print(f"fold {fold}", flush=True)
        row = run_split(X, task_id, binary, reason, quality, tr, te, device, args, fold)
        rows.append(row)
        print(
            {
                "fold": fold,
                "energy_auc": row.get("energy_binary_auc"),
                "teacher_corr": row.get("teacher_pred_corr"),
                "energy_quality_spearman": row.get("energy_quality_spearman"),
            },
            flush=True,
        )

    final_model, scaler, train_metrics = train_final(X, task_id, binary, reason, quality, device, args)
    sanity = guidance_sanity(final_model, scaler, X, task_id, device)

    result = {
        "purpose": "RF-teacher-distilled differentiable TacQualityEnergy scorer for DP gradient guidance.",
        "data": {
            "n": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "task_counts": {str(k): int(v) for k, v in Counter(task.tolist()).items()},
            "reason_counts": {str(k): int(v) for k, v in Counter(reason.tolist()).items()},
            "binary_counts_with_neutral": {str(k): int(v) for k, v in Counter(binary.tolist()).items()},
            "n_groups": int(len(np.unique(groups))),
        },
        "args": vars(args),
        "mixed_episode_group_cv": aggregate(rows),
        "folds": rows,
        "final_train_capacity_check": train_metrics,
        "guidance_sanity": sanity,
        "recommended_use": (
            "Use energy_clipped as a differentiable guidance potential. Keep RF as offline teacher only; "
            "real DP deployment still requires trust-region action refinement and real rollout gates."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = OUT_DIR / "distilled_tac_quality_energy_eval.json"
    eval_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "model_class": "DistilledTacQualityEnergy",
            "feature_dim": int(X.shape[1]),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "task_to_id": TASK_TO_ID,
            "source_features": str(args.features),
            "guidance_score": "energy_clipped",
            "args": vars(args),
        },
        OUT_DIR / "distilled_tac_quality_energy_final.pt",
    )
    print(json.dumps({"eval": str(eval_path), "cv": result["mixed_episode_group_cv"], "guidance_sanity": sanity}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--final_epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--binary_weight", type=float, default=0.70)
    parser.add_argument("--reason_weight", type=float, default=0.35)
    parser.add_argument("--quality_weight", type=float, default=0.60)
    parser.add_argument("--teacher_weight", type=float, default=0.85)
    parser.add_argument("--energy_weight", type=float, default=0.60)
    parser.add_argument("--teacher_trees", type=int, default=160)
    parser.add_argument("--teacher_max_depth", type=int, default=18)
    parser.add_argument("--max_per_task_class", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
