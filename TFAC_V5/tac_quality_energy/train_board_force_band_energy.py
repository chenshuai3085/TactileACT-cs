#!/usr/bin/env python3
"""Train a differentiable force-band board TacQualityEnergy scorer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.force_band_runtime import (  # noqa: E402
    ForceBandTacQualityEnergy,
    ForceBandTacQualityEnergyRuntime,
    REASON_TO_ID,
)


DEFAULT_FEATURES = Path("/home/chenshuai/Project/output/tac_quality_board_force_band_eval_mlp/board_force_band_features.npz")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/board_force_band_tac_quality_energy")


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def finite_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_simple(x: np.ndarray, y: np.ndarray) -> float | None:
    return finite_corr(rankdata_simple(np.asarray(x)), rankdata_simple(np.asarray(y)))


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def load_features(path: Path, variant: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {
        "x": data[variant].astype(np.float32),
        "binary": data["binary"].astype(np.int64),
        "reason": data["reason"].astype(np.int64),
        "quality": data["quality"].astype(np.float32),
        "groups": data["groups"].astype(str),
        "labels": data["labels"].astype(str),
    }


def make_teacher(x: np.ndarray, binary: np.ndarray, groups: np.ndarray, seed: int, n_splits: int) -> np.ndarray:
    splits = list(GroupKFold(n_splits=min(n_splits, len(np.unique(groups)))).split(x, binary, groups))
    oof = np.full(len(x), np.nan, dtype=np.float32)
    for fold, (train_idx, test_idx) in enumerate(splits):
        if fold % 2 == 0:
            teacher = HistGradientBoostingClassifier(max_iter=260, learning_rate=0.05, max_leaf_nodes=31, random_state=seed + fold)
        else:
            teacher = RandomForestClassifier(
                n_estimators=260,
                max_depth=18,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=4,
                random_state=seed + fold,
            )
        teacher.fit(x[train_idx], binary[train_idx])
        proba = teacher.predict_proba(x[test_idx])
        classes = np.asarray(teacher.classes_)
        good_col = int(np.where(classes == 1)[0][0])
        oof[test_idx] = proba[:, good_col].astype(np.float32)
    if np.any(~np.isfinite(oof)):
        fill = float(np.nanmean(oof))
        oof[~np.isfinite(oof)] = fill
    return np.clip(oof, 1e-4, 1.0 - 1e-4)


def evaluate_arrays(binary: np.ndarray, reason: np.ndarray, quality: np.ndarray, outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    p_good = outputs["p_good"]
    reason_pred = outputs["reason_prob"].argmax(axis=1)
    binary_pred = (p_good >= 0.5).astype(np.int64)
    return {
        "binary_auc": safe_auc(binary, p_good),
        "binary_balanced_accuracy": float(balanced_accuracy_score(binary, binary_pred)),
        "binary_macro_f1": float(f1_score(binary, binary_pred, average="macro", zero_division=0)),
        "reason_macro_f1": float(f1_score(reason, reason_pred, average="macro", zero_division=0)),
        "quality_spearman": spearman_simple(outputs["quality_score"], quality),
        "energy_quality_spearman": spearman_simple(outputs["energy_logit"], quality),
        "teacher_spearman": spearman_simple(outputs["teacher_score"], outputs["teacher_target"]),
    }


def collect_outputs(model: ForceBandTacQualityEnergy, x: np.ndarray, teacher_target: np.ndarray, device: torch.device, batch_size: int) -> Dict[str, np.ndarray]:
    model.eval()
    chunks: Dict[str, List[np.ndarray]] = {
        "p_good": [],
        "reason_prob": [],
        "quality_score": [],
        "teacher_score": [],
        "energy_logit": [],
        "teacher_target": [],
    }
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            out = model(xb)
            chunks["p_good"].append(torch.softmax(out["binary_logits"], dim=-1)[:, 1].cpu().numpy())
            chunks["reason_prob"].append(torch.softmax(out["reason_logits"], dim=-1).cpu().numpy())
            chunks["quality_score"].append(torch.sigmoid(out["quality_logit"]).cpu().numpy())
            chunks["teacher_score"].append(torch.sigmoid(out["teacher_logit"]).cpu().numpy())
            chunks["energy_logit"].append(out["energy_logit"].cpu().numpy())
            chunks["teacher_target"].append(teacher_target[start : start + batch_size])
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def make_loader(
    x: np.ndarray,
    binary: np.ndarray,
    reason: np.ndarray,
    quality: np.ndarray,
    teacher: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(binary.astype(np.int64)),
        torch.from_numpy(reason.astype(np.int64)),
        torch.from_numpy(quality.astype(np.float32)),
        torch.from_numpy(teacher.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def compute_loss(out: Dict[str, torch.Tensor], binary: torch.Tensor, reason: torch.Tensor, quality: torch.Tensor, teacher: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    binary_loss = F.cross_entropy(out["binary_logits"], binary)
    reason_loss = F.cross_entropy(out["reason_logits"], reason)
    quality_loss = F.binary_cross_entropy_with_logits(out["quality_logit"], quality)
    teacher_loss = F.binary_cross_entropy_with_logits(out["teacher_logit"], teacher)
    positive_reason_logit = out["reason_logits"][:, REASON_TO_ID["positive"]]
    bad_reason_logit = torch.logsumexp(
        torch.stack(
            [
                out["reason_logits"][:, REASON_TO_ID["too_small"]],
                out["reason_logits"][:, REASON_TO_ID["too_large"]],
                out["reason_logits"][:, REASON_TO_ID["oscillate"]],
            ],
            dim=-1,
        ),
        dim=-1,
    )
    ranking_target = 2.0 * binary.float() - 1.0
    margin_loss = F.softplus(-ranking_target * (positive_reason_logit - bad_reason_logit)).mean()
    energy_target = 0.45 * quality + 0.35 * binary.float() + 0.20 * teacher
    energy_loss = F.smooth_l1_loss(torch.sigmoid(out["energy_logit"]), energy_target.clamp(0.0, 1.0))
    loss = (
        args.binary_weight * binary_loss
        + args.reason_weight * reason_loss
        + args.quality_weight * quality_loss
        + args.teacher_weight * teacher_loss
        + args.margin_weight * margin_loss
        + args.energy_weight * energy_loss
    )
    return loss, {
        "binary_loss": float(binary_loss.detach().cpu()),
        "reason_loss": float(reason_loss.detach().cpu()),
        "quality_loss": float(quality_loss.detach().cpu()),
        "teacher_loss": float(teacher_loss.detach().cpu()),
        "margin_loss": float(margin_loss.detach().cpu()),
        "energy_loss": float(energy_loss.detach().cpu()),
    }


def save_checkpoint(path: Path, model: ForceBandTacQualityEnergy, scaler: StandardScaler, args: argparse.Namespace, metrics: Dict[str, Any]) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "feature_dim": int(args.feature_dim),
        "hidden": int(args.hidden),
        "dropout": float(args.dropout),
        "feature_variant": args.feature_variant,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "reason_to_id": REASON_TO_ID,
        "metrics": metrics,
        "loss_weights": {
            "binary": args.binary_weight,
            "reason": args.reason_weight,
            "quality": args.quality_weight,
            "teacher": args.teacher_weight,
            "margin": args.margin_weight,
            "energy": args.energy_weight,
        },
    }
    torch.save(payload, path)


def plot_history(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    val_auc = [r["val_binary_auc"] for r in history]
    val_f1 = [r["val_reason_macro_f1"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(epochs, val_auc, label="val AUC")
    axes[1].plot(epochs, val_f1, label="val reason F1")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_csv(history: List[Dict[str, Any]], path: Path) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def train_once(args: argparse.Namespace, payload: Dict[str, np.ndarray], out_dir: Path) -> Dict[str, Any]:
    x_raw = payload["x"]
    binary = payload["binary"]
    reason = payload["reason"]
    quality = payload["quality"]
    groups = payload["groups"]
    teacher = make_teacher(x_raw, binary, groups, args.seed, args.teacher_splits) if args.teacher_path is None else np.load(args.teacher_path).astype(np.float32)

    splits = list(GroupKFold(n_splits=min(args.n_splits, len(np.unique(groups)))).split(x_raw, binary, groups))
    train_idx, val_idx = splits[args.val_fold]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_raw[train_idx]).astype(np.float32)
    x_val = scaler.transform(x_raw[val_idx]).astype(np.float32)
    x_all = scaler.transform(x_raw).astype(np.float32)
    args.feature_dim = x_raw.shape[1]

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = ForceBandTacQualityEnergy(in_dim=x_raw.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(x_train, binary[train_idx], reason[train_idx], quality[train_idx], teacher[train_idx], args.batch_size, True)
    val_loader = make_loader(x_val, binary[val_idx], reason[val_idx], quality[val_idx], teacher[val_idx], args.batch_size, False)

    best_score = -float("inf")
    best_metrics: Dict[str, Any] = {}
    patience_left = args.patience
    history: List[Dict[str, Any]] = []
    best_path = out_dir / "force_band_tac_quality_energy_best.pt"
    latest_path = out_dir / "force_band_tac_quality_energy_latest.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, yr, yq, yt in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yr = yr.to(device)
            yq = yq.to(device)
            yt = yt.to(device)
            out = model(xb)
            loss, _ = compute_loss(out, yb, yr, yq, yt, args)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, yr, yq, yt in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yr = yr.to(device)
                yq = yq.to(device)
                yt = yt.to(device)
                out = model(xb)
                loss, _ = compute_loss(out, yb, yr, yq, yt, args)
                val_losses.append(float(loss.detach().cpu()))

        val_outputs = collect_outputs(model, x_val, teacher[val_idx], device, args.batch_size)
        val_metrics = evaluate_arrays(binary[val_idx], reason[val_idx], quality[val_idx], val_outputs)
        train_outputs = collect_outputs(model, x_train, teacher[train_idx], device, args.batch_size)
        train_metrics = evaluate_arrays(binary[train_idx], reason[train_idx], quality[train_idx], train_outputs)

        score = (
            float(val_metrics["binary_auc"] or 0.0)
            + float(val_metrics["binary_balanced_accuracy"] or 0.0)
            + float(val_metrics["reason_macro_f1"] or 0.0)
            + 0.25 * float(val_metrics["quality_spearman"] or 0.0)
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
            "train_binary_auc": train_metrics["binary_auc"],
            "train_reason_macro_f1": train_metrics["reason_macro_f1"],
            "val_binary_auc": val_metrics["binary_auc"],
            "val_balanced_accuracy": val_metrics["binary_balanced_accuracy"],
            "val_binary_macro_f1": val_metrics["binary_macro_f1"],
            "val_reason_macro_f1": val_metrics["reason_macro_f1"],
            "val_quality_spearman": val_metrics["quality_spearman"],
            "val_energy_quality_spearman": val_metrics["energy_quality_spearman"],
            "score": score,
        }
        history.append(row)

        if epoch == 1 or epoch % args.log_interval == 0:
            print(
                f"Ep {epoch:04d}/{args.epochs} train={row['train_loss']:.4f} val={row['val_loss']:.4f} "
                f"auc={fmt(row['val_binary_auc'])} bacc={fmt(row['val_balanced_accuracy'])} "
                f"reason={fmt(row['val_reason_macro_f1'])} qsp={fmt(row['val_quality_spearman'])}",
                flush=True,
            )

        latest_metrics = {"epoch": epoch, "val": val_metrics, "train": train_metrics, "score": score}
        save_checkpoint(latest_path, model, scaler, args, latest_metrics)
        if score > best_score + args.min_delta:
            best_score = score
            best_metrics = latest_metrics
            save_checkpoint(best_path, model, scaler, args, best_metrics)
            patience_left = args.patience
        else:
            patience_left -= 1
            if epoch >= args.min_epochs and patience_left <= 0:
                print(f"Early stop at epoch {epoch}; best epoch={best_metrics.get('epoch')}", flush=True)
                break

    plot_history(history, out_dir / "train_curve.png")
    write_csv(history, out_dir / "train_history.csv")

    best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    all_outputs = collect_outputs(model, x_all, teacher, device, args.batch_size)
    all_metrics = evaluate_arrays(binary, reason, quality, all_outputs)

    result = {
        "feature_variant": args.feature_variant,
        "feature_dim": int(x_raw.shape[1]),
        "n": int(len(x_raw)),
        "n_groups": int(len(np.unique(groups))),
        "train_groups": int(len(np.unique(groups[train_idx]))),
        "val_groups": int(len(np.unique(groups[val_idx]))),
        "best": best_metrics,
        "all_fit_metrics": all_metrics,
        "checkpoint_best": str(best_path),
        "checkpoint_latest": str(latest_path),
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(best_path, out_dir / "force_band_tac_quality_energy_final.pt")
    return result


def gradient_smoke(ckpt_path: Path, out_dir: Path, feature_variant: str) -> Dict[str, Any]:
    device = "cpu"
    runtime = ForceBandTacQualityEnergyRuntime(str(ckpt_path), device=device)
    batch = 4
    marker = torch.randn(batch, 8, 9, 9, 2, requires_grad=True)
    action = torch.randn(batch, 8, 7, requires_grad=True)
    score = runtime.score(marker, right_marker_seq=marker, joint_action_seq=action, mode="profile")
    loss = score.mean()
    loss.backward()
    marker_grad = marker.grad
    action_grad = action.grad
    result = {
        "feature_variant": feature_variant,
        "score_mean": float(score.detach().mean()),
        "marker_grad_finite": bool(torch.isfinite(marker_grad).all()),
        "action_grad_finite": bool(torch.isfinite(action_grad).all()),
        "marker_grad_norm": float(marker_grad.norm()),
        "action_grad_norm": float(action_grad.norm()),
        "pass": bool(torch.isfinite(marker_grad).all() and torch.isfinite(action_grad).all() and marker_grad.norm() > 0 and action_grad.norm() > 0),
    }
    (out_dir / "gradient_smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def write_markdown(result: Dict[str, Any], smoke: Dict[str, Any], path: Path) -> None:
    best_val = result["best"]["val"]
    all_fit = result["all_fit_metrics"]
    lines = [
        "# Force-Band Board TacQualityEnergy Training",
        "",
        "This trains a differentiable MLP scorer for board-wiping DP classifier guidance.",
        "",
        "## Setup",
        "",
        f"- feature variant: `{result['feature_variant']}`",
        f"- feature dim: `{result['feature_dim']}`",
        f"- samples: `{result['n']}`",
        f"- episode groups: `{result['n_groups']}`",
        f"- best checkpoint: `{result['checkpoint_best']}`",
        "",
        "## Held-Out Validation",
        "",
        f"- best epoch: `{result['best']['epoch']}`",
        f"- AUC: `{fmt(best_val['binary_auc'])}`",
        f"- balanced accuracy: `{fmt(best_val['binary_balanced_accuracy'])}`",
        f"- binary macro F1: `{fmt(best_val['binary_macro_f1'])}`",
        f"- reason macro F1: `{fmt(best_val['reason_macro_f1'])}`",
        f"- quality Spearman: `{fmt(best_val['quality_spearman'])}`",
        f"- energy-quality Spearman: `{fmt(best_val['energy_quality_spearman'])}`",
        "",
        "## All-Fit Sanity Metrics",
        "",
        f"- AUC: `{fmt(all_fit['binary_auc'])}`",
        f"- balanced accuracy: `{fmt(all_fit['binary_balanced_accuracy'])}`",
        f"- reason macro F1: `{fmt(all_fit['reason_macro_f1'])}`",
        f"- quality Spearman: `{fmt(all_fit['quality_spearman'])}`",
        "",
        "## Gradient Smoke",
        "",
        f"- pass: `{smoke['pass']}`",
        f"- marker grad finite: `{smoke['marker_grad_finite']}`",
        f"- action grad finite: `{smoke['action_grad_finite']}`",
        f"- marker grad norm: `{fmt(smoke['marker_grad_norm'])}`",
        f"- action grad norm: `{fmt(smoke['action_grad_norm'])}`",
        "",
        "## Interpretation",
        "",
        "- This checkpoint is differentiable and can be used as the board force-band scorer candidate for Foresight-to-action gradient audits.",
        "- It still needs evaluation through the real multistep Foresight bridge before claiming DP guidance quality improvement.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature_variant", default="marker_action", choices=["marker_left", "left_marker_action", "marker_action"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--min_epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_fold", type=int, default=0)
    parser.add_argument("--teacher_splits", type=int, default=5)
    parser.add_argument("--teacher_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary_weight", type=float, default=1.0)
    parser.add_argument("--reason_weight", type=float, default=1.0)
    parser.add_argument("--quality_weight", type=float, default=0.55)
    parser.add_argument("--teacher_weight", type=float, default=0.45)
    parser.add_argument("--margin_weight", type=float, default=0.20)
    parser.add_argument("--energy_weight", type=float, default=0.20)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = load_features(args.features, args.feature_variant)
    result = train_once(args, payload, out_dir)
    smoke = gradient_smoke(Path(result["checkpoint_best"]), out_dir, args.feature_variant)
    write_markdown(result, smoke, out_dir / "force_band_tac_quality_energy_train.md")
    print(json.dumps({"result": result, "gradient_smoke": smoke}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
