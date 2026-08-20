"""Train the board latent chunk energy scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_chunk_energy.labels import BOARD_CLASS_NAMES  # noqa: E402
from TFAC_V5.board_chunk_energy.losses import expert_margin_loss, supervised_contrastive_loss  # noqa: E402
from TFAC_V5.board_latent_energy.dataset import (  # noqa: E402
    BoardLatentChunkDataset,
    ChunkDatasetConfig,
    INPUT_MODE,
    SCHEMA_VERSION,
    TASK,
    build_rows,
    compute_normalization,
    split_rows_by_episode,
    write_manifest,
)
from TFAC_V5.board_latent_energy.model import BoardLatentEnergyScorer  # noqa: E402
from TFAC_V5.board_latent_energy.vae_utils import (  # noqa: E402
    DEFAULT_TACTILE_VAE_CKPT,
    infer_vae_meta,
    load_tactile_vae_checkpoint,
    vae_checkpoint_identity,
)
from TFAC_V5.tac_quality_energy.tactile_only_latent import build_tactile_only_checkpoint  # noqa: E402


DEFAULT_OUT = "/home/chenshuai/Project/output/board_latent_energy"


def class_counts(labels: List[int]) -> Dict[str, int]:
    counts = {name: 0 for name in BOARD_CLASS_NAMES}
    for label in labels:
        counts[BOARD_CLASS_NAMES[int(label)]] += 1
    return counts


def make_loader(dataset: BoardLatentChunkDataset, batch_size: int, balanced: bool, seed: int):
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    labels = np.asarray([row.label for row in dataset.rows], dtype=np.int64)
    counts = np.bincount(labels, minlength=len(BOARD_CLASS_NAMES)).astype(np.float64)
    weights = 1.0 / np.maximum(counts[labels], 1.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, score_good: torch.Tensor, expert_margin: torch.Tensor) -> Dict[str, object]:
    pred = logits.argmax(dim=1)
    labels_cpu = labels.detach().cpu()
    pred_cpu = pred.detach().cpu()
    score_cpu = score_good.detach().cpu()
    margin_cpu = expert_margin.detach().cpu()
    acc = float((pred_cpu == labels_cpu).float().mean().item())
    recalls = []
    precisions = []
    for cls in range(len(BOARD_CLASS_NAMES)):
        tp = ((pred_cpu == cls) & (labels_cpu == cls)).sum().item()
        fn = ((pred_cpu != cls) & (labels_cpu == cls)).sum().item()
        fp = ((pred_cpu == cls) & (labels_cpu != cls)).sum().item()
        recalls.append(tp / max(1, tp + fn))
        precisions.append(tp / max(1, tp + fp))
    macro_f1 = float(np.mean([2 * p * r / max(1e-8, p + r) for p, r in zip(precisions, recalls)]))

    pos = score_cpu[labels_cpu == 0].numpy()
    neg = score_cpu[labels_cpu != 0].numpy()
    margin_pos = margin_cpu[labels_cpu == 0].numpy()
    margin_neg = margin_cpu[labels_cpu != 0].numpy()
    if len(pos) and len(neg):
        auroc = float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))
        score_margin = float(pos.mean() - neg.mean())
        margin_auroc = float(((margin_pos[:, None] > margin_neg[None, :]).mean() + 0.5 * (margin_pos[:, None] == margin_neg[None, :]).mean()))
        expert_margin_gap = float(margin_pos.mean() - margin_neg.mean())
    else:
        auroc = float("nan")
        score_margin = float("nan")
        margin_auroc = float("nan")
        expert_margin_gap = float("nan")
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "expert_vs_negative_auroc": auroc,
        "expert_margin_auroc": margin_auroc,
        "score_good_margin_mean": score_margin,
        "expert_margin_gap_mean": expert_margin_gap,
        "recall": {BOARD_CLASS_NAMES[i]: float(recalls[i]) for i in range(len(BOARD_CLASS_NAMES))},
        "precision": {BOARD_CLASS_NAMES[i]: float(precisions[i]) for i in range(len(BOARD_CLASS_NAMES))},
    }


def run_epoch(model, loader, optimizer, args, device: torch.device, train: bool) -> Dict[str, object]:
    model.train(train)
    all_logits, all_labels, all_score, all_margin = [], [], [], []
    losses, ce_losses, margin_losses, supcon_losses = [], [], [], []
    iterator = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in iterator:
        latent = batch["latent"].to(device, non_blocking=True).float()
        labels = batch["label"].to(device, non_blocking=True).long()
        with torch.set_grad_enabled(train):
            out = model(latent)
            ce = F.cross_entropy(out["logits"], labels)
            margin = expert_margin_loss(out["score_good"], labels, args.margin)
            supcon = supervised_contrastive_loss(out["embedding"], labels, args.supcon_temperature)
            loss = ce + args.margin_weight * margin + args.supcon_weight * supcon
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        ce_losses.append(float(ce.detach().cpu()))
        margin_losses.append(float(margin.detach().cpu()))
        supcon_losses.append(float(supcon.detach().cpu()))
        all_logits.append(out["logits"].detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_score.append(out["score_good"].detach().cpu())
        all_margin.append(out["expert_margin"].detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    score = torch.cat(all_score, dim=0)
    expert_margin = torch.cat(all_margin, dim=0)
    metrics = compute_metrics(logits, labels, score, expert_margin)
    metrics.update({
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else float("nan"),
        "margin_loss": float(np.mean(margin_losses)) if margin_losses else float("nan"),
        "supcon_loss": float(np.mean(supcon_losses)) if supcon_losses else float("nan"),
        "n": int(labels.numel()),
        "class_counts": class_counts(labels.tolist()),
    })
    return metrics


def save_checkpoint(
    path: Path,
    model,
    norm: Mapping[str, np.ndarray],
    vae_meta: Mapping[str, object],
    vae_identity: str,
    args,
    metrics: Mapping[str, object],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {
        "chunk_len": args.chunk_len,
        "latent_dim": int(vae_meta["latent_flat_dim"]),
        "embed_dim": args.embed_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "num_classes": len(BOARD_CLASS_NAMES),
    }
    payload = build_tactile_only_checkpoint(
        model,
        task=TASK,
        latent_mean=norm["latent_mean"],
        latent_std=norm["latent_std"],
        class_names=BOARD_CLASS_NAMES,
        temporal_stride=args.temporal_stride,
        future_offset=args.future_offset,
        vae_identity=vae_identity,
        model_config=model_config,
    )
    payload.update({
        "window_stride": args.stride,
        "vae_meta": dict(vae_meta),
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
    })
    torch.save(payload, path)


def run(args) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    vae, vae_info = load_tactile_vae_checkpoint(args.tactile_vae_ckpt, device)
    vae_meta = infer_vae_meta(vae_info)
    vae_identity = vae_checkpoint_identity(args.tactile_vae_ckpt)

    cfg = ChunkDatasetConfig(
        chunk_len=args.chunk_len,
        stride=args.stride,
        temporal_stride=args.temporal_stride,
        future_offset=args.future_offset,
        contact_only=not args.no_contact_only,
        contact_quantile=args.contact_quantile,
        min_contact_ratio=args.min_contact_ratio,
        max_episodes_per_class=args.max_episodes_per_class,
        seed=args.seed,
    )
    rows, audit = build_rows(cfg)
    train_rows, val_rows, split_meta = split_rows_by_episode(rows, args.val_ratio, args.seed)
    result: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "input_mode": INPUT_MODE,
        "task": TASK,
        "audit": audit,
        "split": split_meta,
        "class_names": list(BOARD_CLASS_NAMES),
        "output_dir": str(out_dir),
        "vae_checkpoint": args.tactile_vae_ckpt,
        "vae_meta": vae_meta,
        "vae_identity": vae_identity,
        "horizon": args.chunk_len,
        "latent_shape": [int(vae_meta["latent_flat_dim"])],
        "temporal_stride": args.temporal_stride,
        "future_offset": args.future_offset,
    }
    if args.audit_only:
        audit_path = out_dir / "board_latent_energy_audit.json"
        audit_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "audit_only": True,
            "total_windows": audit["total_windows"],
            "total_episodes": audit["total_episodes"],
            "vae_meta": vae_meta,
            "audit": str(audit_path),
        }, ensure_ascii=False, indent=2))
        return result

    norm = compute_normalization(train_rows, vae, vae_info, device, max_windows=args.norm_max_windows, seed=args.seed)
    manifest_path = out_dir / "manifest.json"
    write_manifest(
        manifest_path,
        rows=rows,
        train_rows=train_rows,
        val_rows=val_rows,
        audit=audit,
        split_meta=split_meta,
        norm=norm,
        vae_meta=vae_meta,
        vae_identity=vae_identity,
        horizon=args.chunk_len,
        latent_shape=[int(vae_meta["latent_flat_dim"])],
        window_stride=args.stride,
        temporal_stride=args.temporal_stride,
        future_offset=args.future_offset,
    )

    train_ds = BoardLatentChunkDataset(train_rows, norm, vae, vae_info, device, preload=not args.no_preload)
    val_ds = BoardLatentChunkDataset(val_rows, norm, vae, vae_info, device, preload=not args.no_preload)
    train_loader = make_loader(train_ds, args.batch_size, args.balanced_sampler, args.seed)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = BoardLatentEnergyScorer(
        chunk_len=args.chunk_len,
        latent_dim=int(vae_meta["latent_flat_dim"]),
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    best_metric = -float("inf")
    best_path = out_dir / "board_latent_energy_best.pt"
    last_path = out_dir / "board_latent_energy_last.pt"
    for epoch in range(args.epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, args, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, args, device, train=False)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        metric = float(val_metrics["expert_margin_auroc"]) + float(val_metrics["macro_f1"])
        if metric > best_metric:
            best_metric = metric
            save_checkpoint(best_path, model, norm, vae_meta, vae_identity, args, val_metrics, epoch)
        save_checkpoint(last_path, model, norm, vae_meta, vae_identity, args, val_metrics, epoch)
        print(json.dumps({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_expert_auroc": val_metrics["expert_vs_negative_auroc"],
            "val_margin_auroc": val_metrics["expert_margin_auroc"],
            "val_quality_gap": val_metrics["expert_margin_gap_mean"],
        }, ensure_ascii=False))

    result.update({
        "manifest": str(manifest_path),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history": history,
        "best_metric": best_metric,
    })
    metrics_path = out_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "done": True,
        "best_checkpoint": str(best_path),
        "metrics": str(metrics_path),
        "best_metric": best_metric,
    }, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--tactile_vae_ckpt", default=DEFAULT_TACTILE_VAE_CKPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--chunk_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--temporal_stride", type=int, default=1,
                        help="Within-window temporal stride. Use 3 for chunks start,start+3,... aligned to stride=3 DP.")
    parser.add_argument("--future_offset", type=int, default=1,
                        help="Non-negative offset from each index row to the first scored tactile latent.")
    parser.add_argument("--no_contact_only", action="store_true")
    parser.add_argument("--contact_quantile", type=float, default=0.50)
    parser.add_argument("--min_contact_ratio", type=float, default=0.25)
    parser.add_argument("--max_episodes_per_class", type=int, default=None)
    parser.add_argument("--norm_max_windows", type=int, default=4096)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--balanced_sampler", action="store_true", default=True)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--margin_weight", type=float, default=0.5)
    parser.add_argument("--supcon_weight", type=float, default=0.0)
    parser.add_argument("--supcon_temperature", type=float, default=0.10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
