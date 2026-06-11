"""Evaluate a trained board chunk energy scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_chunk_energy.dataset import BoardChunkDataset, load_manifest  # noqa: E402
from TFAC_V5.board_chunk_energy.labels import BOARD_CLASS_NAMES  # noqa: E402
from TFAC_V5.board_chunk_energy.model import BoardChunkEnergyScorer  # noqa: E402
from TFAC_V5.board_chunk_energy.train import compute_metrics  # noqa: E402


def confusion_matrix(labels: torch.Tensor, pred: torch.Tensor, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y, p in zip(labels.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
        mat[int(y), int(p)] += 1
    return mat


def gradient_probe(model, batch, device: torch.device) -> Dict[str, object]:
    action = batch["action"].to(device).float().detach().clone().requires_grad_(True)
    marker = batch["marker"].to(device).float()
    out = model(action, marker)
    grad = torch.autograd.grad(out["score_good"].sum(), action, retain_graph=False)[0]
    norms = grad.flatten(1).norm(dim=1)
    return {
        "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
        "positive_grad_rate": float((norms > 1e-8).float().mean().cpu()),
        "grad_norm_mean": float(norms.mean().detach().cpu()),
        "grad_norm_max": float(norms.max().detach().cpu()),
    }


def evaluate(args) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BoardChunkEnergyScorer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    manifest = load_manifest(Path(args.manifest))
    rows = manifest["val_rows"] if args.split == "val" else manifest["train_rows"]
    dataset = BoardChunkDataset(rows, manifest["norm"], include_path=args.include_path, preload=not args.no_preload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_logits, all_labels, all_score = [], [], []
    grad_summary = None
    with torch.no_grad():
        for batch in loader:
            action = batch["action"].to(device).float()
            marker = batch["marker"].to(device).float()
            labels = batch["label"].to(device).long()
            out = model(action, marker)
            all_logits.append(out["logits"].cpu())
            all_labels.append(labels.cpu())
            all_score.append(out["score_good"].cpu())
    for batch in loader:
        grad_summary = gradient_probe(model, batch, device)
        break

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    score = torch.cat(all_score, dim=0)
    pred = logits.argmax(dim=1)
    metrics = compute_metrics(logits, labels, score)
    cm = confusion_matrix(labels, pred, len(BOARD_CLASS_NAMES))

    score_by_class = {}
    for cls, name in enumerate(BOARD_CLASS_NAMES):
        values = score[labels == cls].numpy()
        score_by_class[name] = {
            "n": int(values.size),
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }

    result = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "split": args.split,
        "n": int(labels.numel()),
        "class_names": list(BOARD_CLASS_NAMES),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "score_good_by_class": score_by_class,
        "gradient_probe": grad_summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "n": result["n"],
        "macro_f1": metrics["macro_f1"],
        "expert_vs_negative_auroc": metrics["expert_vs_negative_auroc"],
        "gradient_probe": grad_summary,
        "output": str(out),
    }, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--include_path", action="store_true")
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--output", default="/home/chenshuai/Project/output/board_chunk_energy/eval_val.json")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
