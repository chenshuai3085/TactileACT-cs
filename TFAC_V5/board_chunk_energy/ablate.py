"""Input ablation evaluation for board chunk energy scorers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_chunk_energy.dataset import BoardChunkDataset, load_manifest  # noqa: E402
from TFAC_V5.board_chunk_energy.model import BoardChunkEnergyScorer  # noqa: E402
from TFAC_V5.board_chunk_energy.train import compute_metrics  # noqa: E402


def eval_mode(model, loader, device: torch.device, mode: str) -> Dict[str, object]:
    logits_all, labels_all, score_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            action = batch["action"].to(device).float()
            marker = batch["marker"].to(device).float()
            if mode == "zero_action":
                action = torch.zeros_like(action)
            elif mode == "zero_marker":
                marker = torch.zeros_like(marker)
            elif mode == "shuffle_action":
                action = action[torch.randperm(action.shape[0], device=action.device)]
            elif mode == "shuffle_marker":
                marker = marker[torch.randperm(marker.shape[0], device=marker.device)]
            elif mode != "full":
                raise ValueError(mode)
            labels = batch["label"].to(device).long()
            out = model(action, marker)
            logits_all.append(out["logits"].cpu())
            labels_all.append(labels.cpu())
            score_all.append(out["score_good"].cpu())
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    score = torch.cat(score_all, dim=0)
    return compute_metrics(logits, labels, score)


def run(args) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = BoardChunkEnergyScorer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    manifest = load_manifest(Path(args.manifest))
    rows = manifest["val_rows"] if args.split == "val" else manifest["train_rows"]
    dataset = BoardChunkDataset(rows, manifest["norm"], preload=not args.no_preload)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    result = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "split": args.split,
        "modes": {},
    }
    for mode in ["full", "zero_action", "zero_marker", "shuffle_action", "shuffle_marker"]:
        result["modes"][mode] = eval_mode(model, loader, device, mode)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(out),
        "macro_f1": {k: v["macro_f1"] for k, v in result["modes"].items()},
        "expert_auroc": {k: v["expert_vs_negative_auroc"] for k, v in result["modes"].items()},
    }, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--output", default="/home/chenshuai/Project/output/board_chunk_energy/ablate_val.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
