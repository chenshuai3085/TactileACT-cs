"""Evaluate a trained board latent chunk energy scorer."""

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

from TFAC_V5.board_chunk_energy.labels import BOARD_CLASS_NAMES  # noqa: E402
from TFAC_V5.board_latent_energy.dataset import BoardLatentChunkDataset, load_manifest  # noqa: E402
from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime  # noqa: E402
from TFAC_V5.board_latent_energy.train import compute_metrics  # noqa: E402
from TFAC_V5.board_latent_energy.vae_utils import load_tactile_vae_checkpoint, vae_checkpoint_identity  # noqa: E402


def strict_preference_accuracy(positive: np.ndarray, negative: np.ndarray) -> Dict[str, object]:
    """Compute P(score_positive > score_negative) without materializing all pairs."""
    negative_sorted = np.sort(np.asarray(negative, dtype=np.float64))
    positive = np.asarray(positive, dtype=np.float64)
    correct = int(np.searchsorted(negative_sorted, positive, side="left").sum())
    pairs = int(positive.size * negative_sorted.size)
    return {
        "correct": correct,
        "pairs": pairs,
        "accuracy": float(correct / pairs) if pairs else float("nan"),
    }


def episode_bootstrap_preference(
    score: np.ndarray,
    labels: np.ndarray,
    paths,
    samples: int,
    seed: int,
) -> Dict[str, object]:
    by_episode = {}
    for value, label, path in zip(score, labels, paths):
        item = by_episode.setdefault(path, {"label": int(label), "score": []})
        item["score"].append(float(value))
    positive_ids = [key for key, value in by_episode.items() if value["label"] == 0]
    negative_ids = [key for key, value in by_episode.items() if value["label"] != 0]
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        sampled_positive = rng.choice(positive_ids, len(positive_ids), replace=True)
        sampled_negative = rng.choice(negative_ids, len(negative_ids), replace=True)
        positive = np.concatenate([by_episode[key]["score"] for key in sampled_positive])
        negative = np.concatenate([by_episode[key]["score"] for key in sampled_negative])
        estimates.append(strict_preference_accuracy(positive, negative)["accuracy"])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "bootstrap_unit": "episode",
        "samples": int(samples),
        "positive_episodes": len(positive_ids),
        "negative_episodes": len(negative_ids),
        "ci95": [float(low), float(high)],
    }


def confusion_matrix(labels: torch.Tensor, pred: torch.Tensor, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for y, p in zip(labels.cpu().numpy().tolist(), pred.cpu().numpy().tolist()):
        mat[int(y), int(p)] += 1
    return mat


def gradient_probe(runtime: BoardLatentEnergyRuntime, batch) -> Dict[str, object]:
    latent = batch["latent"].to(runtime.device).float().detach().clone().requires_grad_(True)
    out = runtime(latent, normalized=True)
    grad = torch.autograd.grad(out["expert_margin"].sum(), latent, retain_graph=False)[0]
    norms = grad.flatten(1).norm(dim=1)
    return {
        "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
        "positive_grad_rate": float((norms > 1e-8).float().mean().cpu()),
        "grad_norm_mean": float(norms.mean().detach().cpu()),
        "grad_norm_max": float(norms.max().detach().cpu()),
    }


def evaluate(args) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    manifest = load_manifest(Path(args.manifest))
    rows = manifest["val_rows"] if args.split == "val" else manifest["train_rows"]
    vae_ckpt = args.tactile_vae_ckpt
    if not vae_ckpt:
        raise ValueError("--tactile_vae_ckpt is required to verify the checkpoint VAE identity.")
    vae_identity = vae_checkpoint_identity(vae_ckpt)
    if manifest["vae_identity"] != vae_identity:
        raise ValueError(
            f"Manifest VAE mismatch: expected {manifest['vae_identity']!r}, got {vae_identity!r}"
        )
    runtime = BoardLatentEnergyRuntime(
        args.checkpoint,
        device=str(device),
        expected_horizon=int(manifest["horizon"]),
        expected_latent_shape=manifest["latent_shape"],
        expected_temporal_stride=int(manifest["temporal_stride"]),
        expected_future_offset=int(manifest["future_offset"]),
        expected_vae_identity=vae_identity,
    )
    vae, vae_info = load_tactile_vae_checkpoint(vae_ckpt, device)
    if int(vae_info["latent_flat_dim"]) != runtime.model.latent_dim:
        raise ValueError(
            f"VAE latent size {vae_info['latent_flat_dim']} does not match scorer latent size {runtime.model.latent_dim}"
        )
    for key, runtime_value in (("latent_mean", runtime.latent_mean), ("latent_std", runtime.latent_std)):
        manifest_value = torch.as_tensor(manifest["norm"][key], dtype=torch.float32).reshape(-1)
        if not torch.allclose(manifest_value, runtime_value.detach().cpu().reshape(-1), atol=1e-6, rtol=1e-6):
            raise ValueError(f"Manifest {key} does not match checkpoint normalization")
    dataset = BoardLatentChunkDataset(
        rows, manifest["norm"], vae, vae_info, device,
        include_path=True, preload=not args.no_preload,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_logits, all_labels, all_score, all_margin, all_quality, all_paths = [], [], [], [], [], []
    grad_summary = None
    with torch.no_grad():
        for batch in loader:
            latent = batch["latent"].to(device).float()
            labels = batch["label"].to(device).long()
            out = runtime(latent, normalized=True)
            all_logits.append(out["logits"].cpu())
            all_labels.append(labels.cpu())
            all_score.append(out["score_good"].cpu())
            all_margin.append(out["expert_margin"].cpu())
            all_quality.append(out["quality_0_100"].cpu())
            all_paths.extend(batch["path"])
    for batch in loader:
        grad_summary = gradient_probe(runtime, batch)
        break

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    score = torch.cat(all_score, dim=0)
    margin = torch.cat(all_margin, dim=0)
    quality = torch.cat(all_quality, dim=0)
    pred = logits.argmax(dim=1)
    metrics = compute_metrics(logits, labels, score, margin)
    cm = confusion_matrix(labels, pred, len(BOARD_CLASS_NAMES))
    positive = margin[labels == 0].numpy()
    negative = margin[labels != 0].numpy()
    preference = strict_preference_accuracy(positive, negative)
    preference.update(episode_bootstrap_preference(
        margin.numpy(), labels.numpy(), all_paths,
        samples=args.bootstrap_samples, seed=args.bootstrap_seed,
    ))

    score_by_class = {}
    for cls, name in enumerate(BOARD_CLASS_NAMES):
        mask = labels == cls
        values = score[mask].numpy()
        margins = margin[mask].numpy()
        qualities = quality[mask].numpy()
        score_by_class[name] = {
            "n": int(values.size),
            "score_good_mean": float(values.mean()) if values.size else None,
            "score_good_std": float(values.std()) if values.size else None,
            "expert_margin_mean": float(margins.mean()) if margins.size else None,
            "expert_margin_std": float(margins.std()) if margins.size else None,
            "quality_0_100_mean": float(qualities.mean()) if qualities.size else None,
            "quality_0_100_std": float(qualities.std()) if qualities.size else None,
            "quality_0_100_min": float(qualities.min()) if qualities.size else None,
            "quality_0_100_max": float(qualities.max()) if qualities.size else None,
        }

    result = {
        "checkpoint": args.checkpoint,
        "manifest": args.manifest,
        "vae_checkpoint": vae_ckpt,
        "vae_identity": vae_identity,
        "schema_version": runtime.schema_version,
        "input_mode": runtime.input_mode,
        "task": runtime.task,
        "horizon": runtime.horizon,
        "latent_shape": list(runtime.latent_shape),
        "temporal_stride": runtime.temporal_stride,
        "future_offset": runtime.future_offset,
        "split": args.split,
        "n": int(labels.numel()),
        "class_names": list(BOARD_CLASS_NAMES),
        "metrics": metrics,
        "preference_order": preference,
        "confusion_matrix": cm.tolist(),
        "score_by_class": score_by_class,
        "gradient_probe": grad_summary,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "split": args.split,
        "n": result["n"],
        "macro_f1": metrics["macro_f1"],
        "expert_margin_auroc": metrics["expert_margin_auroc"],
        "gradient_probe": grad_summary,
        "quality_0_100_mean": {k: v["quality_0_100_mean"] for k, v in score_by_class.items()},
        "output": str(out),
    }, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tactile_vae_ckpt", default=None)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--include_path", action="store_true")
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--output", default="/home/chenshuai/Project/output/board_latent_energy/eval_val.json")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
