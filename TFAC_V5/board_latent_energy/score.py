"""Score HDF5 episodes with a trained board latent energy scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.board_latent_energy.dataset import DEFAULT_MARKER_KEY  # noqa: E402
from TFAC_V5.board_latent_energy.runtime import BoardLatentEnergyRuntime  # noqa: E402
from TFAC_V5.board_latent_energy.vae_utils import (  # noqa: E402
    encode_marker_chunk_to_latents,
    load_tactile_vae_checkpoint,
    vae_checkpoint_identity,
)


def list_hdf5_inputs(path: Path, recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    return sorted(path.glob(pattern))


def contact_filtered_starts(
    marker: np.ndarray,
    chunk_len: int,
    stride: int,
    temporal_stride: int,
    future_offset: int,
    contact_quantile: float,
    min_contact_ratio: float,
) -> List[int]:
    required_span = future_offset + (chunk_len - 1) * temporal_stride + 1
    starts = list(range(0, max(0, marker.shape[0] - required_span + 1), stride))
    if not starts:
        return []
    mag = np.linalg.norm(marker, axis=-1).mean(axis=(1, 2))
    threshold = max(float(np.quantile(mag, contact_quantile)), 1e-5)
    keep = []
    for start in starts:
        indices = start + future_offset + np.arange(chunk_len, dtype=np.int64) * temporal_stride
        window = mag[indices]
        if float(np.mean(window >= threshold)) >= min_contact_ratio:
            keep.append(start)
    return keep


def read_episode(path: Path, marker_key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if marker_key not in f:
            raise KeyError(f"{path}: missing marker key {marker_key!r}")
        marker = f[marker_key][:].astype(np.float32)
    return marker


def score_file(runtime: BoardLatentEnergyRuntime, vae, vae_info, path: Path, args) -> Dict[str, object]:
    marker = read_episode(path, args.marker_key)
    chunk_len = int(runtime.horizon)
    required_span = runtime.future_offset + (chunk_len - 1) * runtime.temporal_stride + 1
    if args.contact_only:
        starts = contact_filtered_starts(
            marker,
            chunk_len,
            args.stride,
            runtime.temporal_stride,
            runtime.future_offset,
            args.contact_quantile,
            args.min_contact_ratio,
        )
    else:
        starts = list(range(0, max(0, marker.shape[0] - required_span + 1), args.stride))

    rows: List[Dict[str, object]] = []
    device = runtime.device
    for start0 in range(0, len(starts), args.batch_size):
        batch_starts = starts[start0:start0 + args.batch_size]
        latent_batch = np.stack([
            encode_marker_chunk_to_latents(
                marker,
                s + runtime.future_offset,
                chunk_len,
                vae,
                vae_info,
                device,
                temporal_stride=runtime.temporal_stride,
            )
            for s in batch_starts
        ], axis=0)
        with torch.no_grad():
            out = runtime(torch.from_numpy(latent_batch), normalized=False)
        score_good = out["score_good"].detach().cpu().numpy()
        energy = out["energy"].detach().cpu().numpy()
        expert_margin = out["expert_margin"].detach().cpu().numpy()
        margin_energy = out["margin_energy"].detach().cpu().numpy()
        quality_0_100 = out["quality_0_100"].detach().cpu().numpy()
        prob = out["prob"].detach().cpu().numpy()
        pred = prob.argmax(axis=1)
        for i, start in enumerate(batch_starts):
            rows.append({
                "path": str(path),
                "start": int(start),
                "end": int(start + required_span),
                "score_good": float(score_good[i]),
                "energy": float(energy[i]),
                "expert_margin": float(expert_margin[i]),
                "margin_energy": float(margin_energy[i]),
                "quality_0_100": float(quality_0_100[i]),
                "pred_class": runtime.class_names[int(pred[i])],
                "prob": {
                    name: float(prob[i, cls])
                    for cls, name in enumerate(runtime.class_names)
                },
            })

    if rows:
        scores = np.asarray([row["score_good"] for row in rows], dtype=np.float32)
        energies = np.asarray([row["energy"] for row in rows], dtype=np.float32)
        margins = np.asarray([row["expert_margin"] for row in rows], dtype=np.float32)
        qualities = np.asarray([row["quality_0_100"] for row in rows], dtype=np.float32)
        pred_counts: Dict[str, int] = {}
        for row in rows:
            pred_counts[row["pred_class"]] = pred_counts.get(row["pred_class"], 0) + 1
        summary = {
            "path": str(path),
            "windows": len(rows),
            "score_good_mean": float(scores.mean()),
            "score_good_min": float(scores.min()),
            "score_good_max": float(scores.max()),
            "energy_mean": float(energies.mean()),
            "expert_margin_mean": float(margins.mean()),
            "expert_margin_min": float(margins.min()),
            "expert_margin_max": float(margins.max()),
            "quality_0_100_mean": float(qualities.mean()),
            "quality_0_100_min": float(qualities.min()),
            "quality_0_100_max": float(qualities.max()),
            "pred_counts": pred_counts,
        }
    else:
        summary = {
            "path": str(path),
            "windows": 0,
            "score_good_mean": None,
            "score_good_min": None,
            "score_good_max": None,
            "energy_mean": None,
            "expert_margin_mean": None,
            "expert_margin_min": None,
            "expert_margin_max": None,
            "quality_0_100_mean": None,
            "quality_0_100_min": None,
            "quality_0_100_max": None,
            "pred_counts": {},
        }
    return {"summary": summary, "rows": rows}


def flatten(items: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in items:
        rows.extend(item["rows"])
    return rows


def run(args) -> Dict[str, object]:
    vae_ckpt = args.tactile_vae_ckpt
    if vae_ckpt is None:
        raise ValueError("--tactile_vae_ckpt is required to verify the checkpoint VAE identity.")
    vae_identity = vae_checkpoint_identity(vae_ckpt)
    runtime = BoardLatentEnergyRuntime(
        args.checkpoint,
        device=args.device,
        expected_vae_identity=vae_identity,
    )
    vae, vae_info = load_tactile_vae_checkpoint(vae_ckpt, runtime.device)
    if int(vae_info["latent_flat_dim"]) != runtime.model.latent_dim:
        raise ValueError(
            f"VAE latent size {vae_info['latent_flat_dim']} does not match scorer latent size {runtime.model.latent_dim}"
        )
    paths = list_hdf5_inputs(Path(args.input), args.recursive)
    if not paths:
        raise FileNotFoundError(f"No .hdf5 files found under {args.input}")
    per_file = [score_file(runtime, vae, vae_info, path, args) for path in paths]
    rows = flatten(per_file)

    if rows:
        scores = np.asarray([row["score_good"] for row in rows], dtype=np.float32)
        energies = np.asarray([row["energy"] for row in rows], dtype=np.float32)
        margins = np.asarray([row["expert_margin"] for row in rows], dtype=np.float32)
        qualities = np.asarray([row["quality_0_100"] for row in rows], dtype=np.float32)
        pred_counts: Dict[str, int] = {}
        for row in rows:
            pred_counts[row["pred_class"]] = pred_counts.get(row["pred_class"], 0) + 1
        overall = {
            "files": len(paths),
            "windows": len(rows),
            "score_good_mean": float(scores.mean()),
            "score_good_min": float(scores.min()),
            "score_good_max": float(scores.max()),
            "energy_mean": float(energies.mean()),
            "expert_margin_mean": float(margins.mean()),
            "expert_margin_min": float(margins.min()),
            "expert_margin_max": float(margins.max()),
            "quality_0_100_mean": float(qualities.mean()),
            "quality_0_100_min": float(qualities.min()),
            "quality_0_100_max": float(qualities.max()),
            "pred_counts": pred_counts,
        }
    else:
        overall = {
            "files": len(paths),
            "windows": 0,
            "score_good_mean": None,
            "score_good_min": None,
            "score_good_max": None,
            "energy_mean": None,
            "expert_margin_mean": None,
            "expert_margin_min": None,
            "expert_margin_max": None,
            "quality_0_100_mean": None,
            "quality_0_100_min": None,
            "quality_0_100_max": None,
            "pred_counts": {},
        }

    result = {
        "checkpoint": args.checkpoint,
        "tactile_vae_ckpt": vae_ckpt,
        "vae_identity": vae_identity,
        "input": args.input,
        "schema_version": runtime.schema_version,
        "input_mode": runtime.input_mode,
        "task": runtime.task,
        "horizon": runtime.horizon,
        "latent_shape": list(runtime.latent_shape),
        "temporal_stride": runtime.temporal_stride,
        "future_offset": runtime.future_offset,
        "stride": args.stride,
        "class_names": runtime.class_names,
        "overall": overall,
        "files": [item["summary"] for item in per_file],
        "rows": rows if args.include_rows else [],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "files": overall["files"],
        "windows": overall["windows"],
        "expert_margin_mean": overall["expert_margin_mean"],
        "quality_0_100_mean": overall["quality_0_100_mean"],
        "pred_counts": overall["pred_counts"],
    }, ensure_ascii=False, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tactile_vae_ckpt", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="/home/chenshuai/Project/output/board_latent_energy/score.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--marker_key", default=DEFAULT_MARKER_KEY)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--contact_only", action="store_true", default=True)
    parser.add_argument("--no_contact_only", dest="contact_only", action="store_false")
    parser.add_argument("--contact_quantile", type=float, default=0.50)
    parser.add_argument("--min_contact_ratio", type=float, default=0.25)
    parser.add_argument("--include_rows", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
