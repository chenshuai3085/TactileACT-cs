"""Real-sample smoke test for the TacQuality deployment manifest.

This script checks the final DP-guidance-facing contract on real cached
insertion windows and real board-wiping HDF5 windows:

  runtime.score(task, real_tactile, real_action, mode="profile")
  refiner.refine(real_action, score_fn)

It is not a robot rollout and it does not replace full-chain Foresight/DP
validation.  It verifies that the unified scorer/refiner API produces finite
gradients, improves its own quality energy, and respects the task trust region
on actual data tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402
from TFAC_V5.tac_quality_trust_region_guidance import from_guidance_profile  # noqa: E402


DEFAULT_MANIFEST = Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json")
INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_manifest_real_sample_smoke")


def summarize_array(x) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def pad_last(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= window:
        return x[-window:]
    pad = np.repeat(x[:1], window - len(x), axis=0)
    return np.concatenate([pad, x], axis=0)


def sample_insertion(args) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    data = np.load(args.insertion_features, allow_pickle=True)
    marker = data["marker"]
    action = data["action"]
    reason = data["reason"]
    quality = data["quality"]
    groups = data["groups"]
    sample_ids = data["sample_ids"]

    rng = np.random.default_rng(args.seed)
    candidate = np.arange(len(marker))
    if args.insertion_include_all_reasons:
        selected = []
        per_reason = max(1, args.n_insertion // max(1, len(np.unique(reason))))
        for cls in sorted(np.unique(reason).tolist()):
            idx = np.flatnonzero(reason == cls)
            if len(idx):
                selected.extend(rng.choice(idx, min(per_reason, len(idx)), replace=False).tolist())
        if len(selected) < args.n_insertion:
            rest = np.setdiff1d(candidate, np.asarray(selected, dtype=np.int64), assume_unique=False)
            fill = rng.choice(rest, min(args.n_insertion - len(selected), len(rest)), replace=False)
            selected.extend(fill.tolist())
        idx = np.asarray(selected[: args.n_insertion], dtype=np.int64)
    else:
        idx = rng.choice(candidate, min(args.n_insertion, len(candidate)), replace=False)
    idx = np.sort(idx)
    meta = {
        "source": str(args.insertion_features),
        "sample_indices": idx[: min(20, len(idx))].astype(int).tolist(),
        "n_samples": int(len(idx)),
        "reason_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(reason[idx], return_counts=True))},
        "quality": summarize_array(quality[idx]),
        "groups_preview": [str(x) for x in groups[idx[: min(10, len(idx))]]],
        "sample_ids_preview": [str(x) for x in sample_ids[idx[: min(10, len(idx))]]],
    }
    return marker[idx].astype(np.float32), action[idx].astype(np.float32), meta


def board_rows(args) -> List[Tuple[Path, int, int]]:
    files = sorted((Path(args.board_dir) / "success").glob("*.hdf5"))
    rows: List[Tuple[Path, int, int]] = []
    for path in files:
        with h5py.File(path, "r") as f:
            n = min(
                len(f["observations/tac/left/marker_offset"]),
                len(f["observations/tac/right/marker_offset"]),
                len(f["actions/eef_abs"]),
                len(f["actions/joint_abs"]),
            )
        for start in range(0, max(1, n - args.board_window + 1), args.board_stride):
            end = min(n, start + args.board_window)
            if end - start >= max(8, args.board_window // 2):
                rows.append((path, start, end))
    rng = np.random.default_rng(args.seed + 17)
    if args.n_board and len(rows) > args.n_board:
        chosen = rng.choice(len(rows), args.n_board, replace=False)
        rows = [rows[i] for i in sorted(chosen.tolist())]
    return rows


def read_board_row(path: Path, start: int, end: int, window: int):
    with h5py.File(path, "r") as f:
        left = pad_last(f["observations/tac/left/marker_offset"][start:end], window)
        right = pad_last(f["observations/tac/right/marker_offset"][start:end], window)
        eef = pad_last(f["actions/eef_abs"][start:end], window)
        joint = pad_last(f["actions/joint_abs"][start:end], window)
    return left, right, eef, joint


def sample_board(args):
    rows = board_rows(args)
    if not rows:
        raise RuntimeError(f"No board windows found under {args.board_dir}")
    left, right, eef, joint = [], [], [], []
    for path, start, end in rows:
        l, r, e, j = read_board_row(path, start, end, args.board_window)
        left.append(l)
        right.append(r)
        eef.append(e)
        joint.append(j)
    meta = {
        "source": str(args.board_dir),
        "n_samples": int(len(rows)),
        "window": int(args.board_window),
        "stride": int(args.board_stride),
        "rows_preview": [
            {"episode": row[0].stem, "start": int(row[1]), "end": int(row[2])}
            for row in rows[: min(10, len(rows))]
        ],
    }
    return (
        np.stack(left).astype(np.float32),
        np.stack(right).astype(np.float32),
        np.stack(eef).astype(np.float32),
        np.stack(joint).astype(np.float32),
        meta,
    )


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def gradient_probe(score: torch.Tensor, action: torch.Tensor) -> Dict[str, object]:
    grad = torch.autograd.grad(score.sum(), action, retain_graph=False)[0]
    flat = grad.flatten(1)
    norms = tensor_to_np(flat.norm(dim=1))
    finite = tensor_to_np(torch.isfinite(flat).all(dim=1)).astype(bool)
    return {
        "grad_norm": summarize_array(norms),
        "finite_grad_rate": float(np.mean(finite)),
        "positive_grad_rate": float(np.mean(norms > 1e-8)),
    }


def run_insertion(args, runtime: TacQualityGuidanceRuntime) -> Dict[str, object]:
    marker_np, action_np, meta = sample_insertion(args)
    device = runtime.device
    marker = torch.tensor(marker_np, dtype=torch.float32, device=device)
    action = torch.tensor(action_np, dtype=torch.float32, device=device, requires_grad=True)
    base_score = runtime.score("insertion", marker, action, mode="profile")
    grad = gradient_probe(base_score, action)

    refiner = from_guidance_profile("insertion", clamp_norm_action=False)

    def score_fn(a: torch.Tensor) -> torch.Tensor:
        return runtime.score("insertion", marker, a, mode="profile")

    refined, report = refiner.refine(action.detach(), score_fn)
    final_score = score_fn(refined).detach()
    delta = tensor_to_np(final_score - base_score.detach())
    delta_norm = tensor_to_np((refined - action.detach()).flatten(1).norm(dim=1))
    result = {
        "task": "insertion",
        "meta": meta,
        "base_score": summarize_array(tensor_to_np(base_score)),
        "final_score": summarize_array(tensor_to_np(final_score)),
        "score_delta": summarize_array(delta),
        "score_improved_rate": float(np.mean(delta > 0)),
        "delta_norm": summarize_array(delta_norm),
        "gradient_probe": grad,
        "trust_region_report": report,
        "passes_real_sample_smoke": bool(
            np.isfinite(delta).all()
            and grad["finite_grad_rate"] >= 0.999
            and grad["positive_grad_rate"] >= 0.999
            and float(np.mean(delta > 0)) >= args.pass_improve_rate
            and bool(report["max_delta_within_trust_region"])
        ),
    }
    return result


def run_board(args, runtime: TacQualityGuidanceRuntime) -> Dict[str, object]:
    left_np, right_np, eef_np, joint_np, meta = sample_board(args)
    device = runtime.device
    left = torch.tensor(left_np, dtype=torch.float32, device=device)
    right = torch.tensor(right_np, dtype=torch.float32, device=device)
    eef = torch.tensor(eef_np, dtype=torch.float32, device=device)
    joint = torch.tensor(joint_np, dtype=torch.float32, device=device, requires_grad=True)
    base_score = runtime.score(
        "board",
        left,
        joint,
        right_marker_seq=right,
        eef_action_seq=eef,
        mode="profile",
    )
    grad = gradient_probe(base_score, joint)

    refiner = from_guidance_profile("board", clamp_norm_action=False)

    def score_fn(a: torch.Tensor) -> torch.Tensor:
        return runtime.score(
            "board",
            left,
            a,
            right_marker_seq=right,
            eef_action_seq=eef,
            mode="profile",
        )

    refined, report = refiner.refine(joint.detach(), score_fn)
    final_score = score_fn(refined).detach()
    delta = tensor_to_np(final_score - base_score.detach())
    delta_norm = tensor_to_np((refined - joint.detach()).flatten(1).norm(dim=1))
    result = {
        "task": "board",
        "meta": meta,
        "base_score": summarize_array(tensor_to_np(base_score)),
        "final_score": summarize_array(tensor_to_np(final_score)),
        "score_delta": summarize_array(delta),
        "score_improved_rate": float(np.mean(delta > 0)),
        "delta_norm": summarize_array(delta_norm),
        "gradient_probe": grad,
        "trust_region_report": report,
        "passes_real_sample_smoke": bool(
            np.isfinite(delta).all()
            and grad["finite_grad_rate"] >= 0.999
            and grad["positive_grad_rate"] >= 0.999
            and float(np.mean(delta > 0)) >= args.pass_improve_rate
            and bool(report["max_delta_within_trust_region"])
        ),
    }
    return result


def write_markdown(result: Dict[str, object], path: Path) -> None:
    ins = result["insertion"]
    board = result["board"]
    lines = [
        "# TacQuality Manifest Real-Sample Smoke Test",
        "",
        "## Scope",
        "",
        "This test uses real insertion feature-cache windows and real board-wiping HDF5 windows.",
        "It validates the final unified score/refine API on real tensors; it is not a robot rollout.",
        "",
        "## Result",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- insertion_pass: `{ins['passes_real_sample_smoke']}`",
        f"- board_pass: `{board['passes_real_sample_smoke']}`",
        "",
        "## Insertion",
        "",
        f"- n_samples: `{ins['meta']['n_samples']}`",
        f"- reason_counts: `{ins['meta']['reason_counts']}`",
        f"- score_improved_rate: `{ins['score_improved_rate']}`",
        f"- score_delta_mean: `{ins['score_delta']['mean']}`",
        f"- delta_norm_max: `{ins['delta_norm']['max']}`",
        f"- action_grad_norm_mean: `{ins['gradient_probe']['grad_norm']['mean']}`",
        "",
        "## Board",
        "",
        f"- n_samples: `{board['meta']['n_samples']}`",
        f"- score_improved_rate: `{board['score_improved_rate']}`",
        f"- score_delta_mean: `{board['score_delta']['mean']}`",
        f"- delta_norm_max: `{board['delta_norm']['max']}`",
        f"- joint_grad_norm_mean: `{board['gradient_probe']['grad_norm']['mean']}`",
        "",
        "## Files",
        "",
        f"- manifest: `{result['manifest_path']}`",
        f"- json: `{result['output_json']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_insertion", type=int, default=256)
    parser.add_argument("--n_board", type=int, default=256)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--pass_improve_rate", type=float, default=0.95)
    parser.add_argument("--insertion_include_all_reasons", action="store_true", default=True)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runtime = TacQualityGuidanceRuntime(device=args.device)

    insertion = run_insertion(args, runtime)
    board = run_board(args, runtime)
    result = {
        "purpose": "Manifest-driven real-sample smoke test for TacQuality gradient guidance API.",
        "scope": "Real data tensor API and local trust-region gradient check; not real-robot validation.",
        "manifest_path": str(args.manifest),
        "manifest_pass": bool(manifest.get("deployment_manifest_pass", False)),
        "score_api": manifest.get("score_api", {}),
        "device": str(runtime.device),
        "seed": int(args.seed),
        "config": {
            "n_insertion": int(args.n_insertion),
            "n_board": int(args.n_board),
            "board_window": int(args.board_window),
            "board_stride": int(args.board_stride),
            "pass_improve_rate": float(args.pass_improve_rate),
        },
        "insertion": insertion,
        "board": board,
    }
    result["overall_pass"] = bool(
        result["manifest_pass"]
        and insertion["passes_real_sample_smoke"]
        and board["passes_real_sample_smoke"]
    )

    json_path = args.output_dir / "manifest_real_sample_smoke.json"
    md_path = args.output_dir / "manifest_real_sample_smoke.md"
    result["output_json"] = str(json_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({k: result[k] for k in ["overall_pass", "manifest_pass", "device", "config"]}, ensure_ascii=False, indent=2))
    print(f"insertion improved={insertion['score_improved_rate']:.4f}, delta={insertion['score_delta']['mean']:.6f}")
    print(f"board improved={board['score_improved_rate']:.4f}, delta={board['score_delta']['mean']:.6f}")
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
