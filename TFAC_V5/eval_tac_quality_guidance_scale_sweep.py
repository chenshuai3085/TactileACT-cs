"""Cross-task local guidance-scale sweep for TacQuality classifier guidance.

The deployment question is not only whether a scorer classifies well.  For
classifier guidance inside DP denoising, the scorer must behave like a useful
local energy: a small step along d score / d action should improve the score,
the improvement should be monotonic over a practical scale range, and the
action update should remain bounded and smooth.

This script runs that local law on real insertion and board-wiping samples with
the final TacQualityGuidanceRuntime profiles.
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

from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep")


def summarize(x) -> Dict[str, float]:
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


def action_smoothness_np(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] < 3:
        return np.zeros(actions.shape[0], dtype=np.float32)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return np.linalg.norm(accel, axis=-1).mean(axis=1)


def action_smoothness_torch(actions: torch.Tensor) -> torch.Tensor:
    if actions.shape[1] < 3:
        return torch.zeros(actions.shape[0], dtype=actions.dtype, device=actions.device)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


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
    rng = np.random.default_rng(args.seed)
    selected: List[int] = []
    per_reason = max(1, args.n_insertion // max(1, len(np.unique(reason))))
    for cls in sorted(np.unique(reason).tolist()):
        idx = np.flatnonzero(reason == cls)
        if len(idx):
            selected.extend(rng.choice(idx, min(per_reason, len(idx)), replace=False).tolist())
    if len(selected) < args.n_insertion:
        rest = np.setdiff1d(np.arange(len(marker)), np.asarray(selected, dtype=np.int64), assume_unique=False)
        selected.extend(rng.choice(rest, min(args.n_insertion - len(selected), len(rest)), replace=False).tolist())
    idx = np.asarray(selected[: args.n_insertion], dtype=np.int64)
    return (
        marker[idx].astype(np.float32),
        action[idx].astype(np.float32),
        {
            "source": str(args.insertion_features),
            "n_samples": int(len(idx)),
            "reason_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(reason[idx], return_counts=True))},
            "quality": summarize(quality[idx]),
        },
    )


def board_rows(args):
    rows = []
    for path in sorted((Path(args.board_dir) / "success").glob("*.hdf5")):
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
    rng = np.random.default_rng(args.seed + 11)
    if len(rows) > args.n_board:
        chosen = rng.choice(len(rows), args.n_board, replace=False)
        rows = [rows[i] for i in sorted(chosen.tolist())]
    return rows


def sample_board(args):
    left, right, eef, joint = [], [], [], []
    rows = board_rows(args)
    if not rows:
        raise RuntimeError(f"No board windows found under {args.board_dir}")
    for path, start, end in rows:
        with h5py.File(path, "r") as f:
            left.append(pad_last(f["observations/tac/left/marker_offset"][start:end], args.board_window))
            right.append(pad_last(f["observations/tac/right/marker_offset"][start:end], args.board_window))
            eef.append(pad_last(f["actions/eef_abs"][start:end], args.board_window))
            joint.append(pad_last(f["actions/joint_abs"][start:end], args.board_window))
    return (
        np.stack(left).astype(np.float32),
        np.stack(right).astype(np.float32),
        np.stack(eef).astype(np.float32),
        np.stack(joint).astype(np.float32),
        {
            "source": str(args.board_dir),
            "n_samples": int(len(rows)),
            "window": int(args.board_window),
            "stride": int(args.board_stride),
        },
    )


def unit_grad(action: torch.Tensor, score: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad = torch.autograd.grad(score.sum(), action, retain_graph=False)[0]
    grad_norm = grad.flatten(1).norm(dim=1)
    grad_unit = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1))).clamp_min(1e-8)
    return grad_unit, grad_norm


def project(proposal: torch.Tensor, base: torch.Tensor, max_total_delta: float) -> torch.Tensor:
    if max_total_delta <= 0:
        return proposal
    delta = proposal - base
    norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.ndim - 1))).clamp_min(1e-8)
    scale = torch.clamp(max_total_delta / norm, max=1.0)
    return base + delta * scale


def eval_task(
    task: str,
    runtime: TacQualityGuidanceRuntime,
    score_fn,
    action_np: np.ndarray,
    scales: List[float],
    max_total_delta: float,
) -> Dict[str, object]:
    device = runtime.device
    base_np = np.asarray(action_np, dtype=np.float32)
    base_smooth = action_smoothness_np(base_np)
    base = torch.tensor(base_np, dtype=torch.float32, device=device, requires_grad=True)
    base_score = score_fn(base)
    grad_unit, grad_norm = unit_grad(base, base_score)
    rows = []
    prev_delta_mean = None
    trust_region_tol = max(1e-5, 1e-4 * max_total_delta)
    for scale in scales:
        with torch.no_grad():
            proposal = project(base.detach() + float(scale) * grad_unit.detach(), base.detach(), max_total_delta)
        proposal_req = proposal.detach().clone().requires_grad_(True)
        score_new = score_fn(proposal_req).detach()
        score_delta = (score_new - base_score.detach()).cpu().numpy()
        smooth_delta = action_smoothness_np(proposal.detach().cpu().numpy()) - base_smooth
        delta_norm = (proposal - base.detach()).flatten(1).norm(dim=1).detach().cpu().numpy()
        accept = score_delta > 0
        accepted_score_delta = np.where(accept, score_delta, 0.0)
        monotonic_vs_prev = True if prev_delta_mean is None else float(score_delta.mean()) >= prev_delta_mean - 1e-8
        prev_delta_mean = float(score_delta.mean())
        rows.append(
            {
                "scale": float(scale),
                "raw": {
                    "score_delta": summarize(score_delta),
                    "improved_rate": float(np.mean(score_delta > 0)),
                    "delta_norm": summarize(delta_norm),
                    "smoothness_delta": summarize(smooth_delta),
                    "within_trust_region": bool(delta_norm.max() <= max_total_delta + trust_region_tol),
                    "trust_region_tolerance": float(trust_region_tol),
                    "monotonic_score_delta_mean_vs_previous_scale": bool(monotonic_vs_prev),
                },
                "accept_only": {
                    "accept_rate": float(np.mean(accept)),
                    "score_delta": summarize(accepted_score_delta),
                    "smoothness_delta": summarize(np.where(accept, smooth_delta, 0.0)),
                },
            }
        )
    good_rows = [
        r
        for r in rows
        if r["raw"]["improved_rate"] >= 0.95
        and r["raw"]["within_trust_region"]
        and r["raw"]["smoothness_delta"]["p95"] <= max(1e-8, abs(r["raw"]["smoothness_delta"]["p50"]) + 10.0)
    ]
    best = max(good_rows or rows, key=lambda r: (r["raw"]["score_delta"]["mean"], -r["raw"]["delta_norm"]["mean"]))
    return {
        "task": task,
        "n_samples": int(base.shape[0]),
        "base_score": summarize(base_score.detach().cpu().numpy()),
        "base_smoothness": summarize(base_smooth),
        "gradient": {
            "finite_rate": float(torch.isfinite(grad_unit).flatten(1).all(dim=1).float().mean().detach().cpu()),
            "positive_norm_rate": float((grad_norm > 1e-8).float().mean().detach().cpu()),
            "grad_norm": summarize(grad_norm.detach().cpu().numpy()),
        },
        "max_total_delta": float(max_total_delta),
        "scale_rows": rows,
        "recommended_scale": float(best["scale"]),
        "recommended_score_delta_mean": float(best["raw"]["score_delta"]["mean"]),
        "recommended_improved_rate": float(best["raw"]["improved_rate"]),
        "passes_guidance_scale_sweep": bool(
            all(r["raw"]["within_trust_region"] for r in rows)
            and rows[0]["raw"]["improved_rate"] >= 0.95
            and best["raw"]["improved_rate"] >= 0.95
            and torch.isfinite(grad_unit).all().item()
            and float((grad_norm > 1e-8).float().mean().detach().cpu()) >= 0.999
        ),
    }


def write_markdown(result: Dict[str, object], path: Path) -> None:
    lines = [
        "# TacQuality Guidance Scale Sweep",
        "",
        "## Purpose",
        "",
        "Validate that TacQuality score behaves as a local DP classifier-guidance energy on real samples.",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- device: `{result['device']}`",
        "",
    ]
    for task in ["insertion", "board"]:
        item = result[task]
        lines.extend(
            [
                f"## {task}",
                "",
                f"- n_samples: `{item['n_samples']}`",
                f"- pass: `{item['passes_guidance_scale_sweep']}`",
                f"- recommended_scale: `{item['recommended_scale']}`",
                f"- recommended_score_delta_mean: `{item['recommended_score_delta_mean']}`",
                f"- recommended_improved_rate: `{item['recommended_improved_rate']}`",
                f"- grad_norm_mean: `{item['gradient']['grad_norm']['mean']}`",
                "",
                "| scale | improved | score_delta_mean | delta_norm_mean | smooth_delta_mean |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["scale_rows"]:
            lines.append(
                f"| {row['scale']} | {row['raw']['improved_rate']} | {row['raw']['score_delta']['mean']} | "
                f"{row['raw']['delta_norm']['mean']} | {row['raw']['smoothness_delta']['mean']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_scales(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n_insertion", type=int, default=256)
    parser.add_argument("--n_board", type=int, default=256)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--insertion_scales", default="0.005,0.01,0.02,0.04,0.08,0.12")
    parser.add_argument("--board_scales", default="0.00005,0.0001,0.0002,0.0004,0.0008,0.0016")
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    runtime = TacQualityGuidanceRuntime(device=args.device)

    ins_marker_np, ins_action_np, ins_meta = sample_insertion(args)
    ins_marker = torch.tensor(ins_marker_np, dtype=torch.float32, device=runtime.device)

    def ins_score(action):
        return runtime.score("insertion", ins_marker, action, mode="profile")

    board_left_np, board_right_np, board_eef_np, board_joint_np, board_meta = sample_board(args)
    board_left = torch.tensor(board_left_np, dtype=torch.float32, device=runtime.device)
    board_right = torch.tensor(board_right_np, dtype=torch.float32, device=runtime.device)
    board_eef = torch.tensor(board_eef_np, dtype=torch.float32, device=runtime.device)

    def board_score(action):
        return runtime.score(
            "board",
            board_left,
            action,
            right_marker_seq=board_right,
            eef_action_seq=board_eef,
            mode="profile",
        )

    insertion_profile = get_guidance_profile("insertion")
    board_profile = get_guidance_profile("board")
    insertion = eval_task(
        "insertion",
        runtime,
        ins_score,
        ins_action_np,
        parse_scales(args.insertion_scales),
        insertion_profile.refinement.max_total_delta,
    )
    insertion["meta"] = ins_meta
    board = eval_task(
        "board",
        runtime,
        board_score,
        board_joint_np,
        parse_scales(args.board_scales),
        board_profile.refinement.max_total_delta,
    )
    board["meta"] = board_meta
    result = {
        "purpose": "Cross-task real-sample local guidance-scale sweep for DP classifier guidance.",
        "scope": "Local score-gradient law on real samples; not a robot rollout.",
        "device": str(runtime.device),
        "seed": int(args.seed),
        "profiles": {
            "insertion": insertion_profile.to_dict(),
            "board": board_profile.to_dict(),
        },
        "insertion": insertion,
        "board": board,
    }
    result["overall_pass"] = bool(insertion["passes_guidance_scale_sweep"] and board["passes_guidance_scale_sweep"])
    json_path = args.output_dir / "tac_quality_guidance_scale_sweep.json"
    md_path = args.output_dir / "tac_quality_guidance_scale_sweep.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "insertion_recommended_scale": insertion["recommended_scale"],
                "insertion_delta": insertion["recommended_score_delta_mean"],
                "board_recommended_scale": board["recommended_scale"],
                "board_delta": board["recommended_score_delta_mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
