"""Evaluate TacQuality score landscape for DP classifier guidance.

Classification accuracy alone is not enough for gradient guidance.  The score
must also be a usable local energy: gradients should be finite, non-zero, not
dominated by saturation, and consistent with finite differences in the action
space that DP/Foresight will optimize through.

This diagnostic uses real insertion and board-wiping windows and checks local
landscape properties around recorded actions:

  - action-gradient norm distribution;
  - positive/negative directional finite-difference response;
  - finite-difference vs autograd directional derivative agreement;
  - local curvature / nonlinearity over trust-region-sized perturbations.
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
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_score_landscape")


def summarize(x) -> Dict[str, float]:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


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
    rng = np.random.default_rng(args.seed + 7)
    if len(rows) > args.n_board:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_board, replace=False).tolist())]
    return rows


def sample_board(args):
    rows = board_rows(args)
    if not rows:
        raise RuntimeError(f"No board windows found under {args.board_dir}")
    left, right, eef, joint = [], [], [], []
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


def score_and_grad(score_fn, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    action_req = action.detach().clone().requires_grad_(True)
    score = score_fn(action_req)
    grad = torch.autograd.grad(score.sum(), action_req, retain_graph=False)[0]
    return score.detach(), grad.detach()


def project_delta(delta: torch.Tensor, radius: float) -> torch.Tensor:
    if radius <= 0:
        return delta
    norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.ndim - 1))).clamp_min(1e-8)
    return delta * torch.clamp(float(radius) / norm, max=1.0)


def unit_direction(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    direction = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    norm = direction.flatten(1).norm(dim=1).view(-1, *([1] * (direction.ndim - 1))).clamp_min(1e-8)
    return direction / norm


def landscape_task(
    task: str,
    score_fn,
    action_np: np.ndarray,
    *,
    trust_radius: float,
    eps_list: List[float],
    random_radius_frac: float,
    min_positive_rate: float,
    min_negative_rate: float,
    min_fd_corr: float,
    max_fd_rel_error_p95: float,
    min_random_corr: float,
    max_random_rel_residual_p95: float,
    seed: int,
) -> Dict[str, object]:
    runtime_device = getattr(score_fn, "runtime_device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    action = torch.tensor(action_np, dtype=torch.float32)
    action = action.to(runtime_device)
    base_score, grad = score_and_grad(score_fn, action)
    grad_norm = grad.flatten(1).norm(dim=1)
    grad_unit = grad / grad_norm.view(-1, *([1] * (grad.ndim - 1))).clamp_min(1e-8)
    eps_results = []
    for eps in eps_list:
        radius = min(float(eps), float(trust_radius))
        pos = action + project_delta(radius * grad_unit, trust_radius)
        neg = action - project_delta(radius * grad_unit, trust_radius)
        score_pos = score_fn(pos.detach().clone().requires_grad_(True)).detach()
        score_neg = score_fn(neg.detach().clone().requires_grad_(True)).detach()
        pos_delta = (score_pos - base_score).cpu().numpy()
        neg_delta = (score_neg - base_score).cpu().numpy()
        central_fd = ((score_pos - score_neg) / (2.0 * radius)).cpu().numpy()
        expected = grad_norm.cpu().numpy()
        rel_error = np.abs(central_fd - expected) / np.maximum(np.abs(expected), 1e-8)
        curvature = ((score_pos + score_neg - 2.0 * base_score) / max(radius * radius, 1e-12)).cpu().numpy()
        eps_results.append(
            {
                "eps": float(radius),
                "positive_direction_delta": summarize(pos_delta),
                "negative_direction_delta": summarize(neg_delta),
                "positive_improved_rate": float(np.mean(pos_delta > 0.0)),
                "negative_worsened_rate": float(np.mean(neg_delta < 0.0)),
                "central_fd_vs_grad_norm_corr": corr(central_fd, expected),
                "central_fd_relative_error": summarize(rel_error),
                "curvature": summarize(curvature),
            }
        )

    gen = torch.Generator(device=action.device)
    gen.manual_seed(seed)
    random_dir = unit_direction(action, gen)
    random_delta = project_delta(float(random_radius_frac) * float(trust_radius) * random_dir, trust_radius)
    score_random = score_fn((action + random_delta).detach().clone().requires_grad_(True)).detach()
    first_order = (grad.flatten(1) * random_delta.flatten(1)).sum(dim=1)
    actual = score_random - base_score
    residual = (actual - first_order).cpu().numpy()
    rel_residual = np.abs(residual) / np.maximum(np.abs(actual.detach().cpu().numpy()), 1e-8)
    base_np = base_score.cpu().numpy()
    grad_np = grad_norm.cpu().numpy()
    pass_eps = all(
        row["positive_improved_rate"] >= min_positive_rate
        and row["negative_worsened_rate"] >= min_negative_rate
        and row["central_fd_vs_grad_norm_corr"] >= min_fd_corr
        and row["central_fd_relative_error"]["p95"] <= max_fd_rel_error_p95
        for row in eps_results
    )
    grad_finite = torch.isfinite(grad).flatten(1).all(dim=1).detach().cpu().numpy()
    positive_norm_rate = float(np.mean(grad_np > 1e-8))
    finite_rate = float(np.mean(grad_finite))
    saturation_rate = float(np.mean(grad_np < 1e-5))
    explosion_rate = float(np.mean(grad_np > 1e3))
    return {
        "task": task,
        "n_samples": int(action.shape[0]),
        "trust_radius": float(trust_radius),
        "base_score": summarize(base_np),
        "gradient": {
            "finite_rate": finite_rate,
            "positive_norm_rate": positive_norm_rate,
            "saturation_rate_grad_norm_lt_1e-5": saturation_rate,
            "explosion_rate_grad_norm_gt_1e3": explosion_rate,
            "grad_norm": summarize(grad_np),
        },
        "directional_eps": eps_results,
        "random_direction": {
            "radius": float(random_radius_frac * trust_radius),
            "actual_delta": summarize(actual.cpu().numpy()),
            "first_order_delta": summarize(first_order.cpu().numpy()),
            "first_order_vs_actual_corr": corr(first_order.cpu().numpy(), actual.cpu().numpy()),
            "relative_residual": summarize(rel_residual),
        },
        "passes_score_landscape": bool(
            finite_rate >= 0.999
            and positive_norm_rate >= 0.999
            and saturation_rate <= 0.05
            and explosion_rate == 0.0
            and pass_eps
            and all(row["positive_improved_rate"] >= min_positive_rate for row in eps_results)
            and corr(first_order.cpu().numpy(), actual.cpu().numpy()) >= min_random_corr
            and summarize(rel_residual)["p95"] <= max_random_rel_residual_p95
        ),
        "pass_thresholds": {
            "min_positive_rate": float(min_positive_rate),
            "min_negative_rate": float(min_negative_rate),
            "min_fd_corr": float(min_fd_corr),
            "max_fd_rel_error_p95": float(max_fd_rel_error_p95),
            "min_random_corr": float(min_random_corr),
            "max_random_rel_residual_p95": float(max_random_rel_residual_p95),
            "max_saturation_rate": 0.05,
            "max_explosion_rate": 0.0,
        },
    }


def make_score_fn(runtime: TacQualityGuidanceRuntime, task: str, **fixed):
    if task == "insertion":
        marker = fixed["marker"]

        def score(action):
            return runtime.score("insertion", marker, action, mode="profile")

    elif task == "board":
        left = fixed["left"]
        right = fixed["right"]
        eef = fixed["eef"]

        def score(action):
            return runtime.score(
                "board",
                left,
                action,
                right_marker_seq=right,
                eef_action_seq=eef,
                mode="profile",
            )

    else:
        raise KeyError(task)
    score.runtime_device = runtime.device
    return score


def write_markdown(result: Dict[str, object], path: Path) -> None:
    lines = [
        "# TacQuality Score Landscape Diagnostic",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- device: `{result['device']}`",
        "",
    ]
    for task in ["insertion", "board"]:
        row = result[task]
        lines.extend(
            [
                f"## {task}",
                "",
                f"- pass: `{row['passes_score_landscape']}`",
                f"- n_samples: `{row['n_samples']}`",
                f"- trust_radius: `{row['trust_radius']}`",
                f"- grad_norm_mean: `{row['gradient']['grad_norm']['mean']}`",
                f"- saturation_rate: `{row['gradient']['saturation_rate_grad_norm_lt_1e-5']}`",
                "",
                "| eps | + improve | - worsen | fd/grad corr | fd rel err p95 | curvature p95 |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in row["directional_eps"]:
            lines.append(
                f"| {item['eps']} | {item['positive_improved_rate']} | {item['negative_worsened_rate']} | "
                f"{item['central_fd_vs_grad_norm_corr']} | {item['central_fd_relative_error']['p95']} | "
                f"{item['curvature']['p95']} |"
            )
        lines.extend(
            [
                "",
                f"- random first_order_vs_actual_corr: `{row['random_direction']['first_order_vs_actual_corr']}`",
                f"- random relative_residual_p95: `{row['random_direction']['relative_residual']['p95']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_eps(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--n_insertion", type=int, default=256)
    parser.add_argument("--n_board", type=int, default=256)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--insertion_eps", default="0.0025,0.005,0.01,0.02")
    parser.add_argument("--board_eps", default="0.000025,0.00005,0.0001,0.0002")
    parser.add_argument("--random_radius_frac", type=float, default=0.25)
    parser.add_argument("--min_positive_rate", type=float, default=0.98)
    parser.add_argument("--min_negative_rate", type=float, default=0.95)
    parser.add_argument("--min_fd_corr", type=float, default=0.95)
    parser.add_argument("--max_fd_rel_error_p95", type=float, default=0.55)
    parser.add_argument("--min_random_corr", type=float, default=0.80)
    parser.add_argument("--max_random_rel_residual_p95", type=float, default=4.0)
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
    ins_score = make_score_fn(runtime, "insertion", marker=ins_marker)
    ins_profile = get_guidance_profile("insertion")
    insertion = landscape_task(
        "insertion",
        ins_score,
        ins_action_np,
        trust_radius=ins_profile.refinement.max_total_delta,
        eps_list=parse_eps(args.insertion_eps),
        random_radius_frac=args.random_radius_frac,
        min_positive_rate=args.min_positive_rate,
        min_negative_rate=args.min_negative_rate,
        min_fd_corr=args.min_fd_corr,
        max_fd_rel_error_p95=args.max_fd_rel_error_p95,
        min_random_corr=args.min_random_corr,
        max_random_rel_residual_p95=args.max_random_rel_residual_p95,
        seed=args.seed,
    )
    insertion["meta"] = ins_meta

    left_np, right_np, eef_np, joint_np, board_meta = sample_board(args)
    left = torch.tensor(left_np, dtype=torch.float32, device=runtime.device)
    right = torch.tensor(right_np, dtype=torch.float32, device=runtime.device)
    eef = torch.tensor(eef_np, dtype=torch.float32, device=runtime.device)
    board_score = make_score_fn(runtime, "board", left=left, right=right, eef=eef)
    board_profile = get_guidance_profile("board")
    board = landscape_task(
        "board",
        board_score,
        joint_np,
        trust_radius=board_profile.refinement.max_total_delta,
        eps_list=parse_eps(args.board_eps),
        random_radius_frac=args.random_radius_frac,
        min_positive_rate=args.min_positive_rate,
        min_negative_rate=args.min_negative_rate,
        min_fd_corr=args.min_fd_corr,
        max_fd_rel_error_p95=args.max_fd_rel_error_p95,
        min_random_corr=args.min_random_corr,
        max_random_rel_residual_p95=args.max_random_rel_residual_p95,
        seed=args.seed + 1,
    )
    board["meta"] = board_meta

    result = {
        "purpose": "Score-landscape diagnostic for TacQuality DP classifier guidance.",
        "scope": "Real-sample local action-space energy geometry; not a robot rollout.",
        "device": str(runtime.device),
        "seed": int(args.seed),
        "insertion": insertion,
        "board": board,
    }
    result["overall_pass"] = bool(insertion["passes_score_landscape"] and board["passes_score_landscape"])
    json_path = args.output_dir / "tac_quality_score_landscape.json"
    md_path = args.output_dir / "tac_quality_score_landscape.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "insertion_pass": insertion["passes_score_landscape"],
                "board_pass": board["passes_score_landscape"],
                "insertion_grad_norm_mean": insertion["gradient"]["grad_norm"]["mean"],
                "board_grad_norm_mean": board["gradient"]["grad_norm"]["mean"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
