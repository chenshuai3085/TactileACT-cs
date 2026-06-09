"""Robustness audit for TacQuality classifier-guidance energy.

DP classifier guidance will evaluate the scorer on predicted tactile/action
states, not perfectly clean ground truth.  A useful guidance energy should keep
stable scores and action-gradient directions under small tactile/action noise.

This audit uses real insertion and board-wiping windows and measures:
  - score correlation under perturbation;
  - action-gradient cosine stability;
  - whether the clean gradient still improves perturbed samples;
  - whether the perturbed gradient improves perturbed samples.
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
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_robustness")


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


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
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


def sample_board(args):
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
    rng = np.random.default_rng(args.seed + 19)
    if len(rows) > args.n_board:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_board, replace=False).tolist())]
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


def field_std(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    return x.detach().float().std(dim=dims, keepdim=True).clamp_min(1e-6)


def score_and_grad(score_fn, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    action_req = action.detach().clone().requires_grad_(True)
    score = score_fn(action_req)
    grad = torch.autograd.grad(score.sum(), action_req, retain_graph=False)[0]
    return score.detach(), grad.detach()


def unit_step(action: torch.Tensor, grad: torch.Tensor, scale: float, max_total_delta: float) -> torch.Tensor:
    norm = grad.flatten(1).norm(dim=1).view(-1, *([1] * (grad.ndim - 1))).clamp_min(1e-8)
    proposal = action + float(scale) * grad / norm
    delta = proposal - action
    if max_total_delta > 0:
        delta_norm = delta.flatten(1).norm(dim=1).view(-1, *([1] * (delta.ndim - 1))).clamp_min(1e-8)
        proposal = action + delta * torch.clamp(max_total_delta / delta_norm, max=1.0)
    return proposal.detach()


def grad_cosine(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    af = a.flatten(1)
    bf = b.flatten(1)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=1, eps=1e-8)
    return cos.detach().cpu().numpy()


def eval_perturbation(
    task: str,
    score_fn_factory,
    marker_tensors: Dict[str, torch.Tensor],
    action: torch.Tensor,
    marker_noise: float,
    action_noise: float,
    guidance_scale: float,
    max_total_delta: float,
    generator: torch.Generator,
) -> Dict[str, object]:
    clean_score_fn = score_fn_factory(marker_tensors)
    clean_score, clean_grad = score_and_grad(clean_score_fn, action)
    perturbed_markers = {}
    for name, tensor in marker_tensors.items():
        if marker_noise > 0:
            noise = torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)
            perturbed_markers[name] = tensor + marker_noise * field_std(tensor) * noise
        else:
            perturbed_markers[name] = tensor
    if action_noise > 0:
        noise = torch.randn(action.shape, dtype=action.dtype, device=action.device, generator=generator)
        perturbed_action = action + action_noise * field_std(action) * noise
    else:
        perturbed_action = action.detach()
    pert_score_fn = score_fn_factory(perturbed_markers)
    pert_score, pert_grad = score_and_grad(pert_score_fn, perturbed_action)

    guided_by_clean = unit_step(perturbed_action, clean_grad, guidance_scale, max_total_delta)
    guided_by_pert = unit_step(perturbed_action, pert_grad, guidance_scale, max_total_delta)
    score_clean_step = pert_score_fn(guided_by_clean.detach().clone().requires_grad_(True)).detach()
    score_pert_step = pert_score_fn(guided_by_pert.detach().clone().requires_grad_(True)).detach()
    clean_np = clean_score.cpu().numpy()
    pert_np = pert_score.cpu().numpy()
    clean_step_delta = (score_clean_step - pert_score).cpu().numpy()
    pert_step_delta = (score_pert_step - pert_score).cpu().numpy()
    return {
        "task": task,
        "marker_noise": float(marker_noise),
        "action_noise": float(action_noise),
        "score": {
            "clean": summarize(clean_np),
            "perturbed": summarize(pert_np),
            "abs_delta": summarize(np.abs(pert_np - clean_np)),
            "pearson_corr": corr(clean_np, pert_np),
            "sign_same_rate": float(np.mean(np.sign(clean_np) == np.sign(pert_np))),
        },
        "gradient": {
            "clean_norm": summarize(clean_grad.flatten(1).norm(dim=1).cpu().numpy()),
            "perturbed_norm": summarize(pert_grad.flatten(1).norm(dim=1).cpu().numpy()),
            "cosine": summarize(grad_cosine(clean_grad, pert_grad)),
            "finite_rate": float(torch.isfinite(pert_grad).flatten(1).all(dim=1).float().mean().cpu()),
            "positive_norm_rate": float((pert_grad.flatten(1).norm(dim=1) > 1e-8).float().mean().cpu()),
        },
        "guided_step": {
            "scale": float(guidance_scale),
            "max_total_delta": float(max_total_delta),
            "clean_gradient_score_delta": summarize(clean_step_delta),
            "clean_gradient_improved_rate": float(np.mean(clean_step_delta > 0)),
            "perturbed_gradient_score_delta": summarize(pert_step_delta),
            "perturbed_gradient_improved_rate": float(np.mean(pert_step_delta > 0)),
        },
    }


def eval_task(
    task: str,
    score_fn_factory,
    marker_tensors: Dict[str, torch.Tensor],
    action: torch.Tensor,
    noise_grid: List[Tuple[float, float]],
    guidance_scale: float,
    max_total_delta: float,
    seed: int,
) -> Dict[str, object]:
    gen = torch.Generator(device=action.device)
    gen.manual_seed(seed)
    rows = [
        eval_perturbation(
            task,
            score_fn_factory,
            marker_tensors,
            action,
            marker_noise,
            action_noise,
            guidance_scale,
            max_total_delta,
            gen,
        )
        for marker_noise, action_noise in noise_grid
    ]
    worst_score_corr = min(r["score"]["pearson_corr"] for r in rows)
    worst_grad_cos_p05 = min(r["gradient"]["cosine"]["p05"] for r in rows)
    worst_pert_improved = min(r["guided_step"]["perturbed_gradient_improved_rate"] for r in rows)
    worst_clean_improved = min(r["guided_step"]["clean_gradient_improved_rate"] for r in rows)
    current_gradient_pass = all(
        r["gradient"]["finite_rate"] >= 0.999
        and r["gradient"]["positive_norm_rate"] >= 0.999
        and r["guided_step"]["perturbed_gradient_improved_rate"] >= 0.95
        and r["guided_step"]["perturbed_gradient_score_delta"]["mean"] > 0
        for r in rows
    )
    stale_gradient_stable = bool(
        worst_score_corr >= 0.95
        and worst_grad_cos_p05 >= 0.50
        and worst_clean_improved >= 0.90
    )
    return {
        "task": task,
        "n_samples": int(action.shape[0]),
        "guidance_scale": float(guidance_scale),
        "max_total_delta": float(max_total_delta),
        "rows": rows,
        "worst_score_corr": float(worst_score_corr),
        "worst_grad_cosine_p05": float(worst_grad_cos_p05),
        "worst_perturbed_gradient_improved_rate": float(worst_pert_improved),
        "worst_clean_gradient_improved_rate": float(worst_clean_improved),
        "passes_current_gradient_robustness": bool(current_gradient_pass),
        "stale_gradient_stable_under_noise": stale_gradient_stable,
        "passes_guidance_robustness": bool(current_gradient_pass),
        "deployment_constraint": (
            "Recompute TacQuality score gradients at every denoising/guidance step. "
            "Do not cache or reuse stale gradients across noisy predicted tactile/action states."
        ),
    }


def write_markdown(result: Dict[str, object], path: Path) -> None:
    lines = [
        "# TacQuality Guidance Robustness Audit",
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
                f"- current_gradient_pass: `{item['passes_current_gradient_robustness']}`",
                f"- stale_gradient_stable_under_noise: `{item['stale_gradient_stable_under_noise']}`",
                f"- worst_score_corr: `{item['worst_score_corr']}`",
                f"- worst_grad_cosine_p05: `{item['worst_grad_cosine_p05']}`",
                f"- worst_perturbed_gradient_improved_rate: `{item['worst_perturbed_gradient_improved_rate']}`",
                f"- worst_clean_gradient_improved_rate: `{item['worst_clean_gradient_improved_rate']}`",
                "",
                "| marker_noise | action_noise | score_corr | grad_cos_p05 | pert_grad_improve | clean_grad_improve |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in item["rows"]:
            lines.append(
                f"| {row['marker_noise']} | {row['action_noise']} | {row['score']['pearson_corr']} | "
                f"{row['gradient']['cosine']['p05']} | {row['guided_step']['perturbed_gradient_improved_rate']} | "
                f"{row['guided_step']['clean_gradient_improved_rate']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_noise_grid(text: str) -> List[Tuple[float, float]]:
    rows = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        marker, action = item.split(":")
        rows.append((float(marker), float(action)))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--n_insertion", type=int, default=256)
    parser.add_argument("--n_board", type=int, default=256)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_dir", type=Path, default=BOARD_DIR)
    parser.add_argument("--insertion_guidance_scale", type=float, default=0.04)
    parser.add_argument("--board_guidance_scale", type=float, default=0.0008)
    parser.add_argument("--noise_grid", default="0.00:0.00,0.02:0.01,0.05:0.02,0.10:0.05")
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
    ins_action = torch.tensor(ins_action_np, dtype=torch.float32, device=runtime.device)

    def ins_factory(markers):
        marker = markers["left"]

        def score(action):
            return runtime.score("insertion", marker, action, mode="profile")

        return score

    board_left_np, board_right_np, board_eef_np, board_joint_np, board_meta = sample_board(args)
    board_left = torch.tensor(board_left_np, dtype=torch.float32, device=runtime.device)
    board_right = torch.tensor(board_right_np, dtype=torch.float32, device=runtime.device)
    board_eef = torch.tensor(board_eef_np, dtype=torch.float32, device=runtime.device)
    board_joint = torch.tensor(board_joint_np, dtype=torch.float32, device=runtime.device)

    def board_factory(markers):
        left = markers["left"]
        right = markers["right"]
        eef = markers["eef"]

        def score(action):
            return runtime.score(
                "board",
                left,
                action,
                right_marker_seq=right,
                eef_action_seq=eef,
                mode="profile",
            )

        return score

    noise_grid = parse_noise_grid(args.noise_grid)
    insertion_profile = get_guidance_profile("insertion")
    board_profile = get_guidance_profile("board")
    insertion = eval_task(
        "insertion",
        ins_factory,
        {"left": ins_marker},
        ins_action,
        noise_grid,
        args.insertion_guidance_scale,
        insertion_profile.refinement.max_total_delta,
        args.seed,
    )
    insertion["meta"] = ins_meta
    board = eval_task(
        "board",
        board_factory,
        {"left": board_left, "right": board_right, "eef": board_eef},
        board_joint,
        noise_grid,
        args.board_guidance_scale,
        board_profile.refinement.max_total_delta,
        args.seed + 1,
    )
    board["meta"] = board_meta
    result = {
        "purpose": "Robustness/stability audit for TacQuality classifier-guidance energy.",
        "scope": "Real-sample score and action-gradient stability under small tactile/action perturbations.",
        "interpretation": (
            "The key deployment criterion is current-gradient robustness, because DP classifier guidance "
            "recomputes d score / d action at the current denoising state.  Stale-gradient stability is "
            "reported as a safety diagnostic and should not be assumed."
        ),
        "device": str(runtime.device),
        "seed": int(args.seed),
        "noise_grid": [{"marker_noise": m, "action_noise": a} for m, a in noise_grid],
        "insertion": insertion,
        "board": board,
    }
    result["overall_pass"] = bool(
        insertion["passes_current_gradient_robustness"]
        and board["passes_current_gradient_robustness"]
    )
    json_path = args.output_dir / "tac_quality_guidance_robustness.json"
    md_path = args.output_dir / "tac_quality_guidance_robustness.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "insertion": {
                    "current_gradient_pass": insertion["passes_current_gradient_robustness"],
                    "stale_gradient_stable": insertion["stale_gradient_stable_under_noise"],
                    "worst_score_corr": insertion["worst_score_corr"],
                    "worst_grad_cosine_p05": insertion["worst_grad_cosine_p05"],
                    "worst_perturbed_gradient_improved_rate": insertion["worst_perturbed_gradient_improved_rate"],
                },
                "board": {
                    "current_gradient_pass": board["passes_current_gradient_robustness"],
                    "stale_gradient_stable": board["stale_gradient_stable_under_noise"],
                    "worst_score_corr": board["worst_score_corr"],
                    "worst_grad_cosine_p05": board["worst_grad_cosine_p05"],
                    "worst_perturbed_gradient_improved_rate": board["worst_perturbed_gradient_improved_rate"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
