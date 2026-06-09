"""Evaluate board-wiping scorer readiness for gradient guidance.

This is the strongest board-side check possible before a board-specific
Foresight/DP stack exists.  It does not claim full-chain guidance.  Instead, it
checks whether PTGProxyScorerV2 board energy provides finite, useful gradients
with respect to the variables that Foresight/DP would control:

  left/right predicted marker field + eef/joint action -> board energy

The acceptance test is local gradient ascent on real board windows.  If a small
step consistently raises the board energy without worsening simple action
smoothness, the scorer is ready to be connected to a board Foresight model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_tac_energy_guided_denoising import summarize
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile


BOARD_DIR = Path("/home/chenshuai/data/dataset/260522_v8l_caheiban")
OUT_DIR = Path("/home/chenshuai/Project/output/board_guidance_readiness")


def action_smoothness(actions: np.ndarray) -> np.ndarray:
    if actions.shape[1] < 3:
        return np.zeros(actions.shape[0], dtype=np.float32)
    accel = actions[:, 2:] - 2 * actions[:, 1:-1] + actions[:, :-2]
    return np.linalg.norm(accel, axis=-1).mean(axis=1)


def load_windows(args):
    files = sorted((Path(args.data_dir) / "success").glob("*.hdf5"))
    rng = np.random.default_rng(args.seed)
    rows = []
    for path in files:
        with h5py.File(path, "r") as f:
            n = min(
                len(f["observations/tac/left/marker_offset"]),
                len(f["observations/tac/right/marker_offset"]),
                len(f["actions/eef_abs"]),
                len(f["actions/joint_abs"]),
            )
        starts = list(range(0, max(1, n - args.window + 1), args.stride))
        for start in starts:
            end = min(n, start + args.window)
            if end - start >= max(8, args.window // 2):
                rows.append((path, start, end))
    if args.n_eval and len(rows) > args.n_eval:
        idx = rng.choice(len(rows), args.n_eval, replace=False)
        rows = [rows[i] for i in sorted(idx)]
    return rows


def pad_last(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) >= window:
        return x[-window:]
    pad = np.repeat(x[:1], window - len(x), axis=0)
    return np.concatenate([pad, x], axis=0)


def read_row(path: Path, start: int, end: int, window: int):
    with h5py.File(path, "r") as f:
        left = pad_last(f["observations/tac/left/marker_offset"][start:end], window)
        right = pad_last(f["observations/tac/right/marker_offset"][start:end], window)
        eef = pad_last(f["actions/eef_abs"][start:end], window)
        joint = pad_last(f["actions/joint_abs"][start:end], window)
    return left, right, eef, joint


def unit_step(x: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
    flat_norm = grad.flatten(1).norm(dim=1).view(-1, *([1] * (grad.ndim - 1))).clamp_min(1e-8)
    return x + step_size * grad / flat_norm


def evaluate_batch(args, scorer, batch):
    left_np, right_np, eef_np, joint_np = batch
    device = scorer.device
    left = torch.tensor(left_np, dtype=torch.float32, device=device, requires_grad=True)
    right = torch.tensor(right_np, dtype=torch.float32, device=device, requires_grad=True)
    eef = torch.tensor(eef_np, dtype=torch.float32, device=device, requires_grad=True)
    joint = torch.tensor(joint_np, dtype=torch.float32, device=device, requires_grad=True)
    task_id = torch.full((left.shape[0],), TASK_TO_ID["board"], dtype=torch.long, device=device)

    score = scorer.weighted_energy_score(
        left,
        right_marker_seq=right,
        eef_action_seq=eef,
        joint_action_seq=joint,
        task_id=task_id,
        quality_weight=args.quality_weight,
        binary_weight=args.binary_weight,
        reason_weight=args.reason_weight,
        clip=not args.no_clip,
    )
    grads = torch.autograd.grad(score.sum(), [left, right, eef, joint], retain_graph=False)

    with torch.no_grad():
        left_p = unit_step(left, grads[0], args.marker_step)
        right_p = unit_step(right, grads[1], args.marker_step)
        eef_p = unit_step(eef, grads[2], args.action_step)
        joint_p = unit_step(joint, grads[3], args.action_step)

    score_new = scorer.weighted_energy_score(
        left_p.detach().clone().requires_grad_(True),
        right_marker_seq=right_p.detach().clone().requires_grad_(True),
        eef_action_seq=eef_p.detach().clone().requires_grad_(True),
        joint_action_seq=joint_p.detach().clone().requires_grad_(True),
        task_id=task_id,
        quality_weight=args.quality_weight,
        binary_weight=args.binary_weight,
        reason_weight=args.reason_weight,
        clip=not args.no_clip,
    )
    grad_norms = [g.flatten(1).norm(dim=1).detach().cpu().numpy() for g in grads]
    finite = [torch.isfinite(g).flatten(1).all(dim=1).detach().cpu().numpy() for g in grads]
    base_eef_smooth = action_smoothness(eef_np)
    new_eef_smooth = action_smoothness(eef_p.detach().cpu().numpy())
    base_joint_smooth = action_smoothness(joint_np)
    new_joint_smooth = action_smoothness(joint_p.detach().cpu().numpy())
    return {
        "score": score.detach().cpu().numpy(),
        "score_new": score_new.detach().cpu().numpy(),
        "left_grad_norm": grad_norms[0],
        "right_grad_norm": grad_norms[1],
        "eef_grad_norm": grad_norms[2],
        "joint_grad_norm": grad_norms[3],
        "left_finite": finite[0],
        "right_finite": finite[1],
        "eef_finite": finite[2],
        "joint_finite": finite[3],
        "eef_smooth_delta": new_eef_smooth - base_eef_smooth,
        "joint_smooth_delta": new_joint_smooth - base_joint_smooth,
    }


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    scorer = PTGProxyScorerV2Runtime(args.scorer_ckpt, device=args.device)
    rows = load_windows(args)
    if not rows:
        raise RuntimeError("No board windows found")

    accum = {
        "score": [],
        "score_new": [],
        "left_grad_norm": [],
        "right_grad_norm": [],
        "eef_grad_norm": [],
        "joint_grad_norm": [],
        "left_finite": [],
        "right_finite": [],
        "eef_finite": [],
        "joint_finite": [],
        "eef_smooth_delta": [],
        "joint_smooth_delta": [],
    }
    detail_rows = []
    for offset in tqdm(range(0, len(rows), args.batch_size), desc="Board guidance readiness"):
        chunk = rows[offset : offset + args.batch_size]
        left, right, eef, joint = [], [], [], []
        for path, start, end in chunk:
            l, r, e, j = read_row(path, start, end, args.window)
            left.append(l)
            right.append(r)
            eef.append(e)
            joint.append(j)
        out = evaluate_batch(
            args,
            scorer,
            (
                np.stack(left).astype(np.float32),
                np.stack(right).astype(np.float32),
                np.stack(eef).astype(np.float32),
                np.stack(joint).astype(np.float32),
            ),
        )
        for key in accum:
            accum[key].append(out[key])
        score_delta = out["score_new"] - out["score"]
        for (path, start, end), before, after, delta in zip(chunk, out["score"], out["score_new"], score_delta):
            detail_rows.append(
                {
                    "episode": path.stem,
                    "start": int(start),
                    "end": int(end),
                    "score": float(before),
                    "score_after_step": float(after),
                    "score_delta": float(delta),
                }
            )

    merged = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    score_delta = merged["score_new"] - merged["score"]
    finite_all = merged["left_finite"] & merged["right_finite"] & merged["eef_finite"] & merged["joint_finite"]
    grad_positive_all = (
        (merged["left_grad_norm"] > 1e-8)
        & (merged["right_grad_norm"] > 1e-8)
        & (merged["eef_grad_norm"] > 1e-8)
        & (merged["joint_grad_norm"] > 1e-8)
    )
    result = {
        "config": vars(args),
        "n_windows": int(len(merged["score"])),
        "summary": {
            "score": summarize(merged["score"]),
            "score_after_step": summarize(merged["score_new"]),
            "score_delta": summarize(score_delta),
            "score_improved_rate": float(np.mean(score_delta > 0)),
            "finite_grad_rate_all_inputs": float(np.mean(finite_all)),
            "positive_grad_rate_all_inputs": float(np.mean(grad_positive_all)),
            "left_grad_norm": summarize(merged["left_grad_norm"]),
            "right_grad_norm": summarize(merged["right_grad_norm"]),
            "eef_grad_norm": summarize(merged["eef_grad_norm"]),
            "joint_grad_norm": summarize(merged["joint_grad_norm"]),
            "eef_smooth_delta": summarize(merged["eef_smooth_delta"]),
            "joint_smooth_delta": summarize(merged["joint_smooth_delta"]),
        },
        "interpretation": {
            "passes_board_guidance_readiness": bool(
                np.mean(finite_all) >= 0.999
                and np.mean(grad_positive_all) >= 0.999
                and np.mean(score_delta > 0) >= args.pass_improve_rate
                and np.mean(score_delta) > 0
            ),
            "scope": "Scorer-level board gradient readiness only. Full chain still requires board Foresight/DP checkpoint.",
        },
        "rows": detail_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))


def parse_args():
    profile = get_guidance_profile("board")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(BOARD_DIR))
    parser.add_argument("--scorer_ckpt", default="/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--n_eval", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality_weight", type=float, default=profile.energy.quality)
    parser.add_argument("--binary_weight", type=float, default=profile.energy.binary_margin)
    parser.add_argument("--reason_weight", type=float, default=profile.energy.reason_margin)
    parser.add_argument("--marker_step", type=float, default=profile.refinement.marker_step)
    parser.add_argument("--action_step", type=float, default=profile.refinement.action_step)
    parser.add_argument("--pass_improve_rate", type=float, default=0.95)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--output", default=str(OUT_DIR / "board_ptg_v2_energy_readiness_N240.json"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
