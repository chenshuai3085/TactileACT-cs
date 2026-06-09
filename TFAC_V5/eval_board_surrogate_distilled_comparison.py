"""Compare PTGProxyV2 and distilled TacQualityEnergy in board surrogate guidance.

This is an action-level gate stronger than feature-only gradients:

  current tactile + action
      -> board tactile surrogate predicts future tactile
      -> scorer energy
      -> d energy / d action
      -> accepted trust-region action refinement

Both scorers are evaluated on the same board windows, surrogate model, and
trust-region parameters.  Scope: surrogate full-chain, not robot rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import DistilledTacQualityEnergyRuntime  # noqa: E402
from TFAC_V5.eval_tac_energy_guided_denoising import summarize  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID  # noqa: E402
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402
from TFAC_V5.train_board_tactile_surrogate import (  # noqa: E402
    BOARD_DIR,
    BoardTactileSurrogate,
    NormStats,
    norm_action,
    unnorm_marker,
)


OUT_DIR = Path("/home/chenshuai/Project/output/board_surrogate_distilled_comparison")
DEFAULT_SURROGATE = Path("/home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_final.pt")
DEFAULT_PTG = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")
DEFAULT_DISTILLED = Path("/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt")


def stats_from_ckpt(ckpt: Dict[str, object]) -> NormStats:
    stats = ckpt["norm_stats"]
    return NormStats(
        marker_mean=np.asarray(stats["marker_mean"], dtype=np.float32),
        marker_std=np.asarray(stats["marker_std"], dtype=np.float32),
        eef_mean=np.asarray(stats["eef_mean"], dtype=np.float32),
        eef_std=np.asarray(stats["eef_std"], dtype=np.float32),
        joint_mean=np.asarray(stats["joint_mean"], dtype=np.float32),
        joint_std=np.asarray(stats["joint_std"], dtype=np.float32),
    )


def load_surrogate(path: Path, device: torch.device) -> Tuple[BoardTactileSurrogate, NormStats, Dict[str, object]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    model = BoardTactileSurrogate(
        window=int(config.get("window", 8)),
        hidden=int(config.get("hidden", 512)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, stats_from_ckpt(ckpt), config


def collect_board_windows(data_dir: Path, window: int, stride: int) -> List[Tuple[Path, int]]:
    rows = []
    for path in sorted((data_dir / "success").glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            n = min(
                len(f["observations/tac/left/marker_offset"]),
                len(f["observations/tac/right/marker_offset"]),
                len(f["actions/eef_abs"]),
                len(f["actions/joint_abs"]),
            )
        max_t = n - window - 1
        for t in range(window - 1, max_t, stride):
            if t + 1 + window <= n:
                rows.append((path, t))
    return rows


def read_window(path: Path, t: int, window: int):
    cur = slice(t - window + 1, t + 1)
    fut = slice(t + 1, t + 1 + window)
    with h5py.File(path, "r") as f:
        left_cur = f["observations/tac/left/marker_offset"][cur].astype(np.float32)
        right_cur = f["observations/tac/right/marker_offset"][cur].astype(np.float32)
        eef = f["actions/eef_abs"][fut].astype(np.float32)
        joint = f["actions/joint_abs"][fut].astype(np.float32)
        left_future = f["observations/tac/left/marker_offset"][fut].astype(np.float32)
        right_future = f["observations/tac/right/marker_offset"][fut].astype(np.float32)
    return left_cur, right_cur, eef, joint, left_future, right_future


def normalize_marker_np(x: np.ndarray, stats: NormStats) -> np.ndarray:
    return ((x - stats.marker_mean.reshape(1, 1, 1, 2)) / stats.marker_std.reshape(1, 1, 1, 2)).astype(np.float32)


def action_smoothness_torch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 3:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    accel = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return torch.linalg.norm(accel, dim=-1).mean(dim=1)


def action_smoothness_np(x: np.ndarray) -> np.ndarray:
    if x.shape[1] < 3:
        return np.zeros(x.shape[0], dtype=np.float32)
    accel = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    return np.linalg.norm(accel, axis=-1).mean(axis=1)


def unit_grad_step(x: torch.Tensor, grad: torch.Tensor, step_size: float) -> torch.Tensor:
    norm = grad.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return x + step_size * grad / norm


def project_trust_region(proposal: torch.Tensor, base: torch.Tensor, max_total_delta: float) -> torch.Tensor:
    delta = proposal - base
    norm = delta.flatten(1).norm(dim=1).view(-1, 1, 1).clamp_min(1e-8)
    return base + delta * torch.clamp(max_total_delta / norm, max=1.0)


def score_with_runtime(args, scorer, scorer_name: str, pred_l, pred_r, eef_raw, joint_raw):
    task_id = torch.full((eef_raw.shape[0],), TASK_TO_ID["board"], dtype=torch.long, device=eef_raw.device)
    if scorer_name == "ptg_proxy_v2":
        score = scorer.weighted_energy_score(
            pred_l,
            right_marker_seq=pred_r,
            eef_action_seq=eef_raw,
            joint_action_seq=joint_raw,
            task_id=task_id,
            quality_weight=args.quality_weight,
            binary_weight=args.binary_weight,
            reason_weight=args.reason_weight,
            clip=not args.no_clip,
        )
    elif scorer_name == "distilled_energy":
        score = scorer.score(
            pred_l,
            right_marker_seq=pred_r,
            eef_action_seq=eef_raw,
            joint_action_seq=joint_raw,
            task_id=task_id,
            mode="energy_clipped" if not args.no_clip else "energy",
        )
    else:
        raise ValueError(scorer_name)
    if args.smooth_weight:
        score = score - args.smooth_weight * (
            action_smoothness_torch(eef_raw) + args.joint_smooth_scale * action_smoothness_torch(joint_raw)
        )
    return score


def predict_and_score(args, model, scorer, scorer_name: str, stats: NormStats, left_cur, right_cur, eef_raw, joint_raw):
    device = eef_raw.device
    eef_norm = norm_action(eef_raw, stats.eef_mean, stats.eef_std, device)
    joint_norm = norm_action(joint_raw, stats.joint_mean, stats.joint_std, device)
    pred_l_norm, pred_r_norm = model(left_cur, right_cur, eef_norm, joint_norm)
    pred_l = unnorm_marker(pred_l_norm, stats, device)
    pred_r = unnorm_marker(pred_r_norm, stats, device)
    return score_with_runtime(args, scorer, scorer_name, pred_l, pred_r, eef_raw, joint_raw), pred_l, pred_r


def refine_batch(args, model, scorer, scorer_name: str, stats: NormStats, batch, device: torch.device):
    left_np, right_np, eef_np, joint_np, left_future_np, right_future_np = batch
    left_cur = torch.tensor(normalize_marker_np(left_np, stats), dtype=torch.float32, device=device)
    right_cur = torch.tensor(normalize_marker_np(right_np, stats), dtype=torch.float32, device=device)
    base_eef = torch.tensor(eef_np, dtype=torch.float32, device=device)
    base_joint = torch.tensor(joint_np, dtype=torch.float32, device=device)
    current_eef = base_eef.clone()
    current_joint = base_joint.clone()
    step_logs = []

    for step_idx in range(args.refine_steps):
        eef = current_eef.detach().clone().requires_grad_(True)
        joint = current_joint.detach().clone().requires_grad_(True)
        score, _, _ = predict_and_score(args, model, scorer, scorer_name, stats, left_cur, right_cur, eef, joint)
        grad_eef, grad_joint = torch.autograd.grad(score.sum(), [eef, joint], retain_graph=False)
        with torch.no_grad():
            proposal_eef = project_trust_region(unit_grad_step(eef, grad_eef, args.action_step), base_eef, args.max_total_delta)
            proposal_joint = project_trust_region(
                unit_grad_step(joint, grad_joint, args.action_step),
                base_joint,
                args.max_total_delta * args.joint_delta_scale,
            )
        score_new, _, _ = predict_and_score(
            args,
            model,
            scorer,
            scorer_name,
            stats,
            left_cur,
            right_cur,
            proposal_eef.detach().clone().requires_grad_(True),
            proposal_joint.detach().clone().requires_grad_(True),
        )
        accept = score_new.detach() > score.detach() if args.accept_only_improved else torch.ones_like(score, dtype=torch.bool)
        with torch.no_grad():
            current_eef = torch.where(accept.view(-1, 1, 1), proposal_eef, current_eef)
            current_joint = torch.where(accept.view(-1, 1, 1), proposal_joint, current_joint)
        step_logs.append(
            {
                "step": int(step_idx),
                "score_mean": float(score.detach().mean().cpu()),
                "score_after_mean": float(score_new.detach().mean().cpu()),
                "accept_rate": float(accept.float().mean().cpu()),
                "eef_grad_norm_mean": float(grad_eef.flatten(1).norm(dim=1).detach().mean().cpu()),
                "joint_grad_norm_mean": float(grad_joint.flatten(1).norm(dim=1).detach().mean().cpu()),
            }
        )

    with torch.no_grad():
        base_score, base_pred_l, base_pred_r = predict_and_score(
            args, model, scorer, scorer_name, stats, left_cur, right_cur, base_eef, base_joint
        )
        refined_score, refined_pred_l, refined_pred_r = predict_and_score(
            args, model, scorer, scorer_name, stats, left_cur, right_cur, current_eef, current_joint
        )
    future = torch.tensor(np.concatenate([left_future_np, right_future_np], axis=0), dtype=torch.float32, device=device)
    base_pred = torch.cat([base_pred_l, base_pred_r], dim=0)
    refined_pred = torch.cat([refined_pred_l, refined_pred_r], dim=0)
    return {
        "base_score": base_score.detach().cpu().numpy(),
        "refined_score": refined_score.detach().cpu().numpy(),
        "eef_delta_norm": (current_eef - base_eef).detach().flatten(1).norm(dim=1).cpu().numpy(),
        "joint_delta_norm": (current_joint - base_joint).detach().flatten(1).norm(dim=1).cpu().numpy(),
        "base_eef_smooth": action_smoothness_np(eef_np),
        "refined_eef_smooth": action_smoothness_np(current_eef.detach().cpu().numpy()),
        "base_joint_smooth": action_smoothness_np(joint_np),
        "refined_joint_smooth": action_smoothness_np(current_joint.detach().cpu().numpy()),
        "base_marker_mae": (base_pred - future).abs().flatten(1).mean(dim=1).cpu().numpy(),
        "refined_marker_mae": (refined_pred - future).abs().flatten(1).mean(dim=1).cpu().numpy(),
        "step_logs": step_logs,
    }


def summarize_run(args, scorer_name: str, accum: Dict[str, List[np.ndarray]], detail_rows: List[Dict[str, Any]], step_logs):
    merged = {k: np.concatenate(v, axis=0) for k, v in accum.items()}
    score_delta = merged["refined_score"] - merged["base_score"]
    eef_smooth_delta = merged["refined_eef_smooth"] - merged["base_eef_smooth"]
    joint_smooth_delta = merged["refined_joint_smooth"] - merged["base_joint_smooth"]
    marker_mae_delta = merged["refined_marker_mae"] - merged["base_marker_mae"]
    return {
        "scorer": scorer_name,
        "n_windows": int(len(merged["base_score"])),
        "summary": {
            "base_score": summarize(merged["base_score"]),
            "refined_score": summarize(merged["refined_score"]),
            "score_delta": summarize(score_delta),
            "score_improved_rate": float(np.mean(score_delta > 0)),
            "eef_delta_norm": summarize(merged["eef_delta_norm"]),
            "joint_delta_norm": summarize(merged["joint_delta_norm"]),
            "eef_smooth_delta": summarize(eef_smooth_delta),
            "joint_smooth_delta": summarize(joint_smooth_delta),
            "base_marker_mae": summarize(merged["base_marker_mae"]),
            "refined_marker_mae": summarize(merged["refined_marker_mae"]),
            "marker_mae_delta": summarize(marker_mae_delta),
        },
        "step_logs": step_logs,
        "rows": detail_rows,
        "passes_board_surrogate_action_refinement": bool(
            np.mean(score_delta > 0) >= args.pass_improve_rate
            and np.mean(score_delta) > 0
            and np.max(merged["eef_delta_norm"]) <= args.max_total_delta + 1e-6
            and np.max(merged["joint_delta_norm"]) <= args.max_total_delta * args.joint_delta_scale + 1e-6
        ),
    }


def run_one(args, model, scorer, scorer_name: str, stats: NormStats, rows, device):
    accum = {
        "base_score": [],
        "refined_score": [],
        "eef_delta_norm": [],
        "joint_delta_norm": [],
        "base_eef_smooth": [],
        "refined_eef_smooth": [],
        "base_joint_smooth": [],
        "refined_joint_smooth": [],
        "base_marker_mae": [],
        "refined_marker_mae": [],
    }
    detail_rows = []
    step_logs = []
    for offset in tqdm(range(0, len(rows), args.batch_size), desc=f"{scorer_name} board surrogate refinement"):
        chunk = rows[offset : offset + args.batch_size]
        cols = [[] for _ in range(6)]
        for path, t in chunk:
            values = read_window(path, t, args.window)
            for col, value in zip(cols, values):
                col.append(value)
        batch = tuple(np.stack(col).astype(np.float32) for col in cols)
        out = refine_batch(args, model, scorer, scorer_name, stats, batch, device)
        for key in accum:
            accum[key].append(out[key])
        step_logs.extend(out["step_logs"])
        delta = out["refined_score"] - out["base_score"]
        for (path, t), base, refined, d in zip(chunk, out["base_score"], out["refined_score"], delta):
            detail_rows.append({"episode": path.stem, "t": int(t), "base_score": float(base), "refined_score": float(refined), "score_delta": float(d)})
    return summarize_run(args, scorer_name, accum, detail_rows, step_logs)


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Board Surrogate Distilled Guidance Comparison",
        "",
        "Action-level surrogate full-chain comparison. This is stronger than feature-level guidance but still not a robot rollout.",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- n_windows: `{result['n_windows']}`",
        "",
        "| scorer | pass | improved | score delta mean | eef delta max | joint delta max | eef smooth delta p95 | marker MAE delta mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["ptg_proxy_v2", "distilled_energy"]:
        row = result[key]
        s = row["summary"]
        lines.append(
            f"| {key} | {row['passes_board_surrogate_action_refinement']} | {s['score_improved_rate']:.4f} | "
            f"{s['score_delta']['mean']:.6f} | {s['eef_delta_norm']['max']:.6f} | {s['joint_delta_norm']['max']:.6f} | "
            f"{s['eef_smooth_delta']['p95']:.6f} | {s['marker_mae_delta']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Both scorers are run through the same board tactile surrogate and accepted trust-region action optimizer.",
            "- A pass means local scorer energy can be improved through action while respecting the trust region.",
            "- This does not prove real DP or robot improvement; it decides whether the distilled scorer deserves a full-chain gate.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    profile = get_guidance_profile("board")
    if args.quality_weight is None:
        args.quality_weight = profile.energy.quality
    if args.binary_weight is None:
        args.binary_weight = profile.energy.binary_margin
    if args.reason_weight is None:
        args.reason_weight = profile.energy.reason_margin
    if args.action_step is None:
        args.action_step = profile.refinement.action_step
    if args.max_total_delta is None:
        args.max_total_delta = profile.refinement.max_total_delta
    if args.refine_steps is None:
        args.refine_steps = profile.refinement.refine_steps

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, stats, surrogate_config = load_surrogate(Path(args.surrogate_ckpt), device)
    ptg = PTGProxyScorerV2Runtime(args.ptg_ckpt, device=str(device))
    distilled = DistilledTacQualityEnergyRuntime(args.distilled_ckpt, device=str(device))

    rows = collect_board_windows(Path(args.data_dir), args.window, args.stride)
    rng = np.random.default_rng(args.seed)
    if args.n_eval and len(rows) > args.n_eval:
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.n_eval, replace=False))]
    if not rows:
        raise RuntimeError("No board windows found")

    ptg_result = run_one(args, model, ptg, "ptg_proxy_v2", stats, rows, device)
    distilled_result = run_one(args, model, distilled, "distilled_energy", stats, rows, device)
    result = {
        "purpose": "Board surrogate full-chain action-level comparison for classifier-guidance scorers.",
        "scope": "Surrogate clean-action refinement only; not production DP and not robot rollout.",
        "config": vars(args),
        "surrogate_config": surrogate_config,
        "n_windows": int(len(rows)),
        "ptg_proxy_v2": ptg_result,
        "distilled_energy": distilled_result,
        "overall_pass": bool(
            ptg_result["passes_board_surrogate_action_refinement"]
            and distilled_result["passes_board_surrogate_action_refinement"]
        ),
        "recommendation": (
            "Distilled energy passes board surrogate action-level gate; compare in production Foresight/DP next."
            if distilled_result["passes_board_surrogate_action_refinement"]
            else "Do not promote distilled energy beyond scorer-level gates."
        ),
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "board_surrogate_distilled_comparison.json"
    md_path = out_dir / "board_surrogate_distilled_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "ptg": {
                    "pass": ptg_result["passes_board_surrogate_action_refinement"],
                    "improved": ptg_result["summary"]["score_improved_rate"],
                    "delta": ptg_result["summary"]["score_delta"]["mean"],
                },
                "distilled": {
                    "pass": distilled_result["passes_board_surrogate_action_refinement"],
                    "improved": distilled_result["summary"]["score_improved_rate"],
                    "delta": distilled_result["summary"]["score_delta"]["mean"],
                },
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(BOARD_DIR))
    parser.add_argument("--surrogate_ckpt", default=str(DEFAULT_SURROGATE))
    parser.add_argument("--ptg_ckpt", default=str(DEFAULT_PTG))
    parser.add_argument("--distilled_ckpt", default=str(DEFAULT_DISTILLED))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--n_eval", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refine_steps", type=int, default=None)
    parser.add_argument("--action_step", type=float, default=None)
    parser.add_argument("--max_total_delta", type=float, default=None)
    parser.add_argument("--joint_delta_scale", type=float, default=1.0)
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--joint_smooth_scale", type=float, default=0.1)
    parser.add_argument("--quality_weight", type=float, default=None)
    parser.add_argument("--binary_weight", type=float, default=None)
    parser.add_argument("--reason_weight", type=float, default=None)
    parser.add_argument("--no_clip", action="store_true")
    parser.add_argument("--accept_only_improved", action="store_true", default=True)
    parser.add_argument("--pass_improve_rate", type=float, default=0.95)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
