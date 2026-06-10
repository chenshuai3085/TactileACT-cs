"""Action-gradient smoke test for the manual-board TacQualityEnergy scorer.

This is a narrow diagnostic for the current distilled TacQualityEnergy model
trained from unchanged insertion labels plus the manually supplied board
positive/too-light directories.  It does not evaluate robot success and it does
not use Foresight.  It checks whether the scorer is locally usable as a DP
classifier-guidance potential on real data tensors:

  real tactile/action window -> score -> d(score)/d(action)

For each task, it samples real windows, computes a score gradient with respect
to the joint action sequence, takes small trust-region gradient-ascent steps,
and records whether the score improves while the action delta stays bounded.
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

from TFAC_V5.build_manual_board_tac_quality_features import contact_segments, pad_window  # noqa: E402
from TFAC_V5.distilled_tac_quality_energy_runtime import DistilledTacQualityEnergyRuntime  # noqa: E402
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_trust_region_guidance import TacQualityTrustRegionRefiner, TrustRegionConfig  # noqa: E402


DEFAULT_CKPT = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy/distilled_tac_quality_energy_final.pt")
INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
BOARD_POS_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609")
BOARD_NEG_LIGHT_DIR = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high")
OUT_DIR = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy_action_gradient_smoke")


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
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def task_refiner(task: str, action_step_scale: float) -> TacQualityTrustRegionRefiner:
    profile = get_guidance_profile(task)
    refine = profile.refinement
    return TacQualityTrustRegionRefiner(
        TrustRegionConfig(
            steps=refine.refine_steps,
            step_size=refine.action_step * action_step_scale,
            max_total_delta=refine.max_total_delta * action_step_scale,
            accept_only_improved=refine.accept_only_improved,
        )
    )


def sample_insertion(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    data = np.load(args.insertion_features, allow_pickle=True)
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
    sample_ids = data["sample_ids"]

    rng = np.random.default_rng(args.seed)
    selected: List[int] = []
    per_reason = max(1, args.n_insertion // max(1, len(np.unique(reason))))
    for cls in sorted(np.unique(reason).tolist()):
        idx = np.flatnonzero(reason == cls)
        if len(idx):
            selected.extend(rng.choice(idx, min(per_reason, len(idx)), replace=False).tolist())
    if len(selected) < args.n_insertion:
        rest = np.setdiff1d(np.arange(len(marker)), np.asarray(selected, dtype=np.int64), assume_unique=False)
        fill = rng.choice(rest, min(args.n_insertion - len(selected), len(rest)), replace=False)
        selected.extend(fill.tolist())
    idx = np.asarray(selected[: args.n_insertion], dtype=np.int64)
    meta = {
        "source": str(args.insertion_features),
        "n_samples": int(len(idx)),
        "sampling": "balanced by available insertion reason classes",
        "reason_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(reason[idx], return_counts=True))},
        "quality": summarize(quality[idx]),
        "sample_ids_preview": [str(x) for x in sample_ids[idx[: min(10, len(idx))]]],
        "note": "Insertion cache has one marker field and joint_abs action only; right marker is mirrored from left and eef action is zero for this smoke.",
    }
    return marker[idx], action[idx], reason[idx], meta


def board_rows_for_dir(root: Path, label: str, args: argparse.Namespace) -> List[Tuple[Path, str, int, int]]:
    rows: List[Tuple[Path, str, int, int]] = []
    for path in sorted(root.glob("*.hdf5")):
        with h5py.File(path, "r") as f:
            n = min(
                len(f["observations/tac/left/marker_offset"]),
                len(f["observations/tac/right/marker_offset"]),
                len(f["actions/eef_abs"]),
                len(f["actions/joint_abs"]),
                len(f["ft"]),
            )
            ft = f["ft"][:n]
        for seg_start, seg_end in contact_segments(ft, min_len=max(args.board_window, 8), q=args.board_contact_quantile):
            for start in range(seg_start, max(seg_start + 1, seg_end - args.board_window + 1), args.board_stride):
                end = min(seg_end, start + args.board_window)
                if end - start >= max(8, args.board_window // 2):
                    rows.append((path, label, int(start), int(end)))
    return rows


def sample_board(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    all_rows = board_rows_for_dir(args.board_positive_dir, "good_smooth", args)
    all_rows += board_rows_for_dir(args.board_negative_light_dir, "too_light_unclean", args)
    if not all_rows:
        raise RuntimeError("No board contact windows found")
    rng = np.random.default_rng(args.seed + 31)
    if len(all_rows) > args.n_board:
        chosen = rng.choice(len(all_rows), args.n_board, replace=False)
        rows = [all_rows[i] for i in sorted(chosen.tolist())]
    else:
        rows = all_rows

    left, right, eef, joint = [], [], [], []
    labels: List[str] = []
    for path, label, start, end in rows:
        with h5py.File(path, "r") as f:
            left.append(pad_window(f["observations/tac/left/marker_offset"][start:end], args.board_window))
            right.append(pad_window(f["observations/tac/right/marker_offset"][start:end], args.board_window))
            eef.append(pad_window(f["actions/eef_abs"][start:end], args.board_window))
            joint.append(pad_window(f["actions/joint_abs"][start:end], args.board_window))
            labels.append(label)
    label_values, label_counts = np.unique(np.asarray(labels), return_counts=True)
    meta = {
        "positive_dir": str(args.board_positive_dir),
        "negative_light_dir": str(args.board_negative_light_dir),
        "n_samples": int(len(rows)),
        "label_counts": {str(k): int(v) for k, v in zip(label_values, label_counts)},
        "window": int(args.board_window),
        "stride": int(args.board_stride),
        "contact_quantile": float(args.board_contact_quantile),
        "rows_preview": [
            {"episode": p.stem, "label": label, "start": start, "end": end}
            for p, label, start, end in rows[: min(10, len(rows))]
        ],
        "note": "Board smoke uses contact-only windows from the manually supplied good and too-light/unclean directories.",
    }
    return (
        np.stack(left).astype(np.float32),
        np.stack(right).astype(np.float32),
        np.stack(eef).astype(np.float32),
        np.stack(joint).astype(np.float32),
        meta,
    )


def gradient_probe(score: torch.Tensor, action: torch.Tensor) -> Dict[str, object]:
    grad = torch.autograd.grad(score.sum(), action, retain_graph=False)[0]
    flat = grad.flatten(1)
    norms = flat.norm(dim=1).detach().cpu().numpy()
    finite = torch.isfinite(flat).all(dim=1).detach().cpu().numpy().astype(bool)
    return {
        "finite_grad_rate": float(np.mean(finite)),
        "positive_grad_rate": float(np.mean(norms > 1e-8)),
        "grad_norm": summarize(norms),
    }


def tensor_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def run_insertion(args: argparse.Namespace, runtime: DistilledTacQualityEnergyRuntime) -> Dict[str, object]:
    marker_np, action_np, _reason, meta = sample_insertion(args)
    device = runtime.device
    marker = torch.tensor(marker_np, dtype=torch.float32, device=device)
    action = torch.tensor(action_np, dtype=torch.float32, device=device, requires_grad=True)
    task_id = torch.zeros(marker.shape[0], dtype=torch.long, device=device)

    def score_fn(a: torch.Tensor) -> torch.Tensor:
        return runtime.score(marker, right_marker_seq=marker, joint_action_seq=a, task_id=task_id, mode=args.score_mode)

    base_score = score_fn(action)
    grad = gradient_probe(base_score, action)
    refined, report = task_refiner("insertion", args.action_step_scale).refine(action.detach(), score_fn)
    final_score = score_fn(refined).detach()
    delta = tensor_np(final_score - base_score.detach())
    delta_norm = tensor_np((refined - action.detach()).flatten(1).norm(dim=1))
    return {
        "task": "insertion",
        "meta": meta,
        "base_score": summarize(tensor_np(base_score)),
        "final_score": summarize(tensor_np(final_score)),
        "score_delta": summarize(delta),
        "score_improved_rate": float(np.mean(delta > 0)),
        "action_delta_norm": summarize(delta_norm),
        "gradient_probe": grad,
        "trust_region_report": report,
        "passes": bool(
            np.isfinite(delta).all()
            and grad["finite_grad_rate"] >= args.min_finite_grad_rate
            and grad["positive_grad_rate"] >= args.min_positive_grad_rate
            and float(np.mean(delta > 0)) >= args.min_improved_rate
            and bool(report["max_delta_within_trust_region"])
        ),
    }


def run_board(args: argparse.Namespace, runtime: DistilledTacQualityEnergyRuntime) -> Dict[str, object]:
    left_np, right_np, eef_np, joint_np, meta = sample_board(args)
    device = runtime.device
    left = torch.tensor(left_np, dtype=torch.float32, device=device)
    right = torch.tensor(right_np, dtype=torch.float32, device=device)
    eef = torch.tensor(eef_np, dtype=torch.float32, device=device)
    joint = torch.tensor(joint_np, dtype=torch.float32, device=device, requires_grad=True)
    task_id = torch.ones(left.shape[0], dtype=torch.long, device=device)

    def score_fn(a: torch.Tensor) -> torch.Tensor:
        return runtime.score(left, right_marker_seq=right, eef_action_seq=eef, joint_action_seq=a, task_id=task_id, mode=args.score_mode)

    base_score = score_fn(joint)
    grad = gradient_probe(base_score, joint)
    refined, report = task_refiner("board", args.action_step_scale).refine(joint.detach(), score_fn)
    final_score = score_fn(refined).detach()
    delta = tensor_np(final_score - base_score.detach())
    delta_norm = tensor_np((refined - joint.detach()).flatten(1).norm(dim=1))
    return {
        "task": "board",
        "meta": meta,
        "base_score": summarize(tensor_np(base_score)),
        "final_score": summarize(tensor_np(final_score)),
        "score_delta": summarize(delta),
        "score_improved_rate": float(np.mean(delta > 0)),
        "action_delta_norm": summarize(delta_norm),
        "gradient_probe": grad,
        "trust_region_report": report,
        "passes": bool(
            np.isfinite(delta).all()
            and grad["finite_grad_rate"] >= args.min_finite_grad_rate
            and grad["positive_grad_rate"] >= args.min_positive_grad_rate
            and float(np.mean(delta > 0)) >= args.min_improved_rate
            and bool(report["max_delta_within_trust_region"])
        ),
    }


def write_markdown(result: Dict[str, object], path: Path) -> None:
    ins = result["insertion"]
    board = result["board"]
    lines = [
        "# Manual-board TacQualityEnergy Action-gradient Smoke",
        "",
        "## Scope",
        "",
        "This diagnostic checks whether the current distilled TacQualityEnergy scorer has finite, non-zero action gradients on real insertion and manually labelled board windows.",
        "It is not a Foresight/DP closed-loop test and not a robot rollout.",
        "",
        "## Summary",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- score_mode: `{result['score_mode']}`",
        f"- device: `{result['device']}`",
        "",
        "## Insertion",
        "",
        f"- pass: `{ins['passes']}`",
        f"- n_samples: `{ins['meta']['n_samples']}`",
        f"- reason_counts: `{ins['meta']['reason_counts']}`",
        f"- score_improved_rate: `{ins['score_improved_rate']:.6f}`",
        f"- score_delta_mean: `{ins['score_delta']['mean']:.8f}`",
        f"- action_delta_norm_max: `{ins['action_delta_norm']['max']:.8f}`",
        f"- grad_finite_rate: `{ins['gradient_probe']['finite_grad_rate']:.6f}`",
        f"- grad_positive_rate: `{ins['gradient_probe']['positive_grad_rate']:.6f}`",
        f"- grad_norm_mean: `{ins['gradient_probe']['grad_norm']['mean']:.8f}`",
        "",
        "## Board",
        "",
        f"- pass: `{board['passes']}`",
        f"- n_samples: `{board['meta']['n_samples']}`",
        f"- label_counts: `{board['meta']['label_counts']}`",
        f"- score_improved_rate: `{board['score_improved_rate']:.6f}`",
        f"- score_delta_mean: `{board['score_delta']['mean']:.8f}`",
        f"- action_delta_norm_max: `{board['action_delta_norm']['max']:.8f}`",
        f"- grad_finite_rate: `{board['gradient_probe']['finite_grad_rate']:.6f}`",
        f"- grad_positive_rate: `{board['gradient_probe']['positive_grad_rate']:.6f}`",
        f"- grad_norm_mean: `{board['gradient_probe']['grad_norm']['mean']:.8f}`",
        "",
        "## Interpretation",
        "",
        "Passing this smoke means the scorer is locally differentiable with respect to real action tensors and a small trust-region ascent step improves the scorer's own energy.",
        "It does not prove that guided DP improves real task outcomes; that still requires Foresight-conditioned DP evaluation and real rollout validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--board_positive_dir", type=Path, default=BOARD_POS_DIR)
    parser.add_argument("--board_negative_light_dir", type=Path, default=BOARD_NEG_LIGHT_DIR)
    parser.add_argument("--output_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score_mode", default="energy_clipped")
    parser.add_argument("--n_insertion", type=int, default=192)
    parser.add_argument("--n_board", type=int, default=192)
    parser.add_argument("--board_window", type=int, default=32)
    parser.add_argument("--board_stride", type=int, default=32)
    parser.add_argument("--board_contact_quantile", type=float, default=0.60)
    parser.add_argument("--action_step_scale", type=float, default=1.0)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.99)
    parser.add_argument("--min_improved_rate", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = DistilledTacQualityEnergyRuntime(str(args.checkpoint), device=args.device)
    insertion = run_insertion(args, runtime)
    board = run_board(args, runtime)
    result = {
        "purpose": "Real-window action-gradient smoke for the manual-board TacQualityEnergy scorer.",
        "scope": "Scorer/action-gradient diagnostic only; not Foresight, DP, or robot rollout validation.",
        "checkpoint": str(args.checkpoint),
        "score_mode": args.score_mode,
        "device": str(runtime.device),
        "config": {
            "n_insertion": int(args.n_insertion),
            "n_board": int(args.n_board),
            "board_window": int(args.board_window),
            "board_stride": int(args.board_stride),
            "board_contact_quantile": float(args.board_contact_quantile),
            "action_step_scale": float(args.action_step_scale),
            "min_finite_grad_rate": float(args.min_finite_grad_rate),
            "min_positive_grad_rate": float(args.min_positive_grad_rate),
            "min_improved_rate": float(args.min_improved_rate),
        },
        "insertion": insertion,
        "board": board,
    }
    result["overall_pass"] = bool(insertion["passes"] and board["passes"])
    json_path = args.output_dir / "manual_board_energy_action_gradient_smoke.json"
    md_path = args.output_dir / "manual_board_energy_action_gradient_smoke.md"
    result["output_json"] = str(json_path)
    result["output_markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps(
        {
            "overall_pass": result["overall_pass"],
            "device": result["device"],
            "insertion_pass": insertion["passes"],
            "insertion_improved_rate": insertion["score_improved_rate"],
            "insertion_score_delta_mean": insertion["score_delta"]["mean"],
            "board_pass": board["passes"],
            "board_improved_rate": board["score_improved_rate"],
            "board_score_delta_mean": board["score_delta"]["mean"],
            "json": str(json_path),
            "md": str(md_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
