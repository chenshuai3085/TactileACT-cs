"""Evaluate ActionAwareMarkerScorer as a DP classifier-guidance potential.

The training eval says whether the model predicts labels under episode-level
splits.  This script asks a different question needed for DP guidance:

  Is the deployed runtime score a useful differentiable potential over action?

It checks score/label agreement, saturation, finite action gradients, and
whether a small normalized action-gradient step increases the score on cached
real samples.  It is not a rollout result and does not claim robot success.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.action_aware_scorer_runtime import ActionAwareScorerRuntime  # noqa: E402


DEFAULT_FEATURES = Path("/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_features.npz")
DEFAULT_EVAL = Path("/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_eval.json")
DEFAULT_CKPT = Path("/home/chenshuai/Project/output/action_aware_marker_scorer/action_aware_marker_scorer_final.pt")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/action_aware_guidance_suitability")
MODES = ("log_p_good", "quality", "hybrid")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def saturation(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    return {
        "lt_0p02": float(np.mean(arr < 0.02)),
        "gt_0p98": float(np.mean(arr > 0.98)),
        "prob_saturated_total": float(np.mean((arr < 0.02) | (arr > 0.98))),
    }


def choose_probe_indices(task: np.ndarray, y_binary: np.ndarray, n_per_task_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx: List[int] = []
    for task_name in sorted(np.unique(task).tolist()):
        for cls in [0, 1]:
            candidates = np.flatnonzero((task == task_name) & (y_binary == cls))
            if len(candidates) == 0:
                continue
            take = rng.choice(candidates, min(n_per_task_class, len(candidates)), replace=False)
            idx.extend(take.tolist())
    rng.shuffle(idx)
    return np.asarray(idx, dtype=np.int64)


def forward_scores(
    runtime: ActionAwareScorerRuntime,
    marker: np.ndarray,
    action: np.ndarray,
    task_id: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    out: Dict[str, List[np.ndarray]] = {mode: [] for mode in MODES}
    out.update({"p_good": [], "quality_prob": []})
    with torch.no_grad():
        for start in range(0, len(marker), batch_size):
            m = torch.from_numpy(marker[start : start + batch_size]).to(runtime.device)
            a = torch.from_numpy(action[start : start + batch_size]).to(runtime.device)
            tid = torch.from_numpy(task_id[start : start + batch_size]).to(runtime.device)
            raw = runtime.forward(m, a, tid)
            p_good = torch.softmax(raw["binary_logits"], dim=-1)[:, 1]
            quality = torch.sigmoid(raw["score"])
            out["p_good"].append(p_good.detach().cpu().numpy())
            out["quality_prob"].append(quality.detach().cpu().numpy())
            for mode in MODES:
                out[mode].append(runtime.score(m, a, tid, mode=mode).detach().cpu().numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in out.items()}


def mode_metrics(scores: np.ndarray, y_binary: np.ndarray, quality: np.ndarray) -> Dict[str, Any]:
    valid = y_binary >= 0
    result = {
        "summary": summarize(scores),
        "corr_with_quality": corr(scores, quality),
        "good_minus_bad_mean": None,
        "binary_auc": None,
    }
    if valid.any() and len(np.unique(y_binary[valid])) == 2:
        result["binary_auc"] = float(roc_auc_score(y_binary[valid], scores[valid]))
        result["good_minus_bad_mean"] = float(scores[y_binary == 1].mean() - scores[y_binary == 0].mean())
    return result


def gradient_probe(
    runtime: ActionAwareScorerRuntime,
    marker: np.ndarray,
    action: np.ndarray,
    task_id: np.ndarray,
    task: np.ndarray,
    y_binary: np.ndarray,
    modes: Iterable[str],
    n_per_task_class: int,
    step_size: float,
    line_search_steps: List[float],
    seed: int,
) -> Dict[str, Any]:
    idx = choose_probe_indices(task, y_binary, n_per_task_class, seed)
    result: Dict[str, Any] = {"n_probe": int(len(idx)), "modes": {}}
    if len(idx) == 0:
        return result

    for mode in modes:
        rows = {}
        for task_name in sorted(np.unique(task[idx]).tolist()) + ["mixed"]:
            if task_name == "mixed":
                sub = idx
            else:
                sub = idx[task[idx] == task_name]
            if len(sub) == 0:
                continue
            m0 = torch.tensor(marker[sub], dtype=torch.float32, device=runtime.device, requires_grad=True)
            a0 = torch.tensor(action[sub], dtype=torch.float32, device=runtime.device, requires_grad=True)
            tid = torch.tensor(task_id[sub], dtype=torch.long, device=runtime.device)
            score0 = runtime.score(m0, a0, tid, mode=mode)
            grad_action = torch.autograd.grad(score0.sum(), a0, retain_graph=False)[0]
            grad_flat = grad_action.flatten(1)
            grad_norm = grad_flat.norm(dim=1)
            grad_rms = grad_norm / math.sqrt(float(grad_flat.shape[1]))
            finite = torch.isfinite(grad_flat).all(dim=1)
            with torch.no_grad():
                direction = grad_action / grad_norm.view(-1, 1, 1).clamp_min(1e-8)
                score1 = runtime.score(m0, a0 + step_size * direction, tid, mode=mode)
                delta = score1 - score0
                ls_scores = []
                for ls_step in line_search_steps:
                    ls_scores.append(runtime.score(m0, a0 + float(ls_step) * direction, tid, mode=mode))
                ls_stack = torch.stack(ls_scores, dim=0) if ls_scores else score1.unsqueeze(0)
                ls_delta_stack = ls_stack - score0.unsqueeze(0)
                best_delta, best_idx = ls_delta_stack.max(dim=0)
                accepted = best_delta > 0
                accepted_delta = torch.where(accepted, best_delta, torch.zeros_like(best_delta))
                line_search_step_tensor = torch.tensor(line_search_steps or [step_size], device=runtime.device)
                best_step = line_search_step_tensor[best_idx]
                accepted_step = torch.where(accepted, best_step, torch.zeros_like(best_step))
            rows[str(task_name)] = {
                "n": int(len(sub)),
                "score_before": summarize(score0.detach().cpu().numpy()),
                "score_after_action_step": summarize(score1.detach().cpu().numpy()),
                "score_delta": summarize(delta.detach().cpu().numpy()),
                "score_delta_mean": float(delta.detach().cpu().mean()),
                "improved_rate": float((delta > 0).detach().cpu().float().mean()),
                "line_search": {
                    "steps": [float(x) for x in (line_search_steps or [step_size])],
                    "accepted_rate": float(accepted.detach().cpu().float().mean()),
                    "best_delta": summarize(best_delta.detach().cpu().numpy()),
                    "accepted_delta": summarize(accepted_delta.detach().cpu().numpy()),
                    "accepted_step": summarize(accepted_step.detach().cpu().numpy()),
                    "mean_accepted_step": float(accepted_step.detach().cpu().mean()),
                },
                "grad_action_rms": summarize(grad_rms.detach().cpu().numpy()),
                "finite_grad_rate": float(finite.detach().cpu().float().mean()),
                "nonzero_grad_rate": float((grad_norm > 1e-10).detach().cpu().float().mean()),
            }
        result["modes"][mode] = rows
    return result


def guidance_passes(score_row: Dict[str, Any], grad_row: Dict[str, Any], args: argparse.Namespace) -> bool:
    mixed = grad_row.get("mixed", {})
    line_search = mixed.get("line_search", {})
    improved = max(float(mixed.get("improved_rate") or 0.0), float(line_search.get("accepted_rate") or 0.0))
    return bool(
        (score_row.get("binary_auc", 0.0) or 0.0) >= args.min_auc
        and (score_row.get("corr_with_quality", 0.0) or 0.0) >= args.min_corr
        and float(mixed.get("finite_grad_rate") or 0.0) >= args.min_finite_grad_rate
        and float(mixed.get("nonzero_grad_rate") or 0.0) >= args.min_nonzero_grad_rate
        and improved >= args.min_improved_rate
    )


def mode_objective(score_row: Dict[str, Any], grad_row: Dict[str, Any]) -> float:
    auc = float(score_row.get("binary_auc") or 0.0)
    corr_score = max(float(score_row.get("corr_with_quality") or 0.0), 0.0)
    margin = float(score_row.get("good_minus_bad_mean") or 0.0)
    mixed = grad_row.get("mixed", {})
    line_search = mixed.get("line_search", {})
    improved = max(float(mixed.get("improved_rate") or 0.0), float(line_search.get("accepted_rate") or 0.0))
    finite = float(mixed.get("finite_grad_rate") or 0.0)
    nonzero = float(mixed.get("nonzero_grad_rate") or 0.0)
    range_stat = score_row.get("summary", {})
    score_range = float(range_stat.get("p95", 0.0) - range_stat.get("p05", 0.0))
    range_score = min(score_range / 4.0, 1.0)
    return (
        0.30 * auc
        + 0.20 * corr_score
        + 0.15 * min(max(margin, 0.0) / 2.0, 1.0)
        + 0.20 * improved
        + 0.10 * finite
        + 0.05 * nonzero
        + 0.05 * range_score
    )


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(args.features, allow_pickle=True)
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    task_id = data["task_id"].astype(np.int64)
    task = data["task"]
    y_binary = data["y_binary"].astype(np.int64)
    y_t4 = data["y_t4"].astype(np.int64)
    quality = data["score"].astype(np.float32)

    runtime = ActionAwareScorerRuntime(str(args.checkpoint), device=args.device)
    scores = forward_scores(runtime, marker, action, task_id, args.batch_size)
    grad = gradient_probe(
        runtime,
        marker,
        action,
        task_id,
        task,
        y_binary,
        MODES,
        args.grad_n_per_task_class,
        args.grad_step_size,
        args.line_search_steps,
        args.seed,
    )
    mode_rows = {}
    for mode in MODES:
        score_row = mode_metrics(scores[mode], y_binary, quality)
        grad_row = grad["modes"].get(mode, {})
        mode_rows[mode] = {
            "score_metrics": score_row,
            "gradient_probe": grad_row,
            "objective": mode_objective(score_row, grad_row),
        }
    for mode in MODES:
        mode_rows[mode]["passes_guidance_constraints"] = guidance_passes(
            mode_rows[mode]["score_metrics"],
            mode_rows[mode]["gradient_probe"],
            args,
        )
    passing_modes = [mode for mode in MODES if mode_rows[mode]["passes_guidance_constraints"]]
    candidate_modes = passing_modes or list(MODES)
    recommended = sorted(candidate_modes, key=lambda name: mode_rows[name]["objective"], reverse=True)[0]
    per_task = {}
    for task_name in sorted(np.unique(task).tolist()):
        idx = task == task_name
        per_task[str(task_name)] = {
            "n": int(idx.sum()),
            "binary_counts_with_neutral": {
                str(k): int(v) for k, v in zip(*np.unique(y_binary[idx], return_counts=True))
            },
            "t4_counts": {str(k): int(v) for k, v in zip(*np.unique(y_t4[idx], return_counts=True))},
            "recommended_score_metrics": mode_metrics(scores[recommended][idx], y_binary[idx], quality[idx]),
        }
    return {
        "name": "ActionAware guidance suitability",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "config": {
            "features": str(args.features),
            "checkpoint": str(args.checkpoint),
            "episode_eval": str(args.episode_eval),
            "batch_size": int(args.batch_size),
            "grad_n_per_task_class": int(args.grad_n_per_task_class),
            "grad_step_size": float(args.grad_step_size),
            "line_search_steps": [float(x) for x in args.line_search_steps],
            "device": str(runtime.device),
        },
        "data": {
            "n": int(len(marker)),
            "task_counts": {str(k): int(v) for k, v in zip(*np.unique(task, return_counts=True))},
            "binary_counts_with_neutral": {
                str(k): int(v) for k, v in zip(*np.unique(y_binary, return_counts=True))
            },
        },
        "score_probability_saturation": {
            "p_good": saturation(scores["p_good"]),
            "quality_prob": saturation(scores["quality_prob"]),
        },
        "modes": mode_rows,
        "per_task": per_task,
        "recommended_mode": recommended,
        "passes_guidance_suitability": bool(
            mode_rows[recommended]["passes_guidance_constraints"]
        ),
        "interpretation": (
            "This validates ActionAware as a differentiable offline candidate. "
            "It does not override the earlier cross-task caveat or replace real rollout gates."
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# ActionAware Guidance Suitability",
        "",
        f"- passes_guidance_suitability: `{result['passes_guidance_suitability']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- recommended_mode: `{result['recommended_mode']}`",
        f"- git_commit: `{result['git_commit']}`",
        "",
        "## Modes",
        "",
        "| mode | objective | AUC | quality corr | mixed improved | line-search accepted | finite grad | nonzero grad |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in result["modes"].items():
        score = row["score_metrics"]
        grad = row["gradient_probe"].get("mixed", {})
        line_search = grad.get("line_search", {})
        lines.append(
            f"| {mode} | {row['objective']:.4f} | "
            f"{(score.get('binary_auc') or 0.0):.4f} | "
            f"{(score.get('corr_with_quality') or 0.0):.4f} | "
            f"{(grad.get('improved_rate') or 0.0):.4f} | "
            f"{(line_search.get('accepted_rate') or 0.0):.4f} | "
            f"{(grad.get('finite_grad_rate') or 0.0):.4f} | "
            f"{(grad.get('nonzero_grad_rate') or 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Saturation",
            "",
            f"- p_good saturated total: `{result['score_probability_saturation']['p_good']['prob_saturated_total']}`",
            f"- quality saturated total: `{result['score_probability_saturation']['quality_prob']['prob_saturated_total']}`",
            "",
            "## Interpretation",
            "",
            result["interpretation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--episode_eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--grad_n_per_task_class", type=int, default=128)
    parser.add_argument("--grad_step_size", type=float, default=0.02)
    parser.add_argument("--line_search_steps", type=float, nargs="+", default=[0.0005, 0.001, 0.002, 0.005, 0.01])
    parser.add_argument("--min_auc", type=float, default=0.95)
    parser.add_argument("--min_corr", type=float, default=0.70)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.99)
    parser.add_argument("--min_nonzero_grad_rate", type=float, default=0.99)
    parser.add_argument("--min_improved_rate", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_report(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "action_aware_guidance_suitability.json"
    md_path = args.output_dir / "action_aware_guidance_suitability.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    short = {
        "passes_guidance_suitability": result["passes_guidance_suitability"],
        "recommended_mode": result["recommended_mode"],
        "modes": {
            mode: {
                "objective": row["objective"],
                "auc": row["score_metrics"].get("binary_auc"),
                "quality_corr": row["score_metrics"].get("corr_with_quality"),
                "mixed_improved": row["gradient_probe"].get("mixed", {}).get("improved_rate"),
                "mixed_line_search_accepted": row["gradient_probe"].get("mixed", {})
                .get("line_search", {})
                .get("accepted_rate"),
            }
            for mode, row in result["modes"].items()
        },
        "json": str(json_path),
        "markdown": str(md_path),
    }
    print(json.dumps(short, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
