"""Compare distilled TacQualityEnergy against PTGProxyV2 for local guidance.

This is a feature/action-level guidance test, not a robot rollout.  It uses the
same cached proxy features that train both scorers and asks a deployment-facing
question:

  If we take a small trust-region step along d energy / d feature, does the
  scorer improve consistently, and how does the distilled energy compare with
  the previous PTGProxyV2 energy?

The distilled model is intended as a differentiable proxy for an RF teacher.
This script verifies its local guidance behavior under the same episode-derived
sample distribution used in GroupKFold evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import DistilledTacQualityEnergyRuntime  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime  # noqa: E402


FEATURES = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")
OUT_DIR = Path("/home/chenshuai/Project/output/distilled_energy_guidance_comparison")


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


def balanced_sample_indices(reason: np.ndarray, task: np.ndarray, n_per_task_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for task_name in np.unique(task):
        for cls in np.unique(reason):
            idx = np.flatnonzero((task == task_name) & (reason == cls))
            if len(idx) == 0:
                continue
            keep.extend(rng.choice(idx, min(n_per_task_class, len(idx)), replace=False).tolist())
    keep_arr = np.asarray(sorted(keep), dtype=np.int64)
    if len(keep_arr) == 0:
        raise RuntimeError("No samples selected")
    return keep_arr


def project(proposal: torch.Tensor, base: torch.Tensor, max_total_delta: float) -> torch.Tensor:
    if max_total_delta <= 0:
        return proposal
    delta = proposal - base
    norm = delta.flatten(1).norm(dim=1).view(-1, 1).clamp_min(1e-8)
    scale = torch.clamp(max_total_delta / norm, max=1.0)
    return base + delta * scale


def unit_grad(x: torch.Tensor, score: torch.Tensor):
    grad = torch.autograd.grad(score.sum(), x, retain_graph=False)[0]
    norm = grad.flatten(1).norm(dim=1)
    unit = grad / norm.view(-1, 1).clamp_min(1e-8)
    return unit, norm


def action_smoothness_proxy(X: np.ndarray) -> np.ndarray:
    """Proxy smoothness from the last 20 dims: eef(10)+joint(10) action features."""
    action_feat = X[:, -20:]
    return np.linalg.norm(action_feat[:, [5, 6, 15, 16]], axis=1)


def ptg_score_from_features(runtime: PTGProxyScorerV2Runtime, feat_norm: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    raw = runtime.model(feat_norm, task_id)
    q = raw["quality"]
    good_margin = raw["binary_logits"][:, 1] - raw["binary_logits"][:, 0]
    bad_reason_logits = torch.stack(
        [raw["reason_logits"][:, 0], torch.logsumexp(raw["reason_logits"][:, 2:], dim=-1)],
        dim=-1,
    )
    reason_margin = raw["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits, dim=-1)
    energy = 0.75 * q + 0.10 * good_margin + 0.0 * reason_margin
    return torch.tanh(energy / 4.0) * 4.0


def distilled_score_from_features(runtime: DistilledTacQualityEnergyRuntime, feat_norm: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
    return runtime.model(feat_norm, task_id)["energy_clipped"]


def eval_one_runtime(
    name: str,
    score_fn,
    feat_norm_np: np.ndarray,
    raw_feat_np: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    task_id_np: np.ndarray,
    scales: List[float],
    max_total_delta: float,
    device: torch.device,
) -> Dict[str, Any]:
    base_raw = np.asarray(raw_feat_np, dtype=np.float32)
    base_smooth = action_smoothness_proxy(base_raw)
    base = torch.tensor(feat_norm_np, dtype=torch.float32, device=device, requires_grad=True)
    task_id = torch.tensor(task_id_np, dtype=torch.long, device=device)
    base_score = score_fn(base, task_id)
    grad_unit, grad_norm = unit_grad(base, base_score)
    rows = []
    prev_mean = None
    for scale in scales:
        with torch.no_grad():
            proposal = project(base.detach() + float(scale) * grad_unit.detach(), base.detach(), max_total_delta)
        proposal_req = proposal.detach().clone().requires_grad_(True)
        score_new = score_fn(proposal_req, task_id).detach()
        score_delta = (score_new - base_score.detach()).cpu().numpy()
        delta_norm = (proposal - base.detach()).flatten(1).norm(dim=1).detach().cpu().numpy()
        proposal_np = proposal.detach().cpu().numpy() * scaler_scale.reshape(1, -1) + scaler_mean.reshape(1, -1)
        smooth_delta = action_smoothness_proxy(proposal_np) - base_smooth
        monotonic = True if prev_mean is None else float(score_delta.mean()) >= prev_mean - 1e-8
        prev_mean = float(score_delta.mean())
        rows.append(
            {
                "scale": float(scale),
                "score_delta": summarize(score_delta),
                "improved_rate": float(np.mean(score_delta > 0)),
                "delta_norm": summarize(delta_norm),
                "smoothness_proxy_delta": summarize(smooth_delta),
                "within_trust_region": bool(delta_norm.max() <= max_total_delta + 1e-6),
                "monotonic_score_delta_mean_vs_previous_scale": bool(monotonic),
            }
        )
    best = max(rows, key=lambda r: (r["improved_rate"], r["score_delta"]["mean"], -r["delta_norm"]["mean"]))
    return {
        "name": name,
        "n": int(len(base_raw)),
        "base_score": summarize(base_score.detach().cpu().numpy()),
        "gradient": {
            "finite_rate": float(torch.isfinite(grad_unit).flatten(1).all(dim=1).float().mean().detach().cpu()),
            "positive_norm_rate": float((grad_norm > 1e-8).float().mean().detach().cpu()),
            "grad_norm": summarize(grad_norm.detach().cpu().numpy()),
        },
        "max_total_delta": float(max_total_delta),
        "scale_rows": rows,
        "recommended_scale": float(best["scale"]),
        "recommended_score_delta_mean": float(best["score_delta"]["mean"]),
        "recommended_improved_rate": float(best["improved_rate"]),
        "passes_local_guidance": bool(
            all(r["within_trust_region"] for r in rows)
            and best["improved_rate"] >= 0.95
            and torch.isfinite(grad_unit).all().item()
            and float((grad_norm > 1e-8).float().mean().detach().cpu()) >= 0.999
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Distilled Energy Guidance Comparison",
        "",
        "Feature-level local guidance comparison between PTGProxyV2 and distilled TacQualityEnergy.",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- n_samples: `{result['data']['n']}`",
        "",
        "| scorer | pass | recommended scale | improved rate | score delta mean | grad norm mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["ptg_proxy_v2", "distilled_energy"]:
        row = result[name]
        lines.append(
            f"| {name} | {row['passes_local_guidance']} | {row['recommended_scale']} | "
            f"{row['recommended_improved_rate']:.4f} | {row['recommended_score_delta_mean']:.6f} | "
            f"{row['gradient']['grad_norm']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This test is not a robot rollout and does not prove policy improvement.",
            "- It verifies whether each scorer provides finite, locally useful gradients under the same feature schema.",
            "- Distilled energy remains a candidate only if it passes local guidance and later trust-region / rollout gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_scales(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(FEATURES))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--n_per_task_class", type=int, default=300)
    parser.add_argument("--scales", default="0.005,0.01,0.02,0.04,0.08,0.12")
    parser.add_argument("--max_total_delta", type=float, default=0.08)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.features, allow_pickle=True)
    X = data["X"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality = data["quality"].astype(np.float32)
    task = data["task"]
    task_id = data["task_id"].astype(np.int64)
    keep = balanced_sample_indices(reason, task, args.n_per_task_class, args.seed)
    X = X[keep]
    reason = reason[keep]
    binary = binary[keep]
    quality = quality[keep]
    task = task[keep]
    task_id = task_id[keep]

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    ptg = PTGProxyScorerV2Runtime(device=device_name)
    distilled = DistilledTacQualityEnergyRuntime(device=device_name)
    feat = torch.tensor(X, dtype=torch.float32, device=device)
    ptg_norm = ((feat - ptg.scaler_mean.view(1, -1)) / (ptg.scaler_scale.view(1, -1) + 1e-8)).detach().cpu().numpy()
    dist_norm = (
        (feat - distilled.scaler_mean.view(1, -1)) / (distilled.scaler_scale.view(1, -1) + 1e-8)
    ).detach().cpu().numpy()
    scales = parse_scales(args.scales)
    result = {
        "purpose": "Same-protocol local feature-gradient comparison for classifier guidance potentials.",
        "scope": "Feature-level guidance, not Foresight/DP full-chain and not robot rollout.",
        "data": {
            "source": str(args.features),
            "n": int(len(X)),
            "feature_dim": int(X.shape[1]),
            "task_counts": {str(k): int(v) for k, v in Counter(task.tolist()).items()},
            "reason_counts": {str(k): int(v) for k, v in Counter(reason.tolist()).items()},
            "binary_counts_with_neutral": {str(k): int(v) for k, v in Counter(binary.tolist()).items()},
            "quality": summarize(quality),
        },
        "config": vars(args),
        "ptg_proxy_v2": eval_one_runtime(
            "ptg_proxy_v2",
            lambda z, tid: ptg_score_from_features(ptg, z, tid),
            ptg_norm,
            X,
            ptg.scaler_mean.detach().cpu().numpy(),
            ptg.scaler_scale.detach().cpu().numpy(),
            task_id,
            scales,
            args.max_total_delta,
            device,
        ),
        "distilled_energy": eval_one_runtime(
            "distilled_energy",
            lambda z, tid: distilled_score_from_features(distilled, z, tid),
            dist_norm,
            X,
            distilled.scaler_mean.detach().cpu().numpy(),
            distilled.scaler_scale.detach().cpu().numpy(),
            task_id,
            scales,
            args.max_total_delta,
            device,
        ),
    }
    result["overall_pass"] = bool(result["ptg_proxy_v2"]["passes_local_guidance"] and result["distilled_energy"]["passes_local_guidance"])
    result["recommendation"] = (
        "Distilled energy is suitable for the next trust-region/action-level gate"
        if result["distilled_energy"]["passes_local_guidance"]
        else "Do not promote distilled energy until local guidance issues are fixed"
    )
    json_path = out_dir / "distilled_energy_guidance_comparison.json"
    md_path = out_dir / "distilled_energy_guidance_comparison.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "ptg": {
                    "pass": result["ptg_proxy_v2"]["passes_local_guidance"],
                    "scale": result["ptg_proxy_v2"]["recommended_scale"],
                    "improved": result["ptg_proxy_v2"]["recommended_improved_rate"],
                    "delta": result["ptg_proxy_v2"]["recommended_score_delta_mean"],
                },
                "distilled": {
                    "pass": result["distilled_energy"]["passes_local_guidance"],
                    "scale": result["distilled_energy"]["recommended_scale"],
                    "improved": result["distilled_energy"]["recommended_improved_rate"],
                    "delta": result["distilled_energy"]["recommended_score_delta_mean"],
                },
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
