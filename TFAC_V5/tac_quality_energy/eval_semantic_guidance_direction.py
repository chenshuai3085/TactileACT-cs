#!/usr/bin/env python3
"""Audit semantic score directions for TacQuality DP guidance.

This audit complements local gradient checks.  A scorer can be differentiable
and still point toward the wrong physical semantics.  Here we test whether the
current insertion and board scorers increase along bad-contact -> good-contact
directions and decrease along good-contact -> bad-contact directions.

The result is still offline evidence only: it does not prove robot improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.force_band_runtime import ForceBandTacQualityEnergy, REASON_TO_ID  # noqa: E402
from TFAC_V5.tac_quality_energy.insertion_runtime import InsertionRiskScorerRuntime  # noqa: E402


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_semantic_direction_audit")
DEFAULT_INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
DEFAULT_INSERTION_CKPT = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
DEFAULT_BOARD_DEFAULT_FEATURES = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260618_deploy/"
    "board_predicted_domain_force_band_features_deploy.npz"
)
DEFAULT_BOARD_DEFAULT_CKPT = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/"
    "force_band_tac_quality_energy_best.pt"
)
DEFAULT_BOARD_S12_FEATURES = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_features_20260619_s12_deploy/"
    "board_predicted_domain_force_band_features_deploy.npz"
)
DEFAULT_BOARD_S12_CKPT = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "force_band_tac_quality_energy_best.pt"
)


INSERTION_REASON_NAMES = {
    1: "good_insert",
    2: "pre_bounce_risk",
    3: "impact_or_recovery",
}
BOARD_REASON_NAMES = {
    0: "too_small",
    1: "positive",
    2: "too_large",
    3: "oscillate",
}


def as_device(device_name: str) -> torch.device:
    if device_name != "cpu" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def summarize(values: np.ndarray | list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
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


def fmt(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{ndigits}f}"


def sample_indices(rng: np.random.Generator, idx: np.ndarray, limit: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) <= limit:
        return idx
    return np.sort(rng.choice(idx, size=limit, replace=False))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def board_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[ForceBandTacQualityEnergy, torch.Tensor, torch.Tensor, str]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ForceBandTacQualityEnergy(
        in_dim=int(ckpt["feature_dim"]),
        hidden=int(ckpt.get("hidden", 192)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    mean = torch.as_tensor(ckpt["scaler_mean"], dtype=torch.float32, device=device)
    scale = torch.as_tensor(ckpt["scaler_scale"], dtype=torch.float32, device=device).clamp_min(1e-8)
    return model, mean, scale, str(ckpt.get("feature_variant", "marker_joint_action"))


def board_scores_from_norm(model: ForceBandTacQualityEnergy, x_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    out = model(x_norm)
    p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1]
    reason_prob = torch.softmax(out["reason_logits"], dim=-1)
    return {
        "quality": torch.sigmoid(out["quality_logit"]),
        "p_good": p_good,
        "reason_good": reason_prob[:, REASON_TO_ID["positive"]],
        "energy_clipped": out["energy_clipped"],
        "profile": 0.5 * out["quality_logit"] + 0.25 * out["good_margin"] + 0.25 * out["reason_margin"],
    }


def insertion_scores(runtime: InsertionRiskScorerRuntime, marker: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
    out = runtime.forward(marker, joint_action_seq=action)
    return {
        "quality": out["quality_score"],
        "p_good": out["p_good"],
        "neg_risk": -out["risk_prob"],
        "energy_clipped": out["energy_clipped"],
        "profile": runtime.profile_score(marker, joint_action_seq=action),
    }


def curve_stats(scores_by_alpha: np.ndarray, *, expect: str) -> dict[str, Any]:
    diffs = np.diff(scores_by_alpha, axis=1)
    endpoint = scores_by_alpha[:, -1] - scores_by_alpha[:, 0]
    if expect == "increase":
        step_ok = diffs >= -1e-8
        endpoint_ok = endpoint > 0.0
    elif expect == "decrease":
        step_ok = diffs <= 1e-8
        endpoint_ok = endpoint < 0.0
    else:
        raise ValueError(expect)
    return {
        "expect": expect,
        "n": int(scores_by_alpha.shape[0]),
        "mean_curve": [float(x) for x in scores_by_alpha.mean(axis=0)],
        "std_curve": [float(x) for x in scores_by_alpha.std(axis=0)],
        "step_direction_rate": float(step_ok.mean()) if step_ok.size else None,
        "endpoint_direction_rate": float(endpoint_ok.mean()) if endpoint_ok.size else None,
        "endpoint_delta": summarize(endpoint),
    }


def plot_task_curves(task_name: str, mode: str, alphas: np.ndarray, rows: list[dict[str, Any]], out_path: Path) -> None:
    selected = [r for r in rows if r["mode"] == mode and r["direction"] == "bad_to_good"]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    for row in selected:
        y = np.asarray(row["mean_curve"], dtype=np.float64)
        ax.plot(alphas, y, marker="o", linewidth=1.8, label=row["bad_reason_name"])
    ax.set_title(f"{task_name}: bad-to-good semantic score curves ({mode})")
    ax.set_xlabel("interpolation alpha: bad sample -> good centroid")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def evaluate_board_case(
    *,
    name: str,
    features_path: Path,
    ckpt_path: Path,
    device: torch.device,
    rng: np.random.Generator,
    max_samples_per_class: int,
    alphas: np.ndarray,
    guidance_mode: str,
    out_dir: Path,
) -> dict[str, Any]:
    data = np.load(features_path, allow_pickle=True)
    model, mean, scale, feature_variant = board_model_from_ckpt(ckpt_path, device)
    if feature_variant not in data:
        raise KeyError(f"{features_path} lacks feature variant {feature_variant}")
    x = data[feature_variant].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality_target = data["quality"].astype(np.float32)

    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        scores = board_scores_from_norm(model, (xt - mean.view(1, -1)) / scale.view(1, -1))
        score_np = {k: v.detach().cpu().numpy() for k, v in scores.items()}

    good_idx = np.where(reason == 1)[0]
    good_sel = sample_indices(rng, good_idx, max_samples_per_class)
    bad_reasons = [0, 2, 3]
    centroids = {rid: x[reason == rid].mean(axis=0) for rid in [1, *bad_reasons]}
    score_by_reason = {}
    for rid in [1, *bad_reasons]:
        mask = reason == rid
        score_by_reason[BOARD_REASON_NAMES[rid]] = {
            "n": int(mask.sum()),
            "quality_target": summarize(quality_target[mask]),
            "scores": {mode: summarize(values[mask]) for mode, values in score_np.items()},
        }

    curve_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for bad_reason in bad_reasons:
        bad_idx = np.where(reason == bad_reason)[0]
        bad_sel = sample_indices(rng, bad_idx, max_samples_per_class)
        bad_to_good = centroids[1] - x[bad_sel]
        good_to_bad = centroids[bad_reason] - x[good_sel]

        for direction_name, base_idx, direction, expect in [
            ("bad_to_good", bad_sel, bad_to_good, "increase"),
            ("good_to_bad", good_sel, good_to_bad, "decrease"),
        ]:
            for mode in score_np:
                curves = []
                for alpha in alphas:
                    xx = x[base_idx] + float(alpha) * direction
                    with torch.no_grad():
                        xt = torch.from_numpy(xx.astype(np.float32)).to(device)
                        ss = board_scores_from_norm(model, (xt - mean.view(1, -1)) / scale.view(1, -1))[mode]
                    curves.append(ss.detach().cpu().numpy())
                scores_by_alpha = np.stack(curves, axis=1)
                stats = curve_stats(scores_by_alpha, expect=expect)
                curve_rows.append(
                    {
                        "task": name,
                        "mode": mode,
                        "direction": direction_name,
                        "bad_reason": int(bad_reason),
                        "bad_reason_name": BOARD_REASON_NAMES[bad_reason],
                        **stats,
                    }
                )

            base = torch.from_numpy(x[base_idx].astype(np.float32)).to(device)
            direction_t = torch.from_numpy(direction.astype(np.float32)).to(device)
            for mode in score_np:
                base_req = base.detach().clone().requires_grad_(True)
                score = board_scores_from_norm(model, (base_req - mean.view(1, -1)) / scale.view(1, -1))[mode].mean()
                grad = torch.autograd.grad(score, base_req, retain_graph=False)[0]
                dot = (grad * direction_t).flatten(1).sum(dim=1)
                denom = grad.flatten(1).norm(dim=1).clamp_min(1e-8) * direction_t.flatten(1).norm(dim=1).clamp_min(1e-8)
                cosine = (dot / denom).detach().cpu().numpy()
                dot_np = dot.detach().cpu().numpy()
                sign_ok = dot_np > 0 if expect == "increase" else dot_np < 0
                gradient_rows.append(
                    {
                        "task": name,
                        "mode": mode,
                        "direction": direction_name,
                        "bad_reason": int(bad_reason),
                        "bad_reason_name": BOARD_REASON_NAMES[bad_reason],
                        "expect": expect,
                        "semantic_projection_rate": float(sign_ok.mean()),
                        "dot": summarize(dot_np),
                        "cosine": summarize(cosine),
                        "grad_norm": summarize(grad.flatten(1).norm(dim=1).detach().cpu().numpy()),
                    }
                )

    return {
        "name": name,
        "kind": "board",
        "features": str(features_path),
        "checkpoint": str(ckpt_path),
        "feature_variant": feature_variant,
        "n": int(len(x)),
        "label_counts": {BOARD_REASON_NAMES[k]: int((reason == k).sum()) for k in [0, 1, 2, 3]},
        "guidance_mode": guidance_mode,
        "score_by_reason": score_by_reason,
        "curves": curve_rows,
        "gradients": gradient_rows,
    }


def evaluate_insertion(
    *,
    features_path: Path,
    ckpt_path: Path,
    device: torch.device,
    rng: np.random.Generator,
    max_samples_per_class: int,
    alphas: np.ndarray,
    guidance_mode: str,
    out_dir: Path,
) -> dict[str, Any]:
    data = np.load(features_path, allow_pickle=True)
    runtime = InsertionRiskScorerRuntime(str(ckpt_path), device=str(device))
    marker = data["marker"].astype(np.float32)
    action = data["action"].astype(np.float32)
    reason = data["reason"].astype(np.int64)
    binary = data["binary"].astype(np.int64)
    quality_target = data["quality"].astype(np.float32)

    score_np: dict[str, list[np.ndarray]] = {k: [] for k in ["quality", "p_good", "neg_risk", "energy_clipped", "profile"]}
    with torch.no_grad():
        for start in range(0, len(marker), 512):
            mt = torch.from_numpy(marker[start : start + 512]).to(device)
            at = torch.from_numpy(action[start : start + 512]).to(device)
            out = insertion_scores(runtime, mt, at)
            for mode, value in out.items():
                score_np[mode].append(value.detach().cpu().numpy())
    score_np = {k: np.concatenate(v) for k, v in score_np.items()}

    # reason 0 is weak approach and is intentionally excluded from bad-contact
    # semantics because it is not the user's bounce/failure definition.
    good_idx = np.where(reason == 1)[0]
    good_sel = sample_indices(rng, good_idx, max_samples_per_class)
    bad_reasons = [2, 3]
    marker_centroids = {rid: marker[reason == rid].mean(axis=0) for rid in [1, *bad_reasons]}
    action_centroids = {rid: action[reason == rid].mean(axis=0) for rid in [1, *bad_reasons]}

    score_by_reason = {}
    for rid in [1, *bad_reasons]:
        mask = reason == rid
        score_by_reason[INSERTION_REASON_NAMES[rid]] = {
            "n": int(mask.sum()),
            "quality_target": summarize(quality_target[mask]),
            "scores": {mode: summarize(values[mask]) for mode, values in score_np.items()},
        }

    curve_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for bad_reason in bad_reasons:
        bad_idx = np.where(reason == bad_reason)[0]
        bad_sel = sample_indices(rng, bad_idx, max_samples_per_class)
        directions = {
            "bad_to_good": (
                bad_sel,
                marker_centroids[1] - marker[bad_sel],
                action_centroids[1] - action[bad_sel],
                "increase",
            ),
            "good_to_bad": (
                good_sel,
                marker_centroids[bad_reason] - marker[good_sel],
                action_centroids[bad_reason] - action[good_sel],
                "decrease",
            ),
        }
        for direction_name, (base_idx, marker_dir, action_dir, expect) in directions.items():
            for mode in score_np:
                curves = []
                for alpha in alphas:
                    mm = marker[base_idx] + float(alpha) * marker_dir
                    aa = action[base_idx] + float(alpha) * action_dir
                    with torch.no_grad():
                        mt = torch.from_numpy(mm.astype(np.float32)).to(device)
                        at = torch.from_numpy(aa.astype(np.float32)).to(device)
                        ss = insertion_scores(runtime, mt, at)[mode]
                    curves.append(ss.detach().cpu().numpy())
                scores_by_alpha = np.stack(curves, axis=1)
                stats = curve_stats(scores_by_alpha, expect=expect)
                curve_rows.append(
                    {
                        "task": "insertion",
                        "mode": mode,
                        "direction": direction_name,
                        "bad_reason": int(bad_reason),
                        "bad_reason_name": INSERTION_REASON_NAMES[bad_reason],
                        **stats,
                    }
                )

            mt_dir = torch.from_numpy(marker_dir.astype(np.float32)).to(device)
            at_dir = torch.from_numpy(action_dir.astype(np.float32)).to(device)
            for mode in score_np:
                mt_base = torch.from_numpy(marker[base_idx].astype(np.float32)).to(device).detach().clone().requires_grad_(True)
                at_base = torch.from_numpy(action[base_idx].astype(np.float32)).to(device).detach().clone().requires_grad_(True)
                score = insertion_scores(runtime, mt_base, at_base)[mode].mean()
                grad_m, grad_a = torch.autograd.grad(score, [mt_base, at_base], retain_graph=False)
                dot = (grad_m * mt_dir).flatten(1).sum(dim=1) + (grad_a * at_dir).flatten(1).sum(dim=1)
                grad_flat = torch.cat([grad_m.flatten(1), grad_a.flatten(1)], dim=1)
                dir_flat = torch.cat([mt_dir.flatten(1), at_dir.flatten(1)], dim=1)
                denom = grad_flat.norm(dim=1).clamp_min(1e-8) * dir_flat.norm(dim=1).clamp_min(1e-8)
                cosine = (dot / denom).detach().cpu().numpy()
                dot_np = dot.detach().cpu().numpy()
                sign_ok = dot_np > 0 if expect == "increase" else dot_np < 0
                gradient_rows.append(
                    {
                        "task": "insertion",
                        "mode": mode,
                        "direction": direction_name,
                        "bad_reason": int(bad_reason),
                        "bad_reason_name": INSERTION_REASON_NAMES[bad_reason],
                        "expect": expect,
                        "semantic_projection_rate": float(sign_ok.mean()),
                        "dot": summarize(dot_np),
                        "cosine": summarize(cosine),
                        "grad_norm": summarize(grad_flat.norm(dim=1).detach().cpu().numpy()),
                    }
                )

    return {
        "name": "insertion",
        "kind": "insertion",
        "features": str(features_path),
        "checkpoint": str(ckpt_path),
        "n": int(len(marker)),
        "label_counts": {
            INSERTION_REASON_NAMES[1]: int((reason == 1).sum()),
            INSERTION_REASON_NAMES[2]: int((reason == 2).sum()),
            INSERTION_REASON_NAMES[3]: int((reason == 3).sum()),
            "weak_approach_excluded": int((reason == 0).sum()),
        },
        "guidance_mode": guidance_mode,
        "score_by_reason": score_by_reason,
        "curves": curve_rows,
        "gradients": gradient_rows,
    }


def flatten_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task_name, section in result["tasks"].items():
        for row in section["curves"]:
            endpoint = row["endpoint_delta"]
            rows.append(
                {
                    "task": task_name,
                    "type": "curve",
                    "mode": row["mode"],
                    "direction": row["direction"],
                    "bad_reason": row["bad_reason_name"],
                    "expect": row["expect"],
                    "step_direction_rate": row["step_direction_rate"],
                    "endpoint_direction_rate": row["endpoint_direction_rate"],
                    "endpoint_delta_mean": endpoint.get("mean"),
                    "endpoint_delta_p05": endpoint.get("p05"),
                    "semantic_projection_rate": None,
                    "cosine_mean": None,
                }
            )
        for row in section["gradients"]:
            rows.append(
                {
                    "task": task_name,
                    "type": "gradient",
                    "mode": row["mode"],
                    "direction": row["direction"],
                    "bad_reason": row["bad_reason_name"],
                    "expect": row["expect"],
                    "step_direction_rate": None,
                    "endpoint_direction_rate": None,
                    "endpoint_delta_mean": None,
                    "endpoint_delta_p05": None,
                    "semantic_projection_rate": row["semantic_projection_rate"],
                    "cosine_mean": row["cosine"].get("mean"),
                }
            )
    return rows


def mode_pass_summary(section: dict[str, Any], mode: str) -> dict[str, Any]:
    curve_checks = [r for r in section["curves"] if r["mode"] == mode]
    grad_checks = [r for r in section["gradients"] if r["mode"] == mode]
    curve_passes = []
    for row in curve_checks:
        endpoint = row["endpoint_delta"].get("mean", 0.0)
        if row["expect"] == "increase":
            signed_mean_ok = endpoint > 0.0
        else:
            signed_mean_ok = endpoint < 0.0
        curve_passes.append(
            bool(
                signed_mean_ok
                and float(row["endpoint_direction_rate"] or 0.0) >= 0.70
                and float(row["step_direction_rate"] or 0.0) >= 0.60
            )
        )
    bad_to_good_grads = [r for r in grad_checks if r["direction"] == "bad_to_good"]
    good_to_bad_grads = [r for r in grad_checks if r["direction"] == "good_to_bad"]
    grad_passes = [float(r["semantic_projection_rate"]) >= 0.55 for r in grad_checks]
    bad_to_good_passes = [float(r["semantic_projection_rate"]) >= 0.55 for r in bad_to_good_grads]
    good_to_bad_passes = [float(r["semantic_projection_rate"]) >= 0.50 for r in good_to_bad_grads]
    bad_to_good_mean = float(np.mean([r["semantic_projection_rate"] for r in bad_to_good_grads])) if bad_to_good_grads else 0.0
    good_to_bad_mean = float(np.mean([r["semantic_projection_rate"] for r in good_to_bad_grads])) if good_to_bad_grads else 0.0
    curve_pass_rate = float(np.mean(curve_passes)) if curve_passes else 0.0
    strict_gradient_pass_rate = float(np.mean(grad_passes)) if grad_passes else 0.0
    correction_gradient_pass_rate = float(np.mean(bad_to_good_passes)) if bad_to_good_passes else 0.0
    protection_gradient_pass_rate = float(np.mean(good_to_bad_passes)) if good_to_bad_passes else 0.0
    correction_pass = bool(curve_passes and bad_to_good_passes and all(curve_passes) and all(bad_to_good_passes))
    strict_pass = bool(curve_passes and grad_passes and all(curve_passes) and all(grad_passes))
    score = (
        2.0 * curve_pass_rate
        + 2.0 * correction_gradient_pass_rate
        + 0.75 * strict_gradient_pass_rate
        + 0.25 * protection_gradient_pass_rate
        + bad_to_good_mean
        + 0.25 * good_to_bad_mean
    )
    return {
        "guidance_mode": mode,
        "n_curve_checks": len(curve_checks),
        "curve_pass_rate": curve_pass_rate,
        "n_gradient_checks": len(grad_checks),
        "strict_gradient_pass_rate": strict_gradient_pass_rate,
        "correction_gradient_pass_rate": correction_gradient_pass_rate,
        "protection_gradient_pass_rate": protection_gradient_pass_rate,
        "bad_to_good_projection_mean": bad_to_good_mean,
        "good_to_bad_projection_mean": good_to_bad_mean,
        "strict_pass": strict_pass,
        "correction_pass": correction_pass,
        "selection_score": float(score),
    }


def pass_summary(section: dict[str, Any]) -> dict[str, Any]:
    modes = sorted({r["mode"] for r in section["curves"]})
    by_mode = {mode: mode_pass_summary(section, mode) for mode in modes}
    recommended_mode = max(by_mode.items(), key=lambda item: item[1]["selection_score"])[0] if by_mode else section["guidance_mode"]
    deployed_mode = section["guidance_mode"]
    return {
        "deployed_guidance_mode": deployed_mode,
        "recommended_mode_by_semantic_direction": recommended_mode,
        "deployed": by_mode.get(deployed_mode, {}),
        "recommended": by_mode.get(recommended_mode, {}),
        "mode_comparison": by_mode,
        "strict_pass": bool(by_mode.get(deployed_mode, {}).get("strict_pass", False)),
        "correction_pass": bool(by_mode.get(deployed_mode, {}).get("correction_pass", False)),
    }


def render_md(result: dict[str, Any]) -> str:
    lines = [
        "# TacQuality Semantic Guidance Direction Audit",
        "",
        f"- created_at: `{result['created_at']}`",
        f"- device: `{result['device']}`",
        f"- evidence boundary: {result['evidence_boundary']}",
        "",
        "## Summary",
        "",
        "| task | deployed mode | semantic recommended mode | deployed correction pass | deployed strict pass | notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for task_name, summary in result["summary"].items():
        notes = "offline semantic direction only"
        lines.append(
            f"| {task_name} | `{summary['deployed_guidance_mode']}` | "
            f"`{summary['recommended_mode_by_semantic_direction']}` | "
            f"{fmt(summary['correction_pass'])} | {fmt(summary['strict_pass'])} | {notes} |"
        )
    lines.extend(
        [
            "",
            "## Score Mode Comparison",
            "",
            "| task | mode | curve pass | correction grad pass | protection grad pass | strict grad pass | bad-to-good projection | selection score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task_name, summary in result["summary"].items():
        for mode, payload in sorted(summary["mode_comparison"].items(), key=lambda item: -item[1]["selection_score"]):
            lines.append(
                f"| {task_name} | `{mode}` | {fmt(payload['curve_pass_rate'])} | "
                f"{fmt(payload['correction_gradient_pass_rate'])} | "
                f"{fmt(payload['protection_gradient_pass_rate'])} | "
                f"{fmt(payload['strict_gradient_pass_rate'])} | "
                f"{fmt(payload['bad_to_good_projection_mean'])} | "
                f"{fmt(payload['selection_score'])} |"
            )
    lines.extend(
        [
            "",
            "## Guidance-Mode Semantic Checks",
            "",
            "| task | direction | bad mode | expect | step rate | endpoint rate | endpoint mean | gradient projection | cosine mean |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task_name, section in result["tasks"].items():
        mode = result["summary"][task_name]["recommended_mode_by_semantic_direction"]
        curves = {(r["direction"], r["bad_reason_name"]): r for r in section["curves"] if r["mode"] == mode}
        grads = {(r["direction"], r["bad_reason_name"]): r for r in section["gradients"] if r["mode"] == mode}
        for key, curve in curves.items():
            grad = grads.get(key, {})
            lines.append(
                f"| {task_name} | {curve['direction']} | {curve['bad_reason_name']} | {curve['expect']} | "
                f"{fmt(curve['step_direction_rate'])} | {fmt(curve['endpoint_direction_rate'])} | "
                f"{fmt(curve['endpoint_delta'].get('mean'), 6)} | "
                f"{fmt(grad.get('semantic_projection_rate'))} | {fmt(getattr(grad.get('cosine', {}), 'get', lambda *_: None)('mean'))} |"
            )
    lines.extend(["", "## Score By Reason", ""])
    for task_name, section in result["tasks"].items():
        lines.extend([f"### {task_name}", "", "| reason | n | target quality mean | guidance score mean | p_good mean |", "|---|---:|---:|---:|---:|"])
        mode = result["summary"][task_name]["recommended_mode_by_semantic_direction"]
        for reason_name, payload in section["score_by_reason"].items():
            scores = payload["scores"]
            lines.append(
                f"| {reason_name} | {payload['n']} | {fmt(payload['quality_target'].get('mean'))} | "
                f"{fmt(scores[mode].get('mean'))} | {fmt(scores.get('p_good', {}).get('mean'))} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- `bad_to_good` interpolates real bad-contact samples toward the good-contact centroid; the guidance score should increase.",
            "- `good_to_bad` interpolates real good-contact samples toward each bad-contact centroid; the guidance score should decrease.",
            "- `gradient projection` checks whether the local score gradient points along the same semantic direction.",
            "- This audit verifies score geometry under saved data distributions. It is not a robot rollout and not a force-improvement claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--max_samples_per_class", type=int, default=192)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--insertion_features", type=Path, default=DEFAULT_INSERTION_FEATURES)
    parser.add_argument("--insertion_ckpt", type=Path, default=DEFAULT_INSERTION_CKPT)
    parser.add_argument("--insertion_guidance_mode", default="profile")
    parser.add_argument("--board_default_features", type=Path, default=DEFAULT_BOARD_DEFAULT_FEATURES)
    parser.add_argument("--board_default_ckpt", type=Path, default=DEFAULT_BOARD_DEFAULT_CKPT)
    parser.add_argument("--board_s12_features", type=Path, default=DEFAULT_BOARD_S12_FEATURES)
    parser.add_argument("--board_s12_ckpt", type=Path, default=DEFAULT_BOARD_S12_CKPT)
    parser.add_argument("--board_guidance_mode", default="quality")
    parser.add_argument("--skip_s12", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = as_device(args.device)
    rng = np.random.default_rng(args.seed)
    alphas = np.asarray([float(x) for x in str(args.alphas).split(",") if x.strip()], dtype=np.float32)

    import datetime as dt

    tasks: dict[str, Any] = {}
    tasks["insertion"] = evaluate_insertion(
        features_path=args.insertion_features,
        ckpt_path=args.insertion_ckpt,
        device=device,
        rng=rng,
        max_samples_per_class=args.max_samples_per_class,
        alphas=alphas,
        guidance_mode=args.insertion_guidance_mode,
        out_dir=out_dir,
    )
    tasks["board_default"] = evaluate_board_case(
        name="board_default",
        features_path=args.board_default_features,
        ckpt_path=args.board_default_ckpt,
        device=device,
        rng=rng,
        max_samples_per_class=args.max_samples_per_class,
        alphas=alphas,
        guidance_mode=args.board_guidance_mode,
        out_dir=out_dir,
    )
    if not args.skip_s12 and args.board_s12_features.exists() and args.board_s12_ckpt.exists():
        tasks["board_s12"] = evaluate_board_case(
            name="board_s12",
            features_path=args.board_s12_features,
            ckpt_path=args.board_s12_ckpt,
            device=device,
            rng=rng,
            max_samples_per_class=args.max_samples_per_class,
            alphas=alphas,
            guidance_mode=args.board_guidance_mode,
            out_dir=out_dir,
        )

    result = {
        "purpose": "Semantic direction audit for TacQuality DP classifier guidance.",
        "evidence_boundary": "Offline score-geometry audit only; no robot outcome is proven.",
        "created_at": f"{dt.datetime.now():%F %T}",
        "device": str(device),
        "seed": int(args.seed),
        "max_samples_per_class": int(args.max_samples_per_class),
        "alphas": [float(x) for x in alphas],
        "tasks": tasks,
        "summary": {name: pass_summary(section) for name, section in tasks.items()},
    }
    result["overall_correction_pass"] = bool(all(section["correction_pass"] for section in result["summary"].values()))
    result["overall_strict_pass"] = bool(all(section["strict_pass"] for section in result["summary"].values()))

    for task_name, section in tasks.items():
        plot_modes = {
            section["guidance_mode"],
            result["summary"][task_name]["recommended_mode_by_semantic_direction"],
        }
        for mode in sorted(plot_modes):
            plot_task_curves(task_name, mode, alphas, section["curves"], out_dir / f"{task_name}_{mode}_bad_to_good_curves.png")

    json_path = out_dir / "tac_quality_semantic_direction_audit.json"
    md_path = out_dir / "tac_quality_semantic_direction_audit.md"
    csv_path = out_dir / "tac_quality_semantic_direction_audit_rows.csv"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_md(result), encoding="utf-8")
    write_csv(csv_path, flatten_rows(result))
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
                "overall_correction_pass": result["overall_correction_pass"],
                "overall_strict_pass": result["overall_strict_pass"],
                "summary": result["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
