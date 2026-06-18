#!/usr/bin/env python3
"""Evaluate TacQuality scorer score ordering for DP guidance.

This script is the formal-package version of the score calibration audit.  It
does not retrain any scorer.  It loads saved feature caches and checkpoints,
then checks whether candidate guidance scores are ordered with the target
quality/good labels.  A good DP guidance potential should be monotonic with
quality, not merely accurate as a classifier.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.insertion_runtime import InsertionRiskScorerRuntime
from TFAC_V5.tac_quality_energy.ptg_proxy_runtime import PTGProxyScorerV2Runtime, PTGProxyScorerV2


DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_score_calibration_formal")
DEFAULT_INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
DEFAULT_INSERTION_CKPT = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
DEFAULT_PTG_FEATURES = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")
DEFAULT_PTG_CKPT = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")


def finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman_simple(x: np.ndarray, y: np.ndarray) -> float:
    return finite_corr(rankdata_simple(np.asarray(x)), rankdata_simple(np.asarray(y)))


def safe_auc(binary: np.ndarray, score: np.ndarray) -> float | None:
    mask = np.asarray(binary) >= 0
    if mask.sum() < 3 or len(np.unique(binary[mask])) < 2:
        return None
    return float(roc_auc_score(binary[mask], np.asarray(score)[mask]))


def decile_bins(score: np.ndarray, binary: np.ndarray, quality: np.ndarray, reason: np.ndarray, n_bins: int) -> list[dict[str, Any]]:
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    rows: list[dict[str, Any]] = []
    for i, idx in enumerate(np.array_split(order, n_bins)):
        if len(idx) == 0:
            continue
        valid_binary = binary[idx] >= 0
        reason_values, reason_counts = np.unique(reason[idx], return_counts=True)
        rows.append(
            {
                "bin": int(i),
                "n": int(len(idx)),
                "score_min": float(np.min(score[idx])),
                "score_max": float(np.max(score[idx])),
                "score_mean": float(np.mean(score[idx])),
                "good_rate": float(np.mean(binary[idx][valid_binary] == 1)) if np.any(valid_binary) else None,
                "quality_mean": float(np.mean(quality[idx])),
                "reason_counts": {str(k): int(v) for k, v in zip(reason_values, reason_counts)},
            }
        )
    return rows


def monotonic_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    quality = np.asarray([r["quality_mean"] for r in rows], dtype=np.float64)
    good = np.asarray([np.nan if r["good_rate"] is None else r["good_rate"] for r in rows], dtype=np.float64)
    q_steps = np.diff(quality)
    good = good[np.isfinite(good)]
    g_steps = np.diff(good)
    return {
        "quality_monotonic_non_decreasing": bool(np.all(q_steps >= -1e-8)),
        "quality_positive_step_rate": float(np.mean(q_steps >= -1e-8)) if len(q_steps) else None,
        "good_rate_monotonic_non_decreasing": bool(np.all(g_steps >= -1e-8)) if len(g_steps) else None,
        "good_rate_positive_step_rate": float(np.mean(g_steps >= -1e-8)) if len(g_steps) else None,
        "bottom_quality_mean": float(quality[0]) if len(quality) else None,
        "top_quality_mean": float(quality[-1]) if len(quality) else None,
        "top_bottom_quality_gap": float(quality[-1] - quality[0]) if len(quality) else None,
        "bottom_good_rate": float(good[0]) if len(good) else None,
        "top_good_rate": float(good[-1]) if len(good) else None,
        "top_bottom_good_rate_gap": float(good[-1] - good[0]) if len(good) else None,
    }


def write_bins_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin", "n", "score_min", "score_max", "score_mean", "good_rate", "quality_mean", "reason_counts"],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["reason_counts"] = json.dumps(out["reason_counts"], ensure_ascii=False)
            writer.writerow(out)


def plot_bins(rows_by_mode: dict[str, list[dict[str, Any]]], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for mode, rows in rows_by_mode.items():
        xs = np.asarray([r["bin"] for r in rows])
        q = np.asarray([r["quality_mean"] for r in rows])
        g = np.asarray([np.nan if r["good_rate"] is None else r["good_rate"] for r in rows])
        axes[0].plot(xs, q, marker="o", linewidth=1.6, label=mode)
        if np.isfinite(g).any():
            axes[1].plot(xs, g, marker="o", linewidth=1.6, label=mode)
    axes[0].set_title("Mean target quality by score decile")
    axes[1].set_title("Good-label rate by score decile")
    for ax in axes:
        ax.set_xlabel("score decile, low to high")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("target quality")
    axes[1].set_ylabel("good rate")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_scores(
    name: str,
    scores: dict[str, np.ndarray],
    binary: np.ndarray,
    quality: np.ndarray,
    reason: np.ndarray,
    out_dir: Path,
    n_bins: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"n": int(len(quality)), "modes": {}}
    plot_payload: dict[str, list[dict[str, Any]]] = {}
    for mode, score in scores.items():
        rows = decile_bins(score, binary, quality, reason, n_bins)
        stats = monotonic_stats(rows)
        stats.update(
            {
                "quality_pearson": finite_corr(score, quality),
                "quality_spearman": spearman_simple(score, quality),
                "binary_auc": safe_auc(binary, score),
                "score_min": float(np.min(score)),
                "score_max": float(np.max(score)),
                "score_std": float(np.std(score)),
            }
        )
        result["modes"][mode] = {"summary": stats, "bins": rows}
        write_bins_csv(rows, out_dir / f"{name}_{mode}_bins.csv")
        plot_payload[mode] = rows
    plot_bins(plot_payload, name, out_dir / f"{name}_calibration.png")
    return result


@torch.no_grad()
def insertion_predictions(args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(args.insertion_features, allow_pickle=True)
    runtime = InsertionRiskScorerRuntime(str(args.insertion_ckpt), device=args.device)
    marker = torch.from_numpy(data["marker"].astype(np.float32)).to(runtime.device)
    action = torch.from_numpy(data["action"].astype(np.float32)).to(runtime.device)
    outputs: list[dict[str, np.ndarray]] = []
    for start in range(0, len(marker), args.batch_size):
        out = runtime.forward(marker[start : start + args.batch_size], joint_action_seq=action[start : start + args.batch_size])
        outputs.append({k: v.detach().cpu().numpy() for k, v in out.items() if k in {"quality_score", "p_good", "risk_prob", "energy_score", "energy_clipped"}})
    quality_score = np.concatenate([o["quality_score"] for o in outputs])
    p_good = np.concatenate([o["p_good"] for o in outputs])
    risk_prob = np.concatenate([o["risk_prob"] for o in outputs])
    energy_score = np.concatenate([o["energy_score"] for o in outputs])
    energy_clipped = np.concatenate([o["energy_clipped"] for o in outputs])
    profile = 0.5 * np.log(np.clip(quality_score, 1e-8, 1.0) / np.clip(1.0 - quality_score, 1e-8, 1.0)) + 0.1 * np.log(np.clip(p_good, 1e-8, 1.0) / np.clip(1.0 - p_good, 1e-8, 1.0))
    return {
        "scores": {
            "quality": quality_score,
            "p_good": p_good,
            "neg_risk": -risk_prob,
            "energy": energy_score,
            "energy_clipped": energy_clipped,
            "profile": np.tanh(profile / 4.0) * 4.0,
        },
        "binary": data["binary"].astype(np.int64),
        "quality": data["quality"].astype(np.float32),
        "reason": data["reason"].astype(np.int64),
    }


@torch.no_grad()
def ptg_predictions(args: argparse.Namespace) -> dict[str, Any]:
    data = np.load(args.ptg_features, allow_pickle=True)
    ckpt = torch.load(args.ptg_ckpt, map_location="cpu", weights_only=False)
    feature_dim = int(ckpt["feature_dim"])
    model = PTGProxyScorerV2(feature_dim, hidden=int(ckpt.get("hidden", 192)), dropout=0.0).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    x = data["X"].astype(np.float32)
    task_id = data["task_id"].astype(np.int64)
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    x = (x - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-8)
    scores: dict[str, list[np.ndarray]] = {k: [] for k in ["quality", "p_good", "reason_good", "energy", "energy_clipped", "profile"]}
    for start in range(0, len(x), args.batch_size):
        xb = torch.from_numpy(x[start : start + args.batch_size]).float().to(args.device)
        tb = torch.from_numpy(task_id[start : start + args.batch_size]).long().to(args.device)
        out = model(xb, tb)
        p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(out["reason_logits"], dim=-1)
        quality = torch.sigmoid(out["quality"])
        good_margin = out["binary_logits"][:, 1] - out["binary_logits"][:, 0]
        bad_reason_logits = torch.stack([out["reason_logits"][:, 0], torch.logsumexp(out["reason_logits"][:, 2:], dim=-1)], dim=-1)
        reason_margin = out["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits, dim=-1)
        energy = out["quality"] + 0.25 * good_margin + 0.25 * reason_margin
        profile = 0.75 * out["quality"] + 0.10 * good_margin
        scores["quality"].append(quality.cpu().numpy())
        scores["p_good"].append(p_good.cpu().numpy())
        scores["reason_good"].append(reason_prob[:, 1].cpu().numpy())
        scores["energy"].append(energy.cpu().numpy())
        scores["energy_clipped"].append((torch.tanh(energy / 4.0) * 4.0).cpu().numpy())
        scores["profile"].append((torch.tanh(profile / 4.0) * 4.0).cpu().numpy())
    return {
        "scores": {k: np.concatenate(v) for k, v in scores.items()},
        "binary": data["binary"].astype(np.int64),
        "quality": data["quality"].astype(np.float32),
        "reason": data["reason"].astype(np.int64),
        "task": data["task"].astype(str),
    }


def subset_payload(payload: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        "scores": {k: v[mask] for k, v in payload["scores"].items()},
        "binary": payload["binary"][mask],
        "quality": payload["quality"][mask],
        "reason": payload["reason"][mask],
    }


def best_mode(section: dict[str, Any]) -> str:
    def key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float, float, float]:
        s = item[1]["summary"]
        auc = -1.0 if s["binary_auc"] is None else float(s["binary_auc"])
        return (
            float(s["quality_positive_step_rate"] or 0.0),
            float(s["quality_spearman"]),
            float(s["top_bottom_quality_gap"] or 0.0),
            float(s["good_rate_positive_step_rate"] or 0.0),
            auc,
        )

    return max(section["modes"].items(), key=key)[0]


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Formal TacQuality Score Calibration",
        "",
        "This audit uses the formal `TFAC_V5.tac_quality_energy` runtime/checkpoint interfaces.",
        "It checks score ordering for classifier/score guidance; it is not a policy rollout metric.",
        "",
        "## Recommended Modes",
        "",
    ]
    for task, mode in result["recommendation"].items():
        if task != "interpretation":
            lines.append(f"- {task}: `{mode}`")
    lines.extend(["", result["recommendation"]["interpretation"], "", "## Metrics", ""])
    for section_name in result["section_order"]:
        lines.extend([f"### {section_name}", "", "| mode | AUC | Spearman(q) | top-bottom q gap | q monotonic | good-rate gap |", "|---|---:|---:|---:|---:|---:|"])
        for mode, payload in result[section_name]["modes"].items():
            s = payload["summary"]
            lines.append(
                f"| {mode} | {fmt(s['binary_auc'])} | {fmt(s['quality_spearman'])} | "
                f"{fmt(s['top_bottom_quality_gap'])} | {fmt(s['quality_positive_step_rate'])} | "
                f"{fmt(s['top_bottom_good_rate_gap'])} |"
            )
        lines.append("")
    lines.extend(["## Evidence Boundaries", "", "- This uses saved labels/features/checkpoints, not new robot rollouts.", "- Board labels are weak labels from force/contact smoothness; real deployment must still be judged by server-side force traces.", "- The score selected for deployment should match `score_mode=profile` in the rollout config.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    insertion = insertion_predictions(args)
    ptg = ptg_predictions(args)
    board_mask = ptg["task"] == "board"
    insertion_mask = ptg["task"] == "insertion"
    result: dict[str, Any] = {
        "purpose": "Formal TacQuality scorer score calibration for DP guidance.",
        "protocol": {
            "n_bins": args.n_bins,
            "note": "Uses saved feature caches and final checkpoints; no retraining.",
        },
        "inputs": {
            "insertion_features": str(args.insertion_features),
            "insertion_ckpt": str(args.insertion_ckpt),
            "ptg_features": str(args.ptg_features),
            "ptg_ckpt": str(args.ptg_ckpt),
        },
        "section_order": [
            "insertion_risk_scorer",
            "ptg_proxy_v2_board",
            "ptg_proxy_v2_insertion_subset",
            "ptg_proxy_v2_mixed",
        ],
    }
    result["insertion_risk_scorer"] = summarize_scores("insertion_risk_scorer", insertion["scores"], insertion["binary"], insertion["quality"], insertion["reason"], out_dir, args.n_bins)
    result["ptg_proxy_v2_board"] = summarize_scores("ptg_proxy_v2_board", out_dir=out_dir, n_bins=args.n_bins, **subset_payload(ptg, board_mask))
    result["ptg_proxy_v2_insertion_subset"] = summarize_scores("ptg_proxy_v2_insertion_subset", out_dir=out_dir, n_bins=args.n_bins, **subset_payload(ptg, insertion_mask))
    result["ptg_proxy_v2_mixed"] = summarize_scores("ptg_proxy_v2_mixed", ptg["scores"], ptg["binary"], ptg["quality"], ptg["reason"], out_dir, args.n_bins)
    result["recommendation"] = {
        "insertion": best_mode(result["insertion_risk_scorer"]),
        "board": best_mode(result["ptg_proxy_v2_board"]),
        "mixed": best_mode(result["ptg_proxy_v2_mixed"]),
        "interpretation": "Selection prioritizes monotonic quality deciles, quality rank correlation, and top-bottom quality gap before AUC.",
    }
    json_path = out_dir / "formal_tac_quality_score_calibration.json"
    md_path = out_dir / "formal_tac_quality_score_calibration.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"recommendation": result["recommendation"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--insertion_features", type=Path, default=DEFAULT_INSERTION_FEATURES)
    parser.add_argument("--insertion_ckpt", type=Path, default=DEFAULT_INSERTION_CKPT)
    parser.add_argument("--ptg_features", type=Path, default=DEFAULT_PTG_FEATURES)
    parser.add_argument("--ptg_ckpt", type=Path, default=DEFAULT_PTG_CKPT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
