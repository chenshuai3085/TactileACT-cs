"""Audit TacQuality scorer calibration and monotonicity.

Classifier guidance needs more than high classification accuracy.  The score
used as a guidance potential should be ordered: higher score should correspond
to higher good-label rate and higher tactile quality on held-out data.  This
script evaluates that property on the saved feature caches/checkpoints without
retraining the scorers.

Outputs:
  - JSON summary with monotonicity, top-vs-bottom gaps, AUC/correlation.
  - CSV bin tables for insertion, board, and mixed subsets.
  - PNG calibration/monotonicity plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime  # noqa: E402
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_score_calibration")

INSERTION_FEATURES = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_features.npz")
INSERTION_CKPT = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt")
PTG_FEATURES = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_features.npz")
PTG_CKPT = Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt")


def finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
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
    binary = np.asarray(binary)
    mask = binary >= 0
    if mask.sum() < 3 or len(np.unique(binary[mask])) < 2:
        return None
    return float(roc_auc_score(binary[mask], score[mask]))


def decile_bins(score: np.ndarray, binary: np.ndarray, quality: np.ndarray, reason: np.ndarray, n_bins: int) -> List[Dict[str, Any]]:
    score = np.asarray(score, dtype=np.float64)
    binary = np.asarray(binary)
    quality = np.asarray(quality, dtype=np.float64)
    reason = np.asarray(reason)
    order = np.argsort(score, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows: List[Dict[str, Any]] = []
    for i, idx in enumerate(chunks):
        if len(idx) == 0:
            continue
        valid = binary[idx] >= 0
        good_rate = float(np.mean(binary[idx][valid] == 1)) if np.any(valid) else None
        reason_counts = {str(k): int(v) for k, v in zip(*np.unique(reason[idx], return_counts=True))}
        rows.append(
            {
                "bin": int(i),
                "n": int(len(idx)),
                "score_min": float(np.min(score[idx])),
                "score_max": float(np.max(score[idx])),
                "score_mean": float(np.mean(score[idx])),
                "good_rate": good_rate,
                "quality_mean": float(np.mean(quality[idx])),
                "reason_counts": reason_counts,
            }
        )
    return rows


def monotonic_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    q = np.array([r["quality_mean"] for r in rows], dtype=np.float64)
    good = np.array([np.nan if r["good_rate"] is None else r["good_rate"] for r in rows], dtype=np.float64)
    quality_steps = np.diff(q)
    valid_good = np.isfinite(good)
    good_steps = np.diff(good[valid_good]) if valid_good.sum() >= 2 else np.array([], dtype=np.float64)
    return {
        "quality_monotonic_non_decreasing": bool(np.all(quality_steps >= -1e-8)),
        "quality_positive_step_rate": float(np.mean(quality_steps >= -1e-8)) if len(quality_steps) else None,
        "good_rate_monotonic_non_decreasing": bool(np.all(good_steps >= -1e-8)) if len(good_steps) else None,
        "good_rate_positive_step_rate": float(np.mean(good_steps >= -1e-8)) if len(good_steps) else None,
        "bottom_quality_mean": float(q[0]) if len(q) else None,
        "top_quality_mean": float(q[-1]) if len(q) else None,
        "top_bottom_quality_gap": float(q[-1] - q[0]) if len(q) else None,
        "bottom_good_rate": None if len(good) == 0 or not np.isfinite(good[0]) else float(good[0]),
        "top_good_rate": None if len(good) == 0 or not np.isfinite(good[-1]) else float(good[-1]),
        "top_bottom_good_rate_gap": (
            None
            if len(good) == 0 or not np.isfinite(good[0]) or not np.isfinite(good[-1])
            else float(good[-1] - good[0])
        ),
    }


def write_bins_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin",
                "n",
                "score_min",
                "score_max",
                "score_mean",
                "good_rate",
                "quality_mean",
                "reason_counts",
            ],
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["reason_counts"] = json.dumps(out["reason_counts"], ensure_ascii=False)
            writer.writerow(out)


def plot_bins(rows_by_mode: Dict[str, List[Dict[str, Any]]], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for mode, rows in rows_by_mode.items():
        xs = np.array([r["bin"] for r in rows])
        q = np.array([r["quality_mean"] for r in rows])
        g = np.array([np.nan if r["good_rate"] is None else r["good_rate"] for r in rows])
        axes[0].plot(xs, q, marker="o", linewidth=1.8, label=mode)
        if np.isfinite(g).any():
            axes[1].plot(xs, g, marker="o", linewidth=1.8, label=mode)
    axes[0].set_title("Mean target quality by score decile")
    axes[0].set_xlabel("score decile, low to high")
    axes[0].set_ylabel("target quality")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Good-label rate by score decile")
    axes[1].set_xlabel("score decile, low to high")
    axes[1].set_ylabel("good rate")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize_scores(
    *,
    name: str,
    scores: Dict[str, np.ndarray],
    binary: np.ndarray,
    quality: np.ndarray,
    reason: np.ndarray,
    out_dir: Path,
    n_bins: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"n": int(len(quality)), "modes": {}}
    plot_payload = {}
    for mode, score in scores.items():
        rows = decile_bins(score, binary, quality, reason, n_bins=n_bins)
        stats = monotonic_stats(rows)
        stats.update(
            {
                "quality_pearson": finite_corr(score, quality),
                "quality_spearman": spearman_simple(score, quality),
                "binary_auc": safe_auc(binary, score),
                "score_min": float(np.min(score)),
                "score_max": float(np.max(score)),
                "score_std": float(np.std(score)),
                "near_min_rate": float(np.mean(score <= np.quantile(score, 0.01))),
                "near_max_rate": float(np.mean(score >= np.quantile(score, 0.99))),
            }
        )
        result["modes"][mode] = {"summary": stats, "bins": rows}
        write_bins_csv(rows, out_dir / f"{name}_{mode}_bins.csv")
        plot_payload[mode] = rows
    plot_bins(plot_payload, name, out_dir / f"{name}_calibration.png")
    return result


@torch.no_grad()
def insertion_predictions(args) -> Dict[str, Any]:
    data = np.load(args.insertion_features, allow_pickle=True)
    runtime = InsertionRiskScorerRuntime(args.insertion_ckpt, device=args.device)
    marker = torch.from_numpy(data["marker"].astype(np.float32)).to(runtime.device)
    action = torch.from_numpy(data["action"].astype(np.float32)).to(runtime.device)
    outputs: List[Dict[str, np.ndarray]] = []
    for start in range(0, len(marker), args.batch_size):
        out = runtime.forward(marker[start : start + args.batch_size], action[start : start + args.batch_size])
        outputs.append({k: v.detach().cpu().numpy() for k, v in out.items() if k in {
            "quality_score",
            "p_good",
            "risk_prob",
            "energy_score",
        }})
    quality_score = np.concatenate([o["quality_score"] for o in outputs])
    p_good = np.concatenate([o["p_good"] for o in outputs])
    risk_prob = np.concatenate([o["risk_prob"] for o in outputs])
    energy_score = np.concatenate([o["energy_score"] for o in outputs])
    risk_guidance = quality_score + 0.35 * np.log(np.clip(p_good, 1e-8, 1.0)) - 0.5 * risk_prob
    return {
        "scores": {
            "quality": quality_score,
            "p_good": p_good,
            "neg_risk": -risk_prob,
            "risk_guidance": risk_guidance,
            "energy": energy_score,
        },
        "binary": data["binary"].astype(np.int64),
        "quality": data["quality"].astype(np.float32),
        "reason": data["reason"].astype(np.int64),
        "groups": data["groups"].astype(str),
    }


@torch.no_grad()
def ptg_predictions(args) -> Dict[str, Any]:
    data = np.load(args.ptg_features, allow_pickle=True)
    ckpt = torch.load(args.ptg_ckpt, map_location="cpu", weights_only=False)
    runtime = PTGProxyScorerV2Runtime(args.ptg_ckpt, device=args.device)
    X = data["X"].astype(np.float32)
    task_id = data["task_id"].astype(np.int64)
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)
    Xs = (X - mean.reshape(1, -1)) / (scale.reshape(1, -1) + 1e-8)
    scores: Dict[str, List[np.ndarray]] = {
        "quality": [],
        "p_good": [],
        "reason_good": [],
        "energy": [],
        "weighted_energy": [],
    }
    board_profile = get_guidance_profile("board")
    for start in range(0, len(Xs), args.batch_size):
        xb = torch.from_numpy(Xs[start : start + args.batch_size]).float().to(runtime.device)
        tb = torch.from_numpy(task_id[start : start + args.batch_size]).long().to(runtime.device)
        out = runtime.model(xb, tb)
        p_good = torch.softmax(out["binary_logits"], dim=-1)[:, 1]
        reason_prob = torch.softmax(out["reason_logits"], dim=-1)
        quality = torch.sigmoid(out["quality"])
        good_margin = out["binary_logits"][:, 1] - out["binary_logits"][:, 0]
        bad_reason_logits = torch.stack(
            [
                out["reason_logits"][:, 0],
                torch.logsumexp(out["reason_logits"][:, 2:], dim=-1),
            ],
            dim=-1,
        )
        reason_margin = out["reason_logits"][:, 1] - torch.logsumexp(bad_reason_logits, dim=-1)
        energy = out["quality"] + 0.25 * good_margin + 0.25 * reason_margin
        weighted = (
            board_profile.energy.quality * out["quality"]
            + board_profile.energy.binary_margin * good_margin
            + board_profile.energy.reason_margin * reason_margin
        )
        weighted = torch.tanh(weighted / board_profile.energy.clip_scale) * board_profile.energy.clip_scale
        scores["quality"].append(quality.cpu().numpy())
        scores["p_good"].append(p_good.cpu().numpy())
        scores["reason_good"].append(reason_prob[:, 1].cpu().numpy())
        scores["energy"].append(energy.cpu().numpy())
        scores["weighted_energy"].append(weighted.cpu().numpy())
    return {
        "scores": {k: np.concatenate(v) for k, v in scores.items()},
        "binary": data["binary"].astype(np.int64),
        "quality": data["quality"].astype(np.float32),
        "reason": data["reason"].astype(np.int64),
        "task": data["task"].astype(str),
        "groups": data["groups"].astype(str),
    }


def subset_payload(payload: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    return {
        "scores": {k: v[mask] for k, v in payload["scores"].items()},
        "binary": payload["binary"][mask],
        "quality": payload["quality"][mask],
        "reason": payload["reason"][mask],
    }


def best_mode(summary: Dict[str, Any]) -> str:
    """Choose a guidance potential, not just the best classifier.

    A probability head can have excellent AUC while being less useful for
    gradient guidance because it may saturate near 0/1 and be less aligned with
    the continuous quality target.  The guidance choice therefore prioritizes
    monotonic quality bins, quality rank correlation, and top-vs-bottom quality
    gap.  AUC is included as a safety constraint.
    """

    def key(item: Tuple[str, Dict[str, Any]]):
        s = item[1]["summary"]
        auc = -math.inf if s["binary_auc"] is None else s["binary_auc"]
        return (
            s["quality_positive_step_rate"] or 0.0,
            s["quality_spearman"],
            s["top_bottom_quality_gap"],
            s["good_rate_positive_step_rate"] or 0.0,
            auc,
        )

    return max(summary["modes"].items(), key=key)[0]


def run(args) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    insertion = insertion_predictions(args)
    ptg = ptg_predictions(args)
    board_mask = ptg["task"] == "board"
    insertion_mask = ptg["task"] == "insertion"

    result = {
        "purpose": "Score calibration / monotonicity audit for DP classifier guidance potentials.",
        "protocol": {
            "note": (
                "Uses saved final checkpoints and saved feature caches. This is not a retraining metric; "
                "it checks whether candidate guidance scores are ordered with target quality/good labels."
            ),
            "n_bins": args.n_bins,
        },
        "inputs": {
            "insertion_features": str(args.insertion_features),
            "insertion_ckpt": str(args.insertion_ckpt),
            "ptg_features": str(args.ptg_features),
            "ptg_ckpt": str(args.ptg_ckpt),
        },
        "insertion_risk_scorer": summarize_scores(
            name="insertion_risk_scorer",
            scores=insertion["scores"],
            binary=insertion["binary"],
            quality=insertion["quality"],
            reason=insertion["reason"],
            out_dir=out_dir,
            n_bins=args.n_bins,
        ),
        "ptg_proxy_v2_board": summarize_scores(
            name="ptg_proxy_v2_board",
            out_dir=out_dir,
            n_bins=args.n_bins,
            **subset_payload(ptg, board_mask),
        ),
        "ptg_proxy_v2_insertion_subset": summarize_scores(
            name="ptg_proxy_v2_insertion_subset",
            out_dir=out_dir,
            n_bins=args.n_bins,
            **subset_payload(ptg, insertion_mask),
        ),
        "ptg_proxy_v2_mixed": summarize_scores(
            name="ptg_proxy_v2_mixed",
            scores=ptg["scores"],
            binary=ptg["binary"],
            quality=ptg["quality"],
            reason=ptg["reason"],
            out_dir=out_dir,
            n_bins=args.n_bins,
        ),
    }
    result["recommendation"] = {
        "insertion": best_mode(result["insertion_risk_scorer"]),
        "board": best_mode(result["ptg_proxy_v2_board"]),
        "mixed": best_mode(result["ptg_proxy_v2_mixed"]),
        "interpretation": (
            "Guidance-mode selection prioritizes monotonic quality deciles, quality rank correlation, "
            "and top-bottom quality gap. A mode with the highest AUC can still be weaker as a "
            "gradient potential if it is probability-saturated or less aligned with continuous quality."
        ),
    }

    json_path = out_dir / "tac_quality_score_calibration.json"
    md_path = out_dir / "tac_quality_score_calibration.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({"recommendation": result["recommendation"], "saved": str(json_path)}, ensure_ascii=False, indent=2))
    return result


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Score Calibration Audit",
        "",
        "This audit checks whether scorer values are ordered enough to be useful as DP classifier-guidance energy.",
        "",
        "## Recommended Modes",
        "",
    ]
    for task, mode in result["recommendation"].items():
        if task == "interpretation":
            continue
        lines.append(f"- {task}: `{mode}`")
    lines.extend(["", result["recommendation"]["interpretation"], "", "## Key Metrics", ""])
    for section in [
        "insertion_risk_scorer",
        "ptg_proxy_v2_board",
        "ptg_proxy_v2_insertion_subset",
        "ptg_proxy_v2_mixed",
    ]:
        lines.append(f"### {section}")
        lines.append("")
        lines.append("| mode | AUC | Spearman(q) | top-bottom q gap | q monotonic | good-rate gap |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for mode, payload in result[section]["modes"].items():
            s = payload["summary"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        mode,
                        fmt(s["binary_auc"]),
                        fmt(s["quality_spearman"]),
                        fmt(s["top_bottom_quality_gap"]),
                        fmt(s["quality_positive_step_rate"]),
                        fmt(s["top_bottom_good_rate_gap"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "- JSON: `tac_quality_score_calibration.json`",
            "- CSV bins: `*_bins.csv`",
            "- Figures: `*_calibration.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--insertion_features", type=Path, default=INSERTION_FEATURES)
    parser.add_argument("--insertion_ckpt", type=Path, default=INSERTION_CKPT)
    parser.add_argument("--ptg_features", type=Path, default=PTG_FEATURES)
    parser.add_argument("--ptg_ckpt", type=Path, default=PTG_CKPT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
