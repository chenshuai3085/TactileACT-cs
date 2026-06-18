#!/usr/bin/env python3
"""Sweep simple ensembles of two board TacQuality scorer alignment outputs.

This is an offline ablation only.  It uses saved Foresight-alignment CSV files
from two already-trained board scorers and asks whether a weighted score keeps
the old scorer's pred/GT consistency while improving force-quality correlation.
It does not retrain a model and does not prove real robot improvement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score


DEFAULT_OLD_CSV = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/"
    "foresight_alignment_quality_include260617_sameset/foresight_score_alignment_samples.csv"
)
DEFAULT_S12_CSV = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "foresight_alignment_quality/foresight_score_alignment_samples.csv"
)
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/board_scorer_ensemble_sweep_20260619")


def finite_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(x), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(len(order), dtype=np.float64)
    return ranks


def spearman_simple(x: np.ndarray, y: np.ndarray) -> float | None:
    return finite_corr(rankdata_simple(np.asarray(x)), rankdata_simple(np.asarray(y)))


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, score))


def summarize(values: Iterable[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def read_rows(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["episode"], row["start"], row["label"])
            parsed: dict[str, Any] = dict(row)
            for k in [
                "good",
                "pred_score",
                "gt_score",
                "marker_mae",
                "future_force_abs_mean",
                "future_force_mag_mean",
                "future_force_delta_abs_mean",
                "future_force_band_quality",
            ]:
                parsed[k] = float(row[k])
            parsed["good"] = int(parsed["good"])
            out[key] = parsed
    return out


def normalize_score(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if mode == "raw":
        return x
    if mode == "zscore":
        std = float(np.std(x))
        return (x - float(np.mean(x))) / max(std, 1e-12)
    if mode == "rank":
        ranks = rankdata_simple(x)
        return ranks / max(len(ranks) - 1, 1)
    raise ValueError(f"unknown normalization mode: {mode}")


def metrics(
    *,
    pred: np.ndarray,
    gt: np.ndarray,
    good: np.ndarray,
    marker_mae: np.ndarray,
    force_abs: np.ndarray,
    force_mag: np.ndarray,
    force_delta: np.ndarray,
    force_quality: np.ndarray,
    label: np.ndarray,
    force_center: float,
) -> dict[str, Any]:
    pred_thr = float(np.median(pred))
    gt_thr = float(np.median(gt))
    return {
        "n": int(len(pred)),
        "pred_auc_good": safe_auc(good, pred),
        "pred_ap_good": safe_ap(good, pred),
        "pred_balanced_accuracy_at_median": float(balanced_accuracy_score(good, pred >= pred_thr)),
        "gt_auc_good": safe_auc(good, gt),
        "gt_ap_good": safe_ap(good, gt),
        "gt_balanced_accuracy_at_median": float(balanced_accuracy_score(good, gt >= gt_thr)),
        "pred_gt_pearson": finite_corr(pred, gt),
        "pred_gt_spearman": spearman_simple(pred, gt),
        "pred_score_vs_marker_mae_spearman": spearman_simple(pred, -marker_mae),
        "pred_score_vs_force_abs_spearman": spearman_simple(pred, -force_abs),
        "pred_score_vs_force_mag_spearman": spearman_simple(pred, -np.abs(force_mag - force_center)),
        "pred_score_vs_force_delta_spearman": spearman_simple(pred, -force_delta),
        "pred_score_vs_force_band_quality_spearman": spearman_simple(pred, force_quality),
        "force_band_quality_auc_good": safe_auc(good, force_quality),
        "pred_score": summarize(pred),
        "gt_score": summarize(gt),
        "by_label": {
            str(name): {
                "n": int((label == name).sum()),
                "pred_score": summarize(pred[label == name]),
                "gt_score": summarize(gt[label == name]),
                "force_band_quality": summarize(force_quality[label == name]),
            }
            for name in sorted(set(label.tolist()))
        },
    }


def scalar(value: Any, default: float = -1.0) -> float:
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if np.isfinite(value) else default


def selection_score(row: dict[str, Any]) -> float:
    """Conservative scalar for ranking ablations.

    Primary goal is credible predicted-domain guidance, so pred/GT consistency
    is weighted highest.  Force-band correlation matters for board wiping, but
    weak force labels alone should not override the tactile consequence match.
    """

    pred_gt = max(scalar(row.get("pred_gt_spearman")), 0.0)
    force = max(scalar(row.get("pred_score_vs_force_band_quality_spearman")), 0.0)
    auc = max(scalar(row.get("pred_auc_good")), 0.5)
    marker = max(scalar(row.get("pred_score_vs_marker_mae_spearman")), 0.0)
    return float(0.45 * pred_gt + 0.25 * force + 0.20 * auc + 0.10 * marker)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = [
        "mode",
        "old_weight",
        "selection_score",
        "pred_auc_good",
        "pred_balanced_accuracy_at_median",
        "gt_auc_good",
        "pred_gt_spearman",
        "pred_gt_pearson",
        "pred_score_vs_force_band_quality_spearman",
        "pred_score_vs_force_delta_spearman",
        "pred_score_vs_marker_mae_spearman",
        "pred_score_vs_force_abs_spearman",
        "pred_score_vs_force_mag_spearman",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def plot_sweep(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150, sharex=True)
    for mode in ["raw", "zscore", "rank"]:
        sub = [r for r in rows if r["mode"] == mode]
        xs = np.asarray([r["old_weight"] for r in sub], dtype=np.float64)
        axes[0].plot(xs, [r["pred_gt_spearman"] for r in sub], marker="o", label=mode)
        axes[1].plot(xs, [r["pred_score_vs_force_band_quality_spearman"] for r in sub], marker="o", label=mode)
        axes[2].plot(xs, [r["selection_score"] for r in sub], marker="o", label=mode)
    axes[0].set_ylabel("pred/GT Spearman")
    axes[1].set_ylabel("pred vs force-quality Spearman")
    axes[2].set_ylabel("selection score")
    for ax in axes:
        ax.set_xlabel("old scorer weight")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Board scorer ensemble sweep")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_md(result: dict[str, Any], path: Path) -> None:
    rows = result["rows"]
    best = result["best_by_selection"]
    old = next(r for r in rows if r["mode"] == "raw" and abs(r["old_weight"] - 1.0) < 1e-9)
    s12 = next(r for r in rows if r["mode"] == "raw" and abs(r["old_weight"]) < 1e-9)
    lines = [
        "# Board Scorer Ensemble Sweep",
        "",
        "Purpose: compare the current board default scorer, the s12 scorer, and simple weighted ensembles on the same Foresight-alignment samples.",
        "",
        "This is an offline ablation only. It does not replace the deployment default and does not prove real robot improvement.",
        "",
        "## Inputs",
        "",
        f"- old CSV: `{result['inputs']['old_csv']}`",
        f"- s12 CSV: `{result['inputs']['s12_csv']}`",
        f"- matched samples: `{result['n_matched']}`",
        "",
        "## Main Result",
        "",
        "| candidate | mode | old weight | pred AUC | pred/GT Spearman | pred vs force-quality Spearman | selection score |",
        "|---|---|---:|---:|---:|---:|---:|",
        (
            f"| current default old | `{old['mode']}` | {old['old_weight']:.2f} | "
            f"{old['pred_auc_good']:.4f} | {old['pred_gt_spearman']:.4f} | "
            f"{old['pred_score_vs_force_band_quality_spearman']:.4f} | {old['selection_score']:.4f} |"
        ),
        (
            f"| s12 | `{s12['mode']}` | {s12['old_weight']:.2f} | "
            f"{s12['pred_auc_good']:.4f} | {s12['pred_gt_spearman']:.4f} | "
            f"{s12['pred_score_vs_force_band_quality_spearman']:.4f} | {s12['selection_score']:.4f} |"
        ),
        (
            f"| best ensemble | `{best['mode']}` | {best['old_weight']:.2f} | "
            f"{best['pred_auc_good']:.4f} | {best['pred_gt_spearman']:.4f} | "
            f"{best['pred_score_vs_force_band_quality_spearman']:.4f} | {best['selection_score']:.4f} |"
        ),
        "",
        "## Top Candidates",
        "",
        "| rank | mode | old weight | pred AUC | pred/GT Spearman | force-quality Spearman | selection score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(result["top_by_selection"][:10], start=1):
        lines.append(
            f"| {i} | `{row['mode']}` | {row['old_weight']:.2f} | "
            f"{row['pred_auc_good']:.4f} | {row['pred_gt_spearman']:.4f} | "
            f"{row['pred_score_vs_force_band_quality_spearman']:.4f} | {row['selection_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the best ensemble is essentially old-weight 1.0, the current default should remain unchanged.",
            "- If a mixed weight wins only by a tiny margin, keep it as an ablation candidate until real rollout force traces exist.",
            "- The selected scalar favors pred/GT consistency first, then board force-band correlation, then good-label AUC.",
            "",
            "## Artifacts",
            "",
            f"- JSON: `{result['artifacts']['json']}`",
            f"- CSV: `{result['artifacts']['csv']}`",
            f"- plot: `{result['artifacts']['plot']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    old_rows = read_rows(args.old_csv)
    s12_rows = read_rows(args.s12_csv)
    keys = sorted(set(old_rows) & set(s12_rows))
    if not keys:
        raise RuntimeError("No matched samples between old and s12 CSV files.")
    if len(keys) != len(old_rows) or len(keys) != len(s12_rows):
        print(f"[warn] partial match: old={len(old_rows)}, s12={len(s12_rows)}, matched={len(keys)}")

    old_pred = np.asarray([old_rows[k]["pred_score"] for k in keys], dtype=np.float64)
    old_gt = np.asarray([old_rows[k]["gt_score"] for k in keys], dtype=np.float64)
    s12_pred = np.asarray([s12_rows[k]["pred_score"] for k in keys], dtype=np.float64)
    s12_gt = np.asarray([s12_rows[k]["gt_score"] for k in keys], dtype=np.float64)
    good = np.asarray([old_rows[k]["good"] for k in keys], dtype=np.int64)
    label = np.asarray([old_rows[k]["label"] for k in keys], dtype=object)
    marker_mae = np.asarray([old_rows[k]["marker_mae"] for k in keys], dtype=np.float64)
    force_abs = np.asarray([old_rows[k]["future_force_abs_mean"] for k in keys], dtype=np.float64)
    force_mag = np.asarray([old_rows[k]["future_force_mag_mean"] for k in keys], dtype=np.float64)
    force_delta = np.asarray([old_rows[k]["future_force_delta_abs_mean"] for k in keys], dtype=np.float64)
    force_quality = np.asarray([old_rows[k]["future_force_band_quality"] for k in keys], dtype=np.float64)

    rows: list[dict[str, Any]] = []
    weights = [round(float(w), 4) for w in np.linspace(0.0, 1.0, int(args.num_weights))]
    for mode in args.modes:
        op = normalize_score(old_pred, mode)
        og = normalize_score(old_gt, mode)
        sp = normalize_score(s12_pred, mode)
        sg = normalize_score(s12_gt, mode)
        for old_weight in weights:
            pred = old_weight * op + (1.0 - old_weight) * sp
            gt = old_weight * og + (1.0 - old_weight) * sg
            row = {
                "mode": mode,
                "old_weight": float(old_weight),
                **metrics(
                    pred=pred,
                    gt=gt,
                    good=good,
                    marker_mae=marker_mae,
                    force_abs=force_abs,
                    force_mag=force_mag,
                    force_delta=force_delta,
                    force_quality=force_quality,
                    label=label,
                    force_center=float(args.force_center),
                ),
            }
            row["selection_score"] = selection_score(row)
            rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (-float(r["selection_score"]), -float(r["pred_gt_spearman"] or -1)))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "board_scorer_ensemble_sweep.csv"
    json_path = out_dir / "board_scorer_ensemble_sweep.json"
    md_path = out_dir / "board_scorer_ensemble_sweep.md"
    plot_path = out_dir / "board_scorer_ensemble_sweep.png"
    result = {
        "purpose": "Offline sweep of old/s12 board TacQuality scorer weighted ensembles.",
        "inputs": {
            "old_csv": str(args.old_csv),
            "s12_csv": str(args.s12_csv),
        },
        "n_matched": int(len(keys)),
        "modes": list(args.modes),
        "weights": weights,
        "selection_score_formula": "0.45*pred_gt_spearman + 0.25*force_quality_spearman + 0.20*pred_auc_good + 0.10*marker_mae_spearman, clipped at zero for correlations and 0.5 for AUC.",
        "best_by_selection": rows_sorted[0],
        "top_by_selection": rows_sorted[:10],
        "rows": rows,
        "artifacts": {
            "json": str(json_path),
            "csv": str(csv_path),
            "markdown": str(md_path),
            "plot": str(plot_path),
        },
        "evidence_boundary": [
            "Uses saved offline Foresight-alignment samples only.",
            "Does not retrain any scorer.",
            "Does not execute robot rollouts.",
            "A mixed ensemble should remain ablation-only unless real rollout force traces improve.",
        ],
    }
    write_csv(rows, csv_path)
    plot_sweep(rows, plot_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(result, md_path)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "plot": str(plot_path),
        "best": {
            "mode": result["best_by_selection"]["mode"],
            "old_weight": result["best_by_selection"]["old_weight"],
            "selection_score": result["best_by_selection"]["selection_score"],
            "pred_gt_spearman": result["best_by_selection"]["pred_gt_spearman"],
            "force_quality_spearman": result["best_by_selection"]["pred_score_vs_force_band_quality_spearman"],
            "pred_auc_good": result["best_by_selection"]["pred_auc_good"],
        },
    }, ensure_ascii=False, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old_csv", type=Path, default=DEFAULT_OLD_CSV)
    parser.add_argument("--s12_csv", type=Path, default=DEFAULT_S12_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_weights", type=int, default=21)
    parser.add_argument("--modes", nargs="+", default=["raw", "zscore", "rank"], choices=["raw", "zscore", "rank"])
    parser.add_argument("--force_center", type=float, default=10.456689834594727)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
