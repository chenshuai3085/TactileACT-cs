#!/usr/bin/env python3
"""Audit board-wiping contact windows for scorer/DP sampling.

The current board TacQuality scripts approximate the wiping phase with a fixed
time fraction, usually 0.25 to 0.85 of each episode.  This script estimates
contact directly from tactile marker and force signals, then reports how well
that fixed fraction covers true contact windows.

It is an offline audit only; it does not train a model and does not modify the
active DP training run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASETS = {
    "positive_old": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609",
    "positive_260617": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617",
    "too_small": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high",
    "too_large": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low",
    "oscillate": "/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate",
}
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/board_contact_window_audit_20260619")


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def robust_smooth(x: np.ndarray, radius: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if radius <= 0 or len(x) == 0:
        return x
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - radius)
        hi = min(len(x), i + radius + 1)
        out[i] = float(np.median(x[lo:hi]))
    return out


def longest_true_segment(mask: np.ndarray, min_len: int) -> Tuple[int | None, int | None]:
    mask = np.asarray(mask, dtype=bool)
    best_start = None
    best_end = None
    best_len = 0
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            end = i
            seg_len = end - start
            if seg_len >= min_len and seg_len > best_len:
                best_start = start
                best_end = end
                best_len = seg_len
            start = None
    return best_start, best_end


def all_true_segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: List[Tuple[int, int]] = []
    start = None
    for i, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            end = i
            if end - start >= min_len:
                segments.append((start, end))
            start = None
    return segments


def marker_signal(marker: np.ndarray) -> np.ndarray:
    marker = np.asarray(marker, dtype=np.float32)
    mag = np.linalg.norm(marker, axis=-1)
    return mag.reshape(len(marker), -1).mean(axis=1).astype(np.float64)


def force_signal(force: np.ndarray) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32)
    if force.ndim != 2 or force.shape[1] < 3:
        return np.zeros(len(force), dtype=np.float64)
    return np.linalg.norm(force[:, :3], axis=1).astype(np.float64)


def adaptive_threshold(sig: np.ndarray, q: float, scale: float, floor: float) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) == 0:
        return floor
    base_n = max(8, int(0.15 * len(sig)))
    baseline = sig[:base_n]
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    robust = med + scale * max(1.4826 * mad, 1e-8)
    quant = float(np.quantile(sig, q))
    return max(floor, min(quant, robust) if quant > floor else robust)


def detect_contact(
    marker: np.ndarray,
    force: np.ndarray,
    *,
    smooth_radius: int,
    min_contact_len: int,
    marker_quantile: float,
    force_quantile: float,
    marker_scale: float,
    force_scale: float,
    marker_floor: float,
    force_floor: float,
) -> Dict[str, Any]:
    marker_mag = robust_smooth(marker_signal(marker), smooth_radius)
    force_mag = robust_smooth(force_signal(force), smooth_radius)
    n = min(len(marker_mag), len(force_mag))
    marker_mag = marker_mag[:n]
    force_mag = force_mag[:n]
    marker_thr = adaptive_threshold(marker_mag, marker_quantile, marker_scale, marker_floor)
    force_thr = adaptive_threshold(force_mag, force_quantile, force_scale, force_floor)
    marker_mask = marker_mag >= marker_thr
    force_mask = force_mag >= force_thr
    contact_mask = marker_mask | force_mask
    segments = all_true_segments(contact_mask, min_contact_len)
    start, end = longest_true_segment(contact_mask, min_contact_len)
    if start is None or end is None:
        # Fall back to strongest local region to avoid dropping an episode from
        # the audit.  This is marked by detection_ok=False in the output.
        score = marker_mag / max(marker_thr, 1e-8) + force_mag / max(force_thr, 1e-8)
        center = int(np.argmax(score)) if len(score) else 0
        half = max(1, min_contact_len // 2)
        start = max(0, center - half)
        end = min(n, start + min_contact_len)
        start = max(0, end - min_contact_len)
        detection_ok = False
    else:
        detection_ok = True
    return {
        "n": int(n),
        "marker_mag": marker_mag,
        "force_mag": force_mag,
        "marker_threshold": float(marker_thr),
        "force_threshold": float(force_thr),
        "contact_mask": contact_mask,
        "segments": segments,
        "contact_start": int(start),
        "contact_end": int(end),
        "detection_ok": bool(detection_ok),
    }


def range_mask(n: int, start: int, end: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    start = max(0, min(n, start))
    end = max(start, min(n, end))
    mask[start:end] = True
    return mask


def mask_metrics(contact: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    contact = np.asarray(contact, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    tp = float(np.sum(contact & candidate))
    fp = float(np.sum(~contact & candidate))
    fn = float(np.sum(contact & ~candidate))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = tp / max(tp + fp + fn, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "candidate_frac": float(np.mean(candidate)) if len(candidate) else 0.0,
        "contact_frac": float(np.mean(contact)) if len(contact) else 0.0,
    }


def load_episode(path: Path, tac_side: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        marker = f[f"observations/tac/{tac_side}/marker_offset"][()].astype(np.float32)
        force_key = f"observations/tac/{tac_side}/force6d"
        if force_key in f:
            force = f[force_key][()].astype(np.float32)
        elif "ft" in f:
            force = f["ft"][()].astype(np.float32)
        else:
            force = np.zeros((len(marker), 6), dtype=np.float32)
    n = min(len(marker), len(force))
    return marker[:n], force[:n]


def collect_audit(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for label, root_str in DEFAULT_DATASETS.items():
        root = Path(root_str)
        for path in sorted(root.glob("episode_*.hdf5")):
            try:
                marker, force = load_episode(path, args.tac_side)
            except (OSError, KeyError):
                continue
            det = detect_contact(
                marker,
                force,
                smooth_radius=args.smooth_radius,
                min_contact_len=args.min_contact_len,
                marker_quantile=args.marker_quantile,
                force_quantile=args.force_quantile,
                marker_scale=args.marker_scale,
                force_scale=args.force_scale,
                marker_floor=args.marker_floor,
                force_floor=args.force_floor,
            )
            n = det["n"]
            contact_start = int(det["contact_start"])
            contact_end = int(det["contact_end"])
            fixed_start = max(args.window - 1, int(n * args.phase_start_frac))
            fixed_end = min(int(n * args.phase_end_frac), n - args.horizon - 1, n - args.action_chunk - 1)
            fixed_end = max(fixed_start, fixed_end)
            contact_mask = range_mask(n, contact_start, contact_end)
            fixed_mask = range_mask(n, fixed_start, fixed_end)
            fixed = mask_metrics(contact_mask, fixed_mask)
            usable_contact_start = max(args.window - 1, contact_start - args.pre_contact_margin)
            usable_contact_end = min(contact_end + args.post_contact_margin, n - args.horizon - 1, n - args.action_chunk - 1)
            usable_contact_end = max(usable_contact_start, usable_contact_end)
            contact_candidate_mask = range_mask(n, usable_contact_start, usable_contact_end)
            contact_candidate = mask_metrics(contact_mask, contact_candidate_mask)

            marker_mag = det["marker_mag"]
            force_mag = det["force_mag"]
            row = {
                "label": label,
                "episode": str(path),
                "n_frames": int(n),
                "detection_ok": bool(det["detection_ok"]),
                "contact_start": contact_start,
                "contact_end": contact_end,
                "contact_start_frac": contact_start / max(n, 1),
                "contact_end_frac": contact_end / max(n, 1),
                "contact_len": int(contact_end - contact_start),
                "contact_frac": float((contact_end - contact_start) / max(n, 1)),
                "marker_threshold": float(det["marker_threshold"]),
                "force_threshold": float(det["force_threshold"]),
                "marker_contact_mean": float(np.mean(marker_mag[contact_start:contact_end])) if contact_end > contact_start else 0.0,
                "force_contact_mean": float(np.mean(force_mag[contact_start:contact_end])) if contact_end > contact_start else 0.0,
                "fixed_start": int(fixed_start),
                "fixed_end": int(fixed_end),
                "fixed_precision": fixed["precision"],
                "fixed_recall": fixed["recall"],
                "fixed_f1": fixed["f1"],
                "fixed_iou": fixed["iou"],
                "contact_candidate_start": int(usable_contact_start),
                "contact_candidate_end": int(usable_contact_end),
                "contact_candidate_precision": contact_candidate["precision"],
                "contact_candidate_recall": contact_candidate["recall"],
                "contact_candidate_f1": contact_candidate["f1"],
                "contact_candidate_iou": contact_candidate["iou"],
            }
            rows.append(row)
            for start in range(usable_contact_start, usable_contact_end):
                candidates.append(
                    {
                        "label": label,
                        "episode": str(path),
                        "start": int(start),
                        "end": int(min(start + args.horizon, n - 1)),
                        "contact_start": contact_start,
                        "contact_end": contact_end,
                        "start_frac": float(start / max(n, 1)),
                    }
                )
    return rows, candidates


def group_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = sorted({r["label"] for r in rows})
    result: Dict[str, Any] = {"all": summarize_rows(rows)}
    for label in labels:
        result[label] = summarize_rows([r for r in rows if r["label"] == label])
    return result


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n_episodes": 0}
    return {
        "n_episodes": int(len(rows)),
        "detection_ok_rate": float(np.mean([bool(r["detection_ok"]) for r in rows])),
        "contact_start_frac": summarize(r["contact_start_frac"] for r in rows),
        "contact_end_frac": summarize(r["contact_end_frac"] for r in rows),
        "contact_frac": summarize(r["contact_frac"] for r in rows),
        "fixed_precision": summarize(r["fixed_precision"] for r in rows),
        "fixed_recall": summarize(r["fixed_recall"] for r in rows),
        "fixed_f1": summarize(r["fixed_f1"] for r in rows),
        "fixed_iou": summarize(r["fixed_iou"] for r in rows),
        "contact_candidate_precision": summarize(r["contact_candidate_precision"] for r in rows),
        "contact_candidate_recall": summarize(r["contact_candidate_recall"] for r in rows),
        "contact_candidate_f1": summarize(r["contact_candidate_f1"] for r in rows),
        "marker_contact_mean": summarize(r["marker_contact_mean"] for r in rows),
        "force_contact_mean": summarize(r["force_contact_mean"] for r in rows),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}"


def plot_summary(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = sorted({r["label"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    data_start = [[r["contact_start_frac"] for r in rows if r["label"] == label] for label in labels]
    data_end = [[r["contact_end_frac"] for r in rows if r["label"] == label] for label in labels]
    data_f1 = [[r["fixed_f1"] for r in rows if r["label"] == label] for label in labels]
    axes[0].boxplot(data_start, labels=labels, showfliers=False)
    axes[0].set_title("Detected contact start fraction")
    axes[0].set_ylim(0, 1)
    axes[1].boxplot(data_end, labels=labels, showfliers=False)
    axes[1].set_title("Detected contact end fraction")
    axes[1].set_ylim(0, 1)
    axes[2].boxplot(data_f1, labels=labels, showfliers=False)
    axes[2].set_title("Fixed 25%-85% contact F1")
    axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Board Contact Window Audit",
        "",
        "This audit estimates contact/wiping windows directly from marker and force signals.",
        "It checks whether fixed phase sampling is a good proxy for actual contact.",
        "",
        "## Protocol",
        "",
        f"- tactile side: `{result['protocol']['tac_side']}`",
        f"- fixed phase fraction: `{result['protocol']['phase_start_frac']}` to `{result['protocol']['phase_end_frac']}`",
        f"- horizon/action chunk/window: `{result['protocol']['horizon']}` / `{result['protocol']['action_chunk']}` / `{result['protocol']['window']}`",
        f"- total episodes audited: `{summary['all']['n_episodes']}`",
        f"- contact candidates exported: `{result['n_contact_candidates']}`",
        "",
        "## Dataset Summary",
        "",
        "| label | episodes | contact start median | contact end median | fixed F1 mean | fixed recall mean | candidate F1 mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in summary.items():
        if label == "all":
            continue
        lines.append(
            "| "
            + f"{label} | {row['n_episodes']} | "
            + f"{fmt(row['contact_start_frac'].get('median'))} | "
            + f"{fmt(row['contact_end_frac'].get('median'))} | "
            + f"{fmt(row['fixed_f1'].get('mean'))} | "
            + f"{fmt(row['fixed_recall'].get('mean'))} | "
            + f"{fmt(row['contact_candidate_f1'].get('mean'))} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        *result["interpretation"],
        "",
        "## Files",
        "",
        f"- per-episode audit: `{result['outputs']['episode_csv']}`",
        f"- contact candidates: `{result['outputs']['candidate_csv']}`",
        f"- plot: `{result['outputs']['plot']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_interpretation(summary: Dict[str, Any]) -> List[str]:
    all_row = summary["all"]
    fixed_f1 = all_row["fixed_f1"].get("mean")
    fixed_recall = all_row["fixed_recall"].get("mean")
    candidate_f1 = all_row["contact_candidate_f1"].get("mean")
    start_med = all_row["contact_start_frac"].get("median")
    end_med = all_row["contact_end_frac"].get("median")
    lines = [
        f"- Detected contact median spans approximately `{fmt(start_med)}` to `{fmt(end_med)}` of each episode.",
        f"- Fixed 25%-85% sampling has mean contact F1 `{fmt(fixed_f1)}` and recall `{fmt(fixed_recall)}`.",
        f"- Contact-aware candidate sampling has mean contact F1 `{fmt(candidate_f1)}`.",
    ]
    if fixed_f1 is not None and candidate_f1 is not None and candidate_f1 > fixed_f1 + 0.08:
        lines.append("- Contact-aware sampling is materially better than fixed phase sampling; use detected contact candidates for the next scorer/DP sampler.")
    else:
        lines.append("- Fixed phase sampling is not obviously broken, but contact-aware candidates are still preferable because they are tied to tactile/force evidence.")
    lines.append("- This audit does not prove policy improvement; it only defines a cleaner training/evaluation sampling protocol.")
    return lines


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, candidates = collect_audit(args)
    summary = group_summary(rows)
    episode_csv = out_dir / "board_contact_window_episode_audit.csv"
    candidate_csv = out_dir / "board_contact_window_candidates.csv"
    plot_path = out_dir / "board_contact_window_audit.png"
    json_path = out_dir / "board_contact_window_audit.json"
    md_path = out_dir / "board_contact_window_audit.md"
    write_csv(episode_csv, rows)
    write_csv(candidate_csv, candidates)
    plot_summary(rows, plot_path)
    result = {
        "purpose": "Estimate board wiping contact windows and audit fixed phase sampling.",
        "inputs": {"datasets": DEFAULT_DATASETS},
        "protocol": {
            "tac_side": args.tac_side,
            "smooth_radius": int(args.smooth_radius),
            "min_contact_len": int(args.min_contact_len),
            "window": int(args.window),
            "horizon": int(args.horizon),
            "action_chunk": int(args.action_chunk),
            "phase_start_frac": float(args.phase_start_frac),
            "phase_end_frac": float(args.phase_end_frac),
            "pre_contact_margin": int(args.pre_contact_margin),
            "post_contact_margin": int(args.post_contact_margin),
            "marker_quantile": float(args.marker_quantile),
            "force_quantile": float(args.force_quantile),
            "marker_scale": float(args.marker_scale),
            "force_scale": float(args.force_scale),
            "marker_floor": float(args.marker_floor),
            "force_floor": float(args.force_floor),
        },
        "summary": summary,
        "n_contact_candidates": int(len(candidates)),
        "interpretation": build_interpretation(summary),
        "outputs": {
            "episode_csv": str(episode_csv),
            "candidate_csv": str(candidate_csv),
            "plot": str(plot_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, result)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {episode_csv}")
    print(f"Saved: {candidate_csv}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--smooth_radius", type=int, default=4)
    parser.add_argument("--min_contact_len", type=int, default=64)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--phase_start_frac", type=float, default=0.25)
    parser.add_argument("--phase_end_frac", type=float, default=0.85)
    parser.add_argument("--pre_contact_margin", type=int, default=16)
    parser.add_argument("--post_contact_margin", type=int, default=16)
    parser.add_argument("--marker_quantile", type=float, default=0.65)
    parser.add_argument("--force_quantile", type=float, default=0.65)
    parser.add_argument("--marker_scale", type=float, default=6.0)
    parser.add_argument("--force_scale", type=float, default=6.0)
    parser.add_argument("--marker_floor", type=float, default=0.05)
    parser.add_argument("--force_floor", type=float, default=0.05)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
