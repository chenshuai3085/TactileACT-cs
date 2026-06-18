#!/usr/bin/env python3
"""Audit 260617 board data against the current ForceBand TacQuality scorer.

This does not train a new scorer.  It checks whether new 260617 wiping windows
look compatible with the force-band scorer trained on 260609/260610 labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_energy.eval_board_force_band_scorer import (  # noqa: E402
    BOARD_DATASETS,
    action_proxy_np,
    force_proxy_np,
    finite_corr,
    marker_proxy_np,
    spearman_simple,
    summarize,
    window_ending,
)
from TFAC_V5.tac_quality_energy.force_band_runtime import (  # noqa: E402
    DEFAULT_CKPT,
    ForceBandTacQualityEnergyRuntime,
)


DEFAULT_NEW_DATA = Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/board_260617_forceband_distribution_audit")


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


def chunk_from(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    end = min(len(arr), start + length)
    chunk = arr[start:end]
    if len(chunk) == 0:
        idx = max(0, min(start, len(arr) - 1))
        chunk = arr[idx : idx + 1]
    if len(chunk) < length:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], length - len(chunk), axis=0)], axis=0)
    return chunk.astype(np.float32)


def collect_windows(
    roots: Dict[str, Path],
    *,
    tac_side: str,
    window: int,
    action_chunk: int,
    samples_per_episode: int,
    phase_start_frac: float,
    phase_end_frac: float,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for label, root in roots.items():
        for path in sorted(root.glob("episode_*.hdf5")):
            try:
                with h5py.File(path, "r") as f:
                    left = f[f"observations/tac/{tac_side}/marker_offset"][()].astype(np.float32)
                    other_side = "right" if tac_side == "left" else "left"
                    right_key = f"observations/tac/{other_side}/marker_offset"
                    right = f[right_key][()].astype(np.float32) if right_key in f else left
                    force = f[f"observations/tac/{tac_side}/force6d"][()].astype(np.float32)
                    joint = f["actions/joint_abs"][()].astype(np.float32)
                    eef = f["actions/eef_abs"][()].astype(np.float32) if "actions/eef_abs" in f else joint[:, :6]
            except (OSError, KeyError):
                continue

            length = min(len(left), len(right), len(force), len(joint), len(eef))
            min_start = max(window - 1, int(length * phase_start_frac))
            max_start = min(int(length * phase_end_frac), length - action_chunk - 1)
            if max_start <= min_start:
                continue
            candidates = np.arange(min_start, max_start, dtype=np.int64)
            if samples_per_episode > 0 and len(candidates) > samples_per_episode:
                starts = np.sort(rng.choice(candidates, samples_per_episode, replace=False))
            else:
                starts = candidates
            for start in starts:
                end = min(int(start) + action_chunk - 1, length - 1)
                left_win = window_ending(left, end, window)
                right_win = window_ending(right, end, window)
                force_win = window_ending(force, end, window)
                joint_chunk = chunk_from(joint, int(start), action_chunk)
                eef_chunk = chunk_from(eef, int(start), action_chunk)
                rows.append(
                    {
                        "label": label,
                        "episode": str(path),
                        "start": int(start),
                        "end": int(end),
                        "left_marker": left_win,
                        "right_marker": right_win,
                        "force": force_win,
                        "joint_action": joint_chunk,
                        "eef_action": eef_chunk,
                    }
                )
    return rows


def marker_action_features(rows: List[Dict[str, Any]]) -> np.ndarray:
    left = np.stack([marker_proxy_np(r["left_marker"]) for r in rows])
    right = np.stack([marker_proxy_np(r["right_marker"]) for r in rows])
    marker_both = np.concatenate([left, right, np.abs(left - right)], axis=1)
    joint = np.stack([action_proxy_np(r["joint_action"], 7) for r in rows])
    eef = np.stack([action_proxy_np(r["eef_action"], 6) for r in rows])
    return np.concatenate([marker_both, joint, eef], axis=1).astype(np.float32)


def force_features(rows: List[Dict[str, Any]]) -> np.ndarray:
    return np.stack([force_proxy_np(r["force"]) for r in rows]).astype(np.float32)


def scorer_outputs(runtime: ForceBandTacQualityEnergyRuntime, rows: List[Dict[str, Any]], batch_size: int) -> Dict[str, np.ndarray]:
    chunks: Dict[str, List[np.ndarray]] = {
        "quality": [],
        "p_good": [],
        "reason_positive": [],
        "energy_clipped": [],
        "profile": [],
        "reason_pred": [],
    }
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        left = torch.from_numpy(np.stack([r["left_marker"] for r in batch])).to(runtime.device)
        right = torch.from_numpy(np.stack([r["right_marker"] for r in batch])).to(runtime.device)
        joint = torch.from_numpy(np.stack([r["joint_action"] for r in batch])).to(runtime.device)
        eef = torch.from_numpy(np.stack([r["eef_action"] for r in batch])).to(runtime.device)
        with torch.no_grad():
            out = runtime(left, right, eef_action_seq=eef, joint_action_seq=joint)
            chunks["quality"].append(out["quality_score"].detach().cpu().numpy())
            chunks["p_good"].append(out["p_good"].detach().cpu().numpy())
            chunks["reason_positive"].append(out["reason_prob"][:, runtime.reason_to_id.get("positive", 1)].detach().cpu().numpy())
            chunks["energy_clipped"].append(out["energy_clipped"].detach().cpu().numpy())
            profile = 0.5 * out["quality_logit"] + 0.25 * out["good_margin"] + 0.25 * out["reason_margin"]
            chunks["profile"].append(profile.detach().cpu().numpy())
            chunks["reason_pred"].append(out["reason_prob"].argmax(dim=-1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in chunks.items()}


def summarize_by_label(labels: np.ndarray, values: np.ndarray) -> Dict[str, Dict[str, Any]]:
    return {label: summarize(values[labels == label]) for label in sorted(set(labels.tolist()))}


def score_alignment(labels: np.ndarray, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    modes = ["scorer_quality", "p_good", "reason_positive", "energy_clipped", "profile"]
    out: Dict[str, Any] = {}
    for subset_name, mask in [("all", np.ones(len(labels), dtype=bool))] + [
        (label, labels == label) for label in sorted(set(labels.tolist()))
    ]:
        out[subset_name] = {}
        for mode in modes:
            out[subset_name][mode] = {
                "pearson_vs_physical_quality": finite_corr(arrays[mode][mask], arrays["physical_quality"][mask]),
                "spearman_vs_physical_quality": spearman_simple(arrays[mode][mask], arrays["physical_quality"][mask]),
            }
    return out


def quantile_position(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    if len(reference) == 0:
        return np.full(len(values), np.nan, dtype=np.float64)
    return np.searchsorted(reference, values, side="right") / float(len(reference))


def physical_quality_from_old_positive(labels: np.ndarray, force: np.ndarray, marker_left: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    pos = labels == "positive"
    force_mag = force[:, 0].astype(np.float64)
    force_delta = force[:, 7].astype(np.float64)
    marker_delta = marker_left[:, 12].astype(np.float64)
    if not np.any(pos):
        raise RuntimeError("Need old positive windows to build physical quality reference.")

    center = float(np.median(force_mag[pos]))
    mad = float(np.median(np.abs(force_mag[pos] - center)))
    sigma = max(1.4826 * mad, float(np.std(force_mag[pos])), 0.75)
    delta_ref = max(float(np.quantile(force_delta[pos], 0.75)), 0.05)
    marker_ref = max(float(np.quantile(marker_delta[pos], 0.75)), 1e-4)

    band_score = np.exp(-0.5 * ((force_mag - center) / sigma) ** 2)
    smooth_score = np.exp(-force_delta / delta_ref)
    marker_smooth = np.exp(-marker_delta / marker_ref)
    physical_quality = 0.62 * band_score + 0.25 * smooth_score + 0.13 * marker_smooth
    physical_quality = np.clip(physical_quality, 0.0, 1.0).astype(np.float32)
    ref = {
        "definition": "0.62*old_positive_force_band + 0.25*force_smooth + 0.13*marker_smooth",
        "force_mag_center_positive_median": center,
        "force_mag_sigma": sigma,
        "force_delta_ref_positive_q75": delta_ref,
        "marker_delta_ref_positive_q75": marker_ref,
    }
    return physical_quality, ref


def write_csv(path: Path, rows: List[Dict[str, Any]], arrays: Dict[str, np.ndarray]) -> None:
    fieldnames = [
        "label",
        "episode",
        "start",
        "end",
        "force_mag",
        "force_delta",
        "fz_abs",
        "marker_delta",
        "physical_quality",
        "scorer_quality",
        "p_good",
        "reason_positive",
        "energy_clipped",
        "profile",
        "reason_pred",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {k: row.get(k) for k in ["label", "episode", "start", "end"]}
            out.update({k: arrays[k][i] for k in fieldnames[4:]})
            writer.writerow(out)


def plot_distributions(labels: np.ndarray, arrays: Dict[str, np.ndarray], out_path: Path) -> None:
    metrics = [
        "force_mag",
        "force_delta",
        "fz_abs",
        "marker_delta",
        "physical_quality",
        "scorer_quality",
        "p_good",
        "energy_clipped",
        "profile",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), dpi=150)
    for ax, metric in zip(axes.reshape(-1), metrics):
        for label in sorted(set(labels.tolist())):
            vals = arrays[metric][labels == label]
            if len(vals):
                ax.hist(vals, bins=40, alpha=0.45, density=True, label=label)
        ax.set_title(metric)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# 260617 Board ForceBand Distribution Audit",
        "",
        "目的：检查 260617-only 擦黑板数据是否落在当前 ForceBand TacQuality scorer 的训练分布附近。",
        "",
        "## Inputs",
        "",
        f"- new data: `{result['inputs']['new_data']}`",
        f"- scorer ckpt: `{result['inputs']['scorer_ckpt']}`",
        f"- reference labeled datasets: 260609/260610 positive, too_small, too_large, oscillate",
        f"- sampled phase fraction: `{result['protocol']['phase_start_frac']}` to `{result['protocol']['phase_end_frac']}`",
        f"- samples per episode: `{result['protocol']['samples_per_episode']}`",
        "",
        "## Counts",
        "",
    ]
    for label, n in result["label_counts"].items():
        lines.append(f"- {label}: `{n}` windows")
    lines.extend(["", "## Key Distributions", ""])
    lines.append("| label | force_mag mean | force_delta mean | marker_delta mean | physical_quality mean | scorer_quality mean | p_good mean | reason_positive mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label in sorted(result["label_counts"]):
        row = result["summary_by_label"][label]
        lines.append(
            f"| {label} | {fmt(row['force_mag'].get('mean'))} | {fmt(row['force_delta'].get('mean'))} | "
            f"{fmt(row['marker_delta'].get('mean'))} | {fmt(row['physical_quality'].get('mean'))} | "
            f"{fmt(row['scorer_quality'].get('mean'))} | "
            f"{fmt(row['p_good'].get('mean'))} | {fmt(row['reason_positive'].get('mean'))} |"
        )
    lines.extend(["", "Physical quality uses the old positive force-band reference:", ""])
    ref = result["physical_quality_reference"]
    lines.append(f"- definition: `{ref['definition']}`")
    lines.append(f"- old positive force center: `{fmt(ref['force_mag_center_positive_median'])}`")
    lines.append(f"- force sigma: `{fmt(ref['force_mag_sigma'])}`")
    lines.append(f"- force-delta ref: `{fmt(ref['force_delta_ref_positive_q75'])}`")
    lines.append(f"- marker-delta ref: `{fmt(ref['marker_delta_ref_positive_q75'])}`")
    lines.extend(["", "## 260617 Relative To Old Positive Windows", ""])
    qp = result["new_vs_old_positive_quantiles"]
    lines.append(f"- force_mag quantile mean: `{fmt(qp['force_mag_quantile']['mean'])}`")
    lines.append(f"- force_delta quantile mean: `{fmt(qp['force_delta_quantile']['mean'])}`")
    lines.append(f"- marker_delta quantile mean: `{fmt(qp['marker_delta_quantile']['mean'])}`")
    lines.append(f"- physical_quality quantile mean: `{fmt(qp['physical_quality_quantile']['mean'])}`")
    lines.append(f"- scorer_quality quantile mean: `{fmt(qp['scorer_quality_quantile']['mean'])}`")
    lines.append(f"- p_good quantile mean: `{fmt(qp['p_good_quantile']['mean'])}`")
    lines.extend(["", "## Score Mode Alignment With Physical Quality", ""])
    align = result["score_physical_alignment"]
    lines.append("| subset | scorer_quality rho | p_good rho | reason_positive rho | energy_clipped rho | profile rho |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for subset in ["new_260617", "positive", "all"]:
        row = align.get(subset, {})
        lines.append(
            f"| {subset} | {fmt(row.get('scorer_quality', {}).get('spearman_vs_physical_quality'))} | "
            f"{fmt(row.get('p_good', {}).get('spearman_vs_physical_quality'))} | "
            f"{fmt(row.get('reason_positive', {}).get('spearman_vs_physical_quality'))} | "
            f"{fmt(row.get('energy_clipped', {}).get('spearman_vs_physical_quality'))} | "
            f"{fmt(row.get('profile', {}).get('spearman_vs_physical_quality'))} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.extend(result["interpretation"])
    lines.extend(["", "## Artifacts", ""])
    for key, value in result["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = {label: Path(root) for label, root in BOARD_DATASETS.items()}
    roots["new_260617"] = Path(args.new_data)
    rows = collect_windows(
        roots,
        tac_side=args.tac_side,
        window=args.window,
        action_chunk=args.action_chunk,
        samples_per_episode=args.samples_per_episode,
        phase_start_frac=args.phase_start_frac,
        phase_end_frac=args.phase_end_frac,
        seed=args.seed,
    )
    labels = np.asarray([r["label"] for r in rows], dtype=str)
    if "new_260617" not in set(labels.tolist()):
        raise RuntimeError("No 260617 windows collected.")

    force = force_features(rows)
    marker_left = np.stack([marker_proxy_np(r["left_marker"]) for r in rows])
    runtime = ForceBandTacQualityEnergyRuntime(str(args.scorer_ckpt), device=args.device)
    scores = scorer_outputs(runtime, rows, args.batch_size)
    physical_quality, physical_ref = physical_quality_from_old_positive(labels, force, marker_left)
    arrays: Dict[str, np.ndarray] = {
        "force_mag": force[:, 0],
        "force_delta": force[:, 7],
        "fz_abs": force[:, 3],
        "marker_delta": marker_left[:, 12],
        "physical_quality": physical_quality,
        **scores,
    }
    arrays["scorer_quality"] = arrays.pop("quality")
    arrays["reason_pred"] = arrays["reason_pred"].astype(np.int64)

    out_npz = out_dir / "board_260617_forceband_distribution_arrays.npz"
    np.savez_compressed(
        out_npz,
        labels=labels,
        episode=np.asarray([r["episode"] for r in rows], dtype=object),
        start=np.asarray([r["start"] for r in rows], dtype=np.int64),
        end=np.asarray([r["end"] for r in rows], dtype=np.int64),
        **arrays,
    )
    out_csv = out_dir / "board_260617_forceband_distribution_windows.csv"
    write_csv(out_csv, rows, arrays)
    out_png = out_dir / "board_260617_forceband_distribution.png"
    plot_distributions(labels, arrays, out_png)

    summary_by_label: Dict[str, Dict[str, Any]] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        summary_by_label[label] = {k: summarize(v[mask]) for k, v in arrays.items() if k != "reason_pred"}

    pos = labels == "positive"
    new = labels == "new_260617"
    quantiles = {
        f"{metric}_quantile": summarize(quantile_position(arrays[metric][pos], arrays[metric][new]))
        for metric in ["force_mag", "force_delta", "marker_delta", "physical_quality", "scorer_quality", "p_good"]
    }
    alignment = score_alignment(labels, arrays)

    new_scorer_quality = float(np.mean(arrays["scorer_quality"][new]))
    pos_scorer_quality = float(np.mean(arrays["scorer_quality"][pos])) if np.any(pos) else float("nan")
    new_physical_quality = float(np.mean(arrays["physical_quality"][new]))
    pos_physical_quality = float(np.mean(arrays["physical_quality"][pos])) if np.any(pos) else float("nan")
    new_p_good = float(np.mean(arrays["p_good"][new]))
    pos_p_good = float(np.mean(arrays["p_good"][pos])) if np.any(pos) else float("nan")
    interp = [
        f"- Current scorer rates 260617 windows with mean scorer_quality `{fmt(new_scorer_quality)}` and p_good `{fmt(new_p_good)}`.",
        f"- Old positive reference has mean scorer_quality `{fmt(pos_scorer_quality)}` and p_good `{fmt(pos_p_good)}` under the same scorer.",
        f"- Old-positive force-band physical quality is `{fmt(new_physical_quality)}` for 260617 and `{fmt(pos_physical_quality)}` for old positive windows.",
    ]
    best_new_mode = max(
        ["scorer_quality", "p_good", "reason_positive", "energy_clipped", "profile"],
        key=lambda m: float(alignment["new_260617"][m]["spearman_vs_physical_quality"] or -2.0),
    )
    interp.append(
        f"- On 260617 windows, the score mode most aligned with physical_quality by Spearman is `{best_new_mode}` "
        f"with rho `{fmt(alignment['new_260617'][best_new_mode]['spearman_vs_physical_quality'])}`."
    )
    if new_scorer_quality >= pos_scorer_quality - 0.05 and new_p_good >= pos_p_good - 0.05:
        interp.append("- 260617 appears compatible with the old positive distribution under the current scorer.")
    else:
        interp.append("- 260617 is shifted relative to the old positive distribution; use the old scorer for guidance cautiously and verify with real force curves.")
    if float(np.mean(quantile_position(arrays["force_mag"][pos], arrays["force_mag"][new]))) < 0.2:
        interp.append("- 260617 force magnitude is mostly below the old positive force band, suggesting the scorer may view it as too-light contact.")
    if float(np.mean(quantile_position(arrays["force_delta"][pos], arrays["force_delta"][new]))) > 0.8:
        interp.append("- 260617 force delta is high relative to old positive windows, suggesting contact instability risk.")

    result: Dict[str, Any] = {
        "purpose": "Audit whether 260617 board data matches current ForceBand scorer/reference distribution.",
        "inputs": {
            "new_data": str(args.new_data),
            "scorer_ckpt": str(args.scorer_ckpt),
            "reference_datasets": {k: str(v) for k, v in BOARD_DATASETS.items()},
        },
        "protocol": {
            "window": int(args.window),
            "action_chunk": int(args.action_chunk),
            "samples_per_episode": int(args.samples_per_episode),
            "phase_start_frac": float(args.phase_start_frac),
            "phase_end_frac": float(args.phase_end_frac),
            "tac_side": args.tac_side,
        },
        "label_counts": {label: int(np.sum(labels == label)) for label in sorted(set(labels.tolist()))},
        "summary_by_label": summary_by_label,
        "physical_quality_reference": physical_ref,
        "new_vs_old_positive_quantiles": quantiles,
        "score_physical_alignment": alignment,
        "interpretation": interp,
        "artifacts": {
            "json": str(out_dir / "board_260617_forceband_distribution_audit.json"),
            "markdown": str(out_dir / "board_260617_forceband_distribution_audit.md"),
            "csv": str(out_csv),
            "npz": str(out_npz),
            "plot": str(out_png),
        },
    }
    json_path = out_dir / "board_260617_forceband_distribution_audit.json"
    md_path = out_dir / "board_260617_forceband_distribution_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {out_png}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_data", type=Path, default=DEFAULT_NEW_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--scorer_ckpt", type=Path, default=Path(DEFAULT_CKPT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tac_side", default="left")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_chunk", type=int, default=16)
    parser.add_argument("--samples_per_episode", type=int, default=12)
    parser.add_argument("--phase_start_frac", type=float, default=0.25)
    parser.add_argument("--phase_end_frac", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
