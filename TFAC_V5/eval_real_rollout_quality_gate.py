"""Evaluate real rollout quality for baseline vs TacQuality-guided policies.

This is the real-robot / production-policy validation entry point.  It does
not run a robot and it does not claim validation by itself.  It consumes two
directories of recorded HDF5 rollouts:

  baseline_dir: rollouts from the original DP policy
  guided_dir:   rollouts from TacQualityEnergy-guided DP policy

The script computes task-specific tactile/action quality metrics from saved
HDF5 fields and writes JSON/Markdown/CSV reports.  It is deliberately separate
from offline scorer gates: offline gates prove the guidance stack is ready for
a dry run; this script is the dry-run result evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import h5py
import numpy as np


DEFAULT_OUT = Path("/home/chenshuai/Project/output/real_rollout_quality_gate")


def safe_get(f: h5py.File, key: str) -> Optional[np.ndarray]:
    if key not in f:
        return None
    return np.asarray(f[key][:])


def force_mag(force6: Optional[np.ndarray]) -> np.ndarray:
    if force6 is None or len(force6) == 0:
        return np.zeros(1, dtype=np.float32)
    arr = np.asarray(force6, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None]
    if arr.shape[-1] >= 3:
        return np.linalg.norm(arr[..., :3], axis=-1).astype(np.float32)
    return np.abs(arr.reshape(-1)).astype(np.float32)


def marker_mag(marker: Optional[np.ndarray]) -> np.ndarray:
    if marker is None or len(marker) == 0:
        return np.zeros(1, dtype=np.float32)
    arr = np.asarray(marker, dtype=np.float32)
    return np.linalg.norm(arr.reshape(len(arr), -1), axis=1).astype(np.float32)


def delta_norm(seq: Optional[np.ndarray]) -> np.ndarray:
    if seq is None or len(seq) < 2:
        return np.zeros(1, dtype=np.float32)
    arr = np.asarray(seq, dtype=np.float32).reshape(len(seq), -1)
    return np.linalg.norm(np.diff(arr, axis=0), axis=1).astype(np.float32)


def accel_norm(seq: Optional[np.ndarray]) -> np.ndarray:
    if seq is None or len(seq) < 3:
        return np.zeros(1, dtype=np.float32)
    arr = np.asarray(seq, dtype=np.float32).reshape(len(seq), -1)
    d = np.diff(arr, axis=0)
    return np.linalg.norm(np.diff(d, axis=0), axis=1).astype(np.float32)


def finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else 0.0


def finite_percentile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else 0.0


def discover_hdf5(root: Path) -> List[Path]:
    if root.is_file() and root.suffix in {".hdf5", ".h5"}:
        return [root]
    return sorted([*root.rglob("*.hdf5"), *root.rglob("*.h5")])


@dataclass
class Rollout:
    path: str
    group: str
    stem: str
    task: str
    metrics: Dict[str, float]


def episode_metrics(path: Path, group: str, task: str) -> Rollout:
    with h5py.File(path, "r") as f:
        ft = safe_get(f, "ft")
        left_force = safe_get(f, "observations/tac/left/force6d")
        right_force = safe_get(f, "observations/tac/right/force6d")
        left_marker = safe_get(f, "observations/tac/left/marker_offset")
        right_marker = safe_get(f, "observations/tac/right/marker_offset")
        eef = safe_get(f, "actions/eef_abs")
        joint = safe_get(f, "actions/joint_abs")
        success_attr = f.attrs.get("success", None)
        stopped_attr = f.attrs.get("stopped_early", None)

    force_sources = [x for x in [ft, left_force, right_force] if x is not None]
    if force_sources:
        fmag = force_mag(force_sources[0])
    else:
        fmag = np.zeros(1, dtype=np.float32)
    lfmag = force_mag(left_force)
    rfmag = force_mag(right_force)
    lmag = marker_mag(left_marker)
    rmag = marker_mag(right_marker)
    both_marker = np.maximum(lmag[: min(len(lmag), len(rmag))], rmag[: min(len(lmag), len(rmag))])
    if len(both_marker) == 0:
        both_marker = np.zeros(1, dtype=np.float32)

    action = joint if joint is not None else eef
    speed = delta_norm(action)
    accel = accel_norm(action)
    marker_delta = np.maximum(delta_norm(left_marker), delta_norm(right_marker))
    force_delta = np.abs(np.diff(fmag)) if len(fmag) > 1 else np.zeros(1, dtype=np.float32)
    force_jerk = np.abs(np.diff(force_delta)) if len(force_delta) > 1 else np.zeros(1, dtype=np.float32)

    n_steps = int(max(len(fmag), len(lmag), len(rmag), len(speed)))
    out: Dict[str, float] = {
        "n_steps": float(n_steps),
        "force_mean": finite_mean(fmag),
        "force_p50": finite_percentile(fmag, 50),
        "force_p95": finite_percentile(fmag, 95),
        "force_max": finite_percentile(fmag, 100),
        "left_force_mean": finite_mean(lfmag),
        "right_force_mean": finite_mean(rfmag),
        "force_delta_mean": finite_mean(force_delta),
        "force_delta_p90": finite_percentile(force_delta, 90),
        "force_jerk_mean": finite_mean(force_jerk),
        "marker_mean": finite_mean(both_marker),
        "marker_p95": finite_percentile(both_marker, 95),
        "marker_max": finite_percentile(both_marker, 100),
        "marker_delta_mean": finite_mean(marker_delta),
        "marker_delta_p90": finite_percentile(marker_delta, 90),
        "action_speed_mean": finite_mean(speed),
        "action_speed_p90": finite_percentile(speed, 90),
        "action_accel_mean": finite_mean(accel),
        "action_accel_p90": finite_percentile(accel, 90),
        "success_attr": float(success_attr) if success_attr is not None else math.nan,
        "stopped_early_attr": float(stopped_attr) if stopped_attr is not None else math.nan,
    }
    if task == "insertion":
        # Conservative tactile-risk proxy for socket insertion.  Human/task
        # success labels should override this once available.
        out["impact_proxy"] = out["marker_delta_p90"] + 0.25 * out["force_delta_p90"]
        out["risk_proxy"] = out["marker_p95"] + 0.35 * out["impact_proxy"]
    elif task == "board":
        out["roughness_proxy"] = (
            out["force_delta_mean"]
            + 0.50 * out["force_jerk_mean"]
            + 0.35 * out["marker_delta_mean"]
            + 0.25 * out["action_accel_mean"]
        )
    else:
        raise ValueError(f"Unknown task {task!r}")
    return Rollout(path=str(path), group=group, stem=path.stem, task=task, metrics=out)


def fit_reference(rows: List[Rollout], task: str) -> Dict[str, float]:
    vals = {k: np.array([r.metrics.get(k, math.nan) for r in rows], dtype=np.float64) for k in rows[0].metrics}
    ref: Dict[str, float] = {}
    for key, arr in vals.items():
        arr = arr[np.isfinite(arr)]
        if len(arr):
            ref[f"{key}_median"] = float(np.median(arr))
            ref[f"{key}_mad"] = float(np.median(np.abs(arr - np.median(arr))) * 1.4826 + 1e-8)
            ref[f"{key}_q20"] = float(np.quantile(arr, 0.20))
            ref[f"{key}_q55"] = float(np.quantile(arr, 0.55))
            ref[f"{key}_q80"] = float(np.quantile(arr, 0.80))
            ref[f"{key}_q90"] = float(np.quantile(arr, 0.90))
            ref[f"{key}_q95"] = float(np.quantile(arr, 0.95))
    if task == "board":
        f = vals["force_mean"]
        f = f[np.isfinite(f)]
        ref["board_target_force"] = float(np.quantile(f, 0.55)) if len(f) else 0.0
        ref["board_force_sigma"] = float(max(np.quantile(f, 0.75) - np.quantile(f, 0.25), 1e-6)) if len(f) else 1.0
    return ref


def score_rollout(r: Rollout, ref: Dict[str, float]) -> None:
    m = r.metrics
    if r.task == "board":
        target = ref.get("board_target_force", m["force_mean"])
        sigma = max(ref.get("board_force_sigma", 1.0), 1e-6)
        force_band = math.exp(-0.5 * ((m["force_mean"] - target) / sigma) ** 2)
        rough_terms = [
            max(0.0, (m["force_delta_mean"] - ref.get("force_delta_mean_median", 0.0)) / ref.get("force_delta_mean_mad", 1.0)),
            0.50 * max(0.0, (m["force_jerk_mean"] - ref.get("force_jerk_mean_median", 0.0)) / ref.get("force_jerk_mean_mad", 1.0)),
            0.35 * max(0.0, (m["marker_delta_mean"] - ref.get("marker_delta_mean_median", 0.0)) / ref.get("marker_delta_mean_mad", 1.0)),
            0.25 * max(0.0, (m["action_accel_mean"] - ref.get("action_accel_mean_median", 0.0)) / ref.get("action_accel_mean_mad", 1.0)),
            0.50 * max(0.0, (m["force_p95"] - ref.get("force_p95_q90", m["force_p95"])) / ref.get("force_p95_mad", 1.0)),
        ]
        rough = float(sum(rough_terms))
        m["force_band_score"] = float(force_band)
        m["smoothness_score"] = float(math.exp(-0.55 * rough))
        m["quality_score"] = float(np.clip(force_band * m["smoothness_score"], 0.0, 1.0))
        m["too_light_flag"] = float(m["force_mean"] < ref.get("force_mean_q20", -math.inf))
        m["too_heavy_flag"] = float(m["force_mean"] > ref.get("force_mean_q80", math.inf) or m["force_p95"] > ref.get("force_p95_q90", math.inf))
        m["rough_flag"] = float(rough > 1.0)
    else:
        risk_z = (m["risk_proxy"] - ref.get("risk_proxy_median", 0.0)) / ref.get("risk_proxy_mad", 1.0)
        impact_z = (m["impact_proxy"] - ref.get("impact_proxy_median", 0.0)) / ref.get("impact_proxy_mad", 1.0)
        force_z = (m["force_p95"] - ref.get("force_p95_median", 0.0)) / ref.get("force_p95_mad", 1.0)
        risk = max(0.0, risk_z) + 0.50 * max(0.0, impact_z) + 0.20 * max(0.0, force_z)
        m["risk_score"] = float(risk)
        m["quality_score"] = float(math.exp(-0.55 * risk))
        m["risk_flag"] = float(risk > 1.0)


def aggregate(rows: List[Rollout]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    keys = sorted({k for r in rows for k in r.metrics})
    out: Dict[str, Any] = {"n": len(rows)}
    for key in keys:
        vals = np.array([r.metrics.get(key, math.nan) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }
    return out


def paired_deltas(baseline: List[Rollout], guided: List[Rollout]) -> Dict[str, Any]:
    b_map = {r.stem: r for r in baseline}
    g_map = {r.stem: r for r in guided}
    common = sorted(set(b_map) & set(g_map))
    if not common:
        return {"n_pairs": 0, "note": "No matching stems; aggregate comparison only."}
    keys = sorted(set().union(*(b_map[k].metrics for k in common), *(g_map[k].metrics for k in common)))
    out: Dict[str, Any] = {"n_pairs": len(common)}
    for key in keys:
        vals = []
        for stem in common:
            bv = b_map[stem].metrics.get(key, math.nan)
            gv = g_map[stem].metrics.get(key, math.nan)
            if np.isfinite(bv) and np.isfinite(gv):
                vals.append(gv - bv)
        if vals:
            arr = np.array(vals, dtype=np.float64)
            out[f"{key}_delta"] = {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p05": float(np.percentile(arr, 5)),
                "p95": float(np.percentile(arr, 95)),
                "guided_better_rate": float(np.mean(arr > 0.0)) if key == "quality_score" else None,
            }
    return out


def decision(task: str, summary: Dict[str, Any], min_episodes: int) -> Dict[str, Any]:
    guided = summary["guided"]
    baseline = summary["baseline"]
    paired = summary["paired"]
    if baseline.get("n", 0) < min_episodes or guided.get("n", 0) < min_episodes:
        return {
            "production_validation_pass": False,
            "reason": (
                f"Insufficient rollout count: baseline_n={baseline.get('n', 0)}, "
                f"guided_n={guided.get('n', 0)}, min_episodes={min_episodes}."
            ),
            "quality_improved": None,
            "paired_quality_delta_mean": None,
            "paired_quality_guided_better_rate": None,
        }
    q_delta = paired.get("quality_score_delta", {})
    guided_quality = guided.get("quality_score", {}).get("mean")
    baseline_quality = baseline.get("quality_score", {}).get("mean")
    if guided_quality is None or baseline_quality is None:
        return {"production_validation_pass": False, "reason": "Missing quality_score metrics."}
    quality_improved = guided_quality > baseline_quality
    paired_ok = paired.get("n_pairs", 0) == 0 or q_delta.get("mean", 0.0) > 0.0
    paired_note = None
    if paired.get("n_pairs", 0) == 0:
        paired_note = "No paired stems; decision uses aggregate group comparison only."
    if task == "board":
        bad_rate_ok = guided.get("too_heavy_flag", {}).get("mean", 1.0) <= baseline.get("too_heavy_flag", {}).get("mean", 1.0) + 0.05
        rough_ok = guided.get("rough_flag", {}).get("mean", 1.0) <= baseline.get("rough_flag", {}).get("mean", 1.0) + 0.05
        passed = quality_improved and paired_ok and bad_rate_ok and rough_ok
        reason = "board quality improved without increasing too-heavy/rough rates beyond tolerance"
    else:
        risk_ok = guided.get("risk_score", {}).get("mean", 1.0) <= baseline.get("risk_score", {}).get("mean", 1.0)
        flag_ok = guided.get("risk_flag", {}).get("mean", 1.0) <= baseline.get("risk_flag", {}).get("mean", 1.0) + 0.05
        passed = quality_improved and paired_ok and risk_ok and flag_ok
        reason = "insertion quality improved while risk proxy did not increase"
    return {
        "production_validation_pass": bool(passed),
        "reason": reason,
        "quality_improved": bool(quality_improved),
        "paired_quality_delta_mean": q_delta.get("mean"),
        "paired_quality_guided_better_rate": q_delta.get("guided_better_rate"),
        "paired_note": paired_note,
    }


def write_csv(rows: Iterable[Rollout], path: Path) -> None:
    rows = list(rows)
    keys = sorted({k for r in rows for k in r.metrics})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "task", "stem", "path"] + keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({"group": r.group, "task": r.task, "stem": r.stem, "path": r.path, **r.metrics})


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Real Rollout Quality Gate",
        "",
        f"- task: `{result['task']}`",
        f"- production_validation_pass: `{result['decision']['production_validation_pass']}`",
        f"- reason: {result['decision']['reason']}",
        f"- baseline_n: `{result['summary']['baseline']['n']}`",
        f"- guided_n: `{result['summary']['guided']['n']}`",
        f"- paired_n: `{result['summary']['paired'].get('n_pairs', 0)}`",
        "",
        "## Decision",
        "",
        "```json",
        json.dumps(result["decision"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(result["summary"], ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["insertion", "board"], required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--guided_dir", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--min_episodes", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_paths = discover_hdf5(Path(args.baseline_dir))
    guided_paths = discover_hdf5(Path(args.guided_dir))
    if not baseline_paths:
        raise FileNotFoundError(f"No HDF5 files under baseline_dir={args.baseline_dir}")
    if not guided_paths:
        raise FileNotFoundError(f"No HDF5 files under guided_dir={args.guided_dir}")

    baseline = [episode_metrics(p, "baseline", args.task) for p in baseline_paths]
    guided = [episode_metrics(p, "guided", args.task) for p in guided_paths]
    ref = fit_reference(baseline, args.task)
    for row in baseline + guided:
        score_rollout(row, ref)

    summary = {
        "baseline": aggregate(baseline),
        "guided": aggregate(guided),
        "paired": paired_deltas(baseline, guided),
        "reference": ref,
    }
    result = {
        "task": args.task,
        "baseline_dir": str(Path(args.baseline_dir)),
        "guided_dir": str(Path(args.guided_dir)),
        "n_baseline_files": len(baseline_paths),
        "n_guided_files": len(guided_paths),
        "summary": summary,
        "decision": decision(args.task, summary, args.min_episodes),
        "note": (
            "This report evaluates recorded rollout consequences.  It should be "
            "used after collecting baseline and TacQuality-guided robot runs."
        ),
    }

    tag = args.tag or f"{args.task}_baseline_vs_guided"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_rollout_quality_gate.json"
    md_path = out_dir / "real_rollout_quality_gate.md"
    csv_path = out_dir / "real_rollout_episode_metrics.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    write_csv([*baseline, *guided], csv_path)
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
