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


def resolve_rollout_path(value: str, root: Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    direct = root / p
    if direct.exists():
        return direct
    matches = list(root.rglob(value))
    if len(matches) == 1:
        return matches[0]
    stem_matches = [m for m in root.rglob("*.hdf5") if m.stem == value]
    stem_matches += [m for m in root.rglob("*.h5") if m.stem == value]
    if len(stem_matches) == 1:
        return stem_matches[0]
    raise FileNotFoundError(f"Cannot resolve rollout path {value!r} under {root}")


def read_pairing_csv(path: Path, baseline_root: Path, guided_root: Path) -> List[Dict[str, Path]]:
    pairs: List[Dict[str, Path]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"baseline", "guided"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pairing_csv must contain columns {sorted(required)}, missing {sorted(missing)}")
        for i, row in enumerate(reader):
            pair_id = row.get("pair_id") or f"pair_{i:04d}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "baseline": resolve_rollout_path(row["baseline"], baseline_root),
                    "guided": resolve_rollout_path(row["guided"], guided_root),
                }
            )
    if not pairs:
        raise ValueError(f"pairing_csv has no rows: {path}")
    return pairs


def parse_boolish(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return math.nan
    if text in {"1", "true", "yes", "y", "success", "succeeded", "pass"}:
        return 1.0
    if text in {"0", "false", "no", "n", "fail", "failed"}:
        return 0.0
    return float(value)


def read_metadata_csv(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not ({"file", "stem", "path"} & fields):
            raise ValueError("metadata_csv must contain at least one of columns: file, stem, path")
        for row in reader:
            keys = []
            for col in ("file", "path"):
                if row.get(col):
                    p = Path(row[col])
                    keys.extend([str(p), p.name, p.stem])
            if row.get("stem"):
                keys.append(row["stem"])
            meta: Dict[str, float] = {}
            if "success" in row:
                meta["success_attr"] = parse_boolish(row.get("success"))
            if "stopped_early" in row:
                meta["stopped_early_attr"] = parse_boolish(row.get("stopped_early"))
            if "task_success" in row:
                meta["success_attr"] = parse_boolish(row.get("task_success"))
            if "early_stop" in row:
                meta["stopped_early_attr"] = parse_boolish(row.get("early_stop"))
            if meta:
                for key in keys:
                    out[key] = meta
    return out


def apply_metadata(rows: List["Rollout"], metadata: Dict[str, Dict[str, float]]) -> None:
    if not metadata:
        return
    for row in rows:
        p = Path(row.path)
        for key in (row.path, p.name, p.stem):
            if key in metadata:
                row.metrics.update(metadata[key])
                break


def outcome_metadata_coverage(rows: List["Rollout"]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "success_present": 0,
            "stopped_early_present": 0,
            "success_rate": 0.0,
            "stopped_early_rate": 0.0,
            "complete": False,
            "missing_examples": [],
        }
    missing = []
    success_present = 0
    stopped_present = 0
    for row in rows:
        success_ok = np.isfinite(row.metrics.get("success_attr", math.nan))
        stopped_ok = np.isfinite(row.metrics.get("stopped_early_attr", math.nan))
        success_present += int(success_ok)
        stopped_present += int(stopped_ok)
        if not (success_ok and stopped_ok):
            missing.append(row.path)
    return {
        "n": int(n),
        "success_present": int(success_present),
        "stopped_early_present": int(stopped_present),
        "success_rate": float(success_present / n),
        "stopped_early_rate": float(stopped_present / n),
        "complete": bool(success_present == n and stopped_present == n),
        "missing_examples": missing[:10],
    }


def combined_outcome_metadata_coverage(groups: Dict[str, List["Rollout"]]) -> Dict[str, Any]:
    by_group = {name: outcome_metadata_coverage(rows) for name, rows in groups.items()}
    return {
        "by_group": by_group,
        "complete": bool(all(row["complete"] for row in by_group.values())),
        "missing_examples": [p for row in by_group.values() for p in row["missing_examples"]][:10],
    }


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
        ]
        if ref.get("board_target_force_explicit", False):
            rough_terms.append(0.50 * max(0.0, (m["force_p95"] - (target + 2.5 * sigma)) / sigma))
        else:
            rough_terms.append(
                0.50 * max(0.0, (m["force_p95"] - ref.get("force_p95_q90", m["force_p95"])) / ref.get("force_p95_mad", 1.0))
            )
        rough = float(sum(rough_terms))
        m["force_band_score"] = float(force_band)
        m["smoothness_score"] = float(math.exp(-0.55 * rough))
        m["quality_score"] = float(np.clip(force_band * m["smoothness_score"], 0.0, 1.0))
        if ref.get("board_target_force_explicit", False):
            m["too_light_flag"] = float(m["force_mean"] < target - 2.0 * sigma)
            m["too_heavy_flag"] = float(m["force_mean"] > target + 2.0 * sigma or m["force_p95"] > target + 2.5 * sigma)
        else:
            m["too_light_flag"] = float(m["force_mean"] < ref.get("force_mean_q20", -math.inf))
            m["too_heavy_flag"] = float(
                m["force_mean"] > ref.get("force_mean_q80", math.inf)
                or m["force_p95"] > ref.get("force_p95_q90", math.inf)
            )
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


def paired_deltas(
    baseline: List[Rollout],
    guided: List[Rollout],
    pair_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if pair_ids is not None:
        n = min(len(pair_ids), len(baseline), len(guided))
        pair_keys = list(pair_ids[:n])
        pairs = [(pair_keys[i], baseline[i], guided[i]) for i in range(n)]
    else:
        b_map = {r.stem: r for r in baseline}
        g_map = {r.stem: r for r in guided}
        common = sorted(set(b_map) & set(g_map))
        pairs = [(stem, b_map[stem], g_map[stem]) for stem in common]
    if not pairs:
        return {"n_pairs": 0, "note": "No matching stems; aggregate comparison only."}
    keys = sorted(set().union(*(b.metrics for _, b, _ in pairs), *(g.metrics for _, _, g in pairs)))
    out: Dict[str, Any] = {
        "n_pairs": len(pairs),
        "pairing_source": "csv" if pair_ids is not None else "matching_stem",
    }
    for key in keys:
        vals = []
        for _, b, g in pairs:
            bv = b.metrics.get(key, math.nan)
            gv = g.metrics.get(key, math.nan)
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


def _values(rows: List[Rollout], key: str) -> np.ndarray:
    vals = np.array([r.metrics.get(key, math.nan) for r in rows], dtype=np.float64)
    return vals[np.isfinite(vals)]


def bootstrap_mean_delta_ci(
    baseline: List[Rollout],
    guided: List[Rollout],
    key: str,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    b = _values(baseline, key)
    g = _values(guided, key)
    if len(b) == 0 or len(g) == 0:
        return {"available": False}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        bs = rng.choice(b, size=len(b), replace=True)
        gs = rng.choice(g, size=len(g), replace=True)
        deltas[i] = gs.mean() - bs.mean()
    observed = float(g.mean() - b.mean())
    return {
        "available": True,
        "key": key,
        "observed_delta": observed,
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "p_delta_positive": float(np.mean(deltas > 0.0)),
        "n_boot": int(n_boot),
    }


def paired_delta_values(baseline: List[Rollout], guided: List[Rollout], key: str) -> np.ndarray:
    vals = []
    for b, g in zip(baseline, guided):
        bv = b.metrics.get(key, math.nan)
        gv = g.metrics.get(key, math.nan)
        if np.isfinite(bv) and np.isfinite(gv):
            vals.append(gv - bv)
    return np.array(vals, dtype=np.float64)


def bootstrap_vector_ci(values: np.ndarray, *, n_boot: int, seed: int) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"available": False}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means[i] = sample.mean()
    return {
        "available": True,
        "observed_delta": float(values.mean()),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "p_delta_positive": float(np.mean(means > 0.0)),
        "n": int(len(values)),
        "n_boot": int(n_boot),
    }


def decision(
    task: str,
    summary: Dict[str, Any],
    baseline_rows: List[Rollout],
    guided_rows: List[Rollout],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    guided = summary["guided"]
    baseline = summary["baseline"]
    paired = summary["paired"]
    if baseline.get("n", 0) < args.min_episodes or guided.get("n", 0) < args.min_episodes:
        return {
            "production_validation_pass": False,
            "reason": (
                f"Insufficient rollout count: baseline_n={baseline.get('n', 0)}, "
                f"guided_n={guided.get('n', 0)}, min_episodes={args.min_episodes}."
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
    quality_ci = bootstrap_mean_delta_ci(
        baseline_rows,
        guided_rows,
        "quality_score",
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    )
    quality_delta = guided_quality - baseline_quality
    aggregate_quality_improved = (
        quality_delta >= args.min_quality_delta
        and quality_ci.get("available", False)
        and quality_ci.get("ci95_low", -math.inf) > 0.0
    )
    paired_ok = paired.get("n_pairs", 0) == 0 or q_delta.get("mean", 0.0) > 0.0
    paired_ci = summary.get("paired_quality_delta_ci", {"available": False})
    if paired.get("n_pairs", 0) > 0:
        paired_ok = (
            q_delta.get("mean", 0.0) >= args.min_quality_delta
            and paired_ci.get("available", False)
            and paired_ci.get("ci95_low", -math.inf) > 0.0
        )
    quality_improved = aggregate_quality_improved
    if paired.get("n_pairs", 0) > 0 and paired_ok and not args.require_aggregate_ci_for_paired:
        quality_improved = True
    paired_note = None
    if paired.get("n_pairs", 0) == 0:
        paired_note = "No paired stems; decision uses aggregate group comparison only."
    outcome_coverage = combined_outcome_metadata_coverage(
        {
            "baseline": baseline_rows,
            "guided": guided_rows,
        }
    )
    outcome_metadata_ok = outcome_coverage["complete"] or not args.require_outcome_metadata
    success_ok = True
    stopped_ok = True
    if "success_attr" in baseline and "success_attr" in guided:
        success_ok = guided["success_attr"]["mean"] + 1e-9 >= baseline["success_attr"]["mean"] - args.max_success_rate_drop
    if "stopped_early_attr" in baseline and "stopped_early_attr" in guided:
        stopped_ok = guided["stopped_early_attr"]["mean"] <= baseline["stopped_early_attr"]["mean"] + args.max_bad_rate_increase
    if task == "board":
        bad_rate_ok = guided.get("too_heavy_flag", {}).get("mean", 1.0) <= baseline.get("too_heavy_flag", {}).get("mean", 1.0) + args.max_bad_rate_increase
        rough_ok = guided.get("rough_flag", {}).get("mean", 1.0) <= baseline.get("rough_flag", {}).get("mean", 1.0) + args.max_bad_rate_increase
        passed = (
            quality_improved
            and paired_ok
            and bad_rate_ok
            and rough_ok
            and success_ok
            and stopped_ok
            and outcome_metadata_ok
        )
        reason = "board quality improves with positive bootstrap CI and no excessive too-heavy/rough-rate increase"
    else:
        risk_ok = guided.get("risk_score", {}).get("mean", 1.0) <= baseline.get("risk_score", {}).get("mean", 1.0)
        flag_ok = guided.get("risk_flag", {}).get("mean", 1.0) <= baseline.get("risk_flag", {}).get("mean", 1.0) + args.max_bad_rate_increase
        passed = (
            quality_improved
            and paired_ok
            and risk_ok
            and flag_ok
            and success_ok
            and stopped_ok
            and outcome_metadata_ok
        )
        reason = "insertion quality improves with positive bootstrap CI while risk proxy does not increase"
    return {
        "production_validation_pass": bool(passed),
        "reason": reason,
        "quality_improved": bool(quality_improved),
        "aggregate_quality_improved": bool(aggregate_quality_improved),
        "quality_delta_mean": float(quality_delta),
        "min_quality_delta": float(args.min_quality_delta),
        "quality_delta_ci": quality_ci,
        "paired_quality_delta_mean": q_delta.get("mean"),
        "paired_quality_guided_better_rate": q_delta.get("guided_better_rate"),
        "paired_quality_delta_ci": paired_ci,
        "paired_note": paired_note,
        "max_bad_rate_increase": float(args.max_bad_rate_increase),
        "require_aggregate_ci_for_paired": bool(args.require_aggregate_ci_for_paired),
        "success_rate_ok": bool(success_ok),
        "stopped_early_rate_ok": bool(stopped_ok),
        "require_outcome_metadata": bool(args.require_outcome_metadata),
        "outcome_metadata_ok": bool(outcome_metadata_ok),
        "outcome_metadata_coverage": outcome_coverage,
        "max_success_rate_drop": float(args.max_success_rate_drop),
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
        f"- debug_or_underpowered: `{result['debug_or_underpowered']}`",
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
    parser.add_argument("--pairing_csv", default=None)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--min_quality_delta", type=float, default=0.03)
    parser.add_argument("--max_bad_rate_increase", type=float, default=0.05)
    parser.add_argument("--max_success_rate_drop", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--board_target_force", type=float, default=None)
    parser.add_argument("--board_force_sigma", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_aggregate_ci_for_paired", action="store_true")
    parser.add_argument(
        "--require_outcome_metadata",
        action="store_true",
        help="Require success and stopped_early metadata for every rollout before passing the production gate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_dir)
    guided_root = Path(args.guided_dir)
    pair_ids = None
    if args.pairing_csv:
        pairs = read_pairing_csv(Path(args.pairing_csv), baseline_root, guided_root)
        baseline_paths = [p["baseline"] for p in pairs]
        guided_paths = [p["guided"] for p in pairs]
        pair_ids = [str(p["pair_id"]) for p in pairs]
    else:
        baseline_paths = discover_hdf5(baseline_root)
        guided_paths = discover_hdf5(guided_root)
    if not baseline_paths:
        raise FileNotFoundError(f"No HDF5 files under baseline_dir={args.baseline_dir}")
    if not guided_paths:
        raise FileNotFoundError(f"No HDF5 files under guided_dir={args.guided_dir}")

    baseline = [episode_metrics(p, "baseline", args.task) for p in baseline_paths]
    guided = [episode_metrics(p, "guided", args.task) for p in guided_paths]
    metadata = read_metadata_csv(args.metadata_csv)
    apply_metadata(baseline, metadata)
    apply_metadata(guided, metadata)
    ref = fit_reference(baseline, args.task)
    if args.task == "board":
        if args.board_target_force is not None:
            ref["board_target_force"] = float(args.board_target_force)
            ref["board_target_force_explicit"] = True
        if args.board_force_sigma is not None:
            ref["board_force_sigma"] = float(args.board_force_sigma)
    for row in baseline + guided:
        score_rollout(row, ref)

    summary = {
        "baseline": aggregate(baseline),
        "guided": aggregate(guided),
        "paired": paired_deltas(baseline, guided, pair_ids=pair_ids),
        "reference": ref,
    }
    if pair_ids is not None:
        summary["paired_quality_delta_ci"] = bootstrap_vector_ci(
            paired_delta_values(baseline, guided, "quality_score"),
            n_boot=args.bootstrap_samples,
            seed=args.seed + 17,
        )
    result = {
        "task": args.task,
        "baseline_dir": str(Path(args.baseline_dir)),
        "guided_dir": str(Path(args.guided_dir)),
        "n_baseline_files": len(baseline_paths),
        "n_guided_files": len(guided_paths),
        "pairing_csv": str(Path(args.pairing_csv)) if args.pairing_csv else None,
        "metadata_csv": str(Path(args.metadata_csv)) if args.metadata_csv else None,
        "debug_or_underpowered": bool(len(baseline_paths) < 10 or len(guided_paths) < 10),
        "summary": summary,
        "decision": decision(args.task, summary, baseline, guided, args),
        "decision_config": {
            "min_episodes": args.min_episodes,
            "min_quality_delta": args.min_quality_delta,
            "max_bad_rate_increase": args.max_bad_rate_increase,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
        },
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
