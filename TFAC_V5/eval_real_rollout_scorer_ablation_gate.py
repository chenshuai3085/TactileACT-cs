"""Evaluate three-arm real rollout ablation for TacQuality scorers.

This is the formal result evaluator for the scorer-selection question:

  baseline DP
  vs task-default TacQuality-guided DP
  vs DistilledTacQualityEnergy-guided DP

It reuses the same task-specific quality metrics from
eval_real_rollout_quality_gate.py, but compares two guided policies against
the same baseline and against each other.  It consumes recorded HDF5 rollouts;
it does not run a robot and does not claim validation without data.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from TFAC_V5.eval_real_rollout_quality_gate import (  # noqa: E402
    Rollout,
    aggregate,
    apply_metadata,
    bootstrap_mean_delta_ci,
    bootstrap_vector_ci,
    discover_hdf5,
    episode_metrics,
    fit_reference,
    paired_delta_values,
    read_metadata_csv,
    resolve_rollout_path,
    score_rollout,
    write_csv,
)


DEFAULT_OUT = Path("/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate")
ARMS = ["baseline", "default_guided", "distilled_guided"]


def read_three_arm_pairing_csv(
    path: Path,
    baseline_root: Path,
    default_root: Path,
    distilled_root: Path,
) -> Dict[str, Any]:
    rows: List[Dict[str, Path]] = []
    pair_ids: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"baseline", "default_guided", "distilled_guided"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"pairing_csv missing columns: {sorted(missing)}")
        for i, row in enumerate(reader):
            pair_id = row.get("pair_id") or f"pair_{i:04d}"
            pair_ids.append(pair_id)
            rows.append(
                {
                    "baseline": resolve_rollout_path(row["baseline"], baseline_root),
                    "default_guided": resolve_rollout_path(row["default_guided"], default_root),
                    "distilled_guided": resolve_rollout_path(row["distilled_guided"], distilled_root),
                }
            )
    if not rows:
        raise ValueError(f"pairing_csv has no rows: {path}")
    return {
        "pair_ids": pair_ids,
        "baseline": [r["baseline"] for r in rows],
        "default_guided": [r["default_guided"] for r in rows],
        "distilled_guided": [r["distilled_guided"] for r in rows],
    }


def build_rows(args: argparse.Namespace) -> Dict[str, List[Rollout]]:
    roots = {
        "baseline": Path(args.baseline_dir),
        "default_guided": Path(args.default_guided_dir),
        "distilled_guided": Path(args.distilled_guided_dir),
    }
    if args.pairing_csv:
        paired = read_three_arm_pairing_csv(
            Path(args.pairing_csv),
            roots["baseline"],
            roots["default_guided"],
            roots["distilled_guided"],
        )
        paths = {arm: paired[arm] for arm in ARMS}
    else:
        paired = {"pair_ids": None}
        paths = {arm: discover_hdf5(root) for arm, root in roots.items()}
    for arm, arm_paths in paths.items():
        if not arm_paths:
            raise FileNotFoundError(f"No HDF5 files for {arm}: {roots[arm]}")
    rows = {
        arm: [episode_metrics(path, arm, args.task) for path in paths[arm]]
        for arm in ARMS
    }
    metadata = read_metadata_csv(args.metadata_csv)
    for arm_rows in rows.values():
        apply_metadata(arm_rows, metadata)
    ref = fit_reference(rows["baseline"], args.task)
    for arm_rows in rows.values():
        for row in arm_rows:
            score_rollout(row, ref)
    return {"rows": rows, "reference": ref, "pair_ids": paired.get("pair_ids")}


def paired_delta_summary(
    left: List[Rollout],
    right: List[Rollout],
    key: str,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    vals = paired_delta_values(left, right, key)
    ci = bootstrap_vector_ci(vals, n_boot=n_boot, seed=seed)
    if not ci.get("available", False):
        return {"available": False}
    return {
        **ci,
        "guided_better_rate": float(np.mean(vals > 0.0)),
        "median": float(np.median(vals)),
        "p05": float(np.percentile(vals, 5)),
        "p95": float(np.percentile(vals, 95)),
    }


def quality_delta_ci(
    baseline: List[Rollout],
    guided: List[Rollout],
    args: argparse.Namespace,
    seed_offset: int,
) -> Dict[str, Any]:
    if args.pairing_csv:
        return paired_delta_summary(
            baseline,
            guided,
            "quality_score",
            n_boot=args.bootstrap_samples,
            seed=args.seed + seed_offset,
        )
    return bootstrap_mean_delta_ci(
        baseline,
        guided,
        "quality_score",
        n_boot=args.bootstrap_samples,
        seed=args.seed + seed_offset,
    )


def arm_mean(rows: List[Rollout], key: str, default: float = math.nan) -> float:
    vals = np.array([r.metrics.get(key, math.nan) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else default


def non_degradation_checks(task: str, baseline: List[Rollout], guided: List[Rollout], args: argparse.Namespace) -> Dict[str, Any]:
    success_ok = True
    stopped_ok = True
    b_success = arm_mean(baseline, "success_attr")
    g_success = arm_mean(guided, "success_attr")
    if np.isfinite(b_success) and np.isfinite(g_success):
        success_ok = g_success + 1e-9 >= b_success - args.max_success_rate_drop
    b_stop = arm_mean(baseline, "stopped_early_attr")
    g_stop = arm_mean(guided, "stopped_early_attr")
    if np.isfinite(b_stop) and np.isfinite(g_stop):
        stopped_ok = g_stop <= b_stop + args.max_bad_rate_increase
    checks: Dict[str, Any] = {
        "success_rate_ok": bool(success_ok),
        "stopped_early_rate_ok": bool(stopped_ok),
        "baseline_success_rate": b_success,
        "guided_success_rate": g_success,
        "baseline_stopped_rate": b_stop,
        "guided_stopped_rate": g_stop,
    }
    if task == "board":
        b_heavy = arm_mean(baseline, "too_heavy_flag", 1.0)
        g_heavy = arm_mean(guided, "too_heavy_flag", 1.0)
        b_rough = arm_mean(baseline, "rough_flag", 1.0)
        g_rough = arm_mean(guided, "rough_flag", 1.0)
        checks.update(
            {
                "bad_rate_ok": bool(g_heavy <= b_heavy + args.max_bad_rate_increase),
                "rough_rate_ok": bool(g_rough <= b_rough + args.max_bad_rate_increase),
                "baseline_too_heavy_rate": b_heavy,
                "guided_too_heavy_rate": g_heavy,
                "baseline_rough_rate": b_rough,
                "guided_rough_rate": g_rough,
            }
        )
    else:
        b_risk = arm_mean(baseline, "risk_score", 1.0)
        g_risk = arm_mean(guided, "risk_score", 1.0)
        b_flag = arm_mean(baseline, "risk_flag", 1.0)
        g_flag = arm_mean(guided, "risk_flag", 1.0)
        checks.update(
            {
                "risk_score_ok": bool(g_risk <= b_risk),
                "risk_flag_ok": bool(g_flag <= b_flag + args.max_bad_rate_increase),
                "baseline_risk_score": b_risk,
                "guided_risk_score": g_risk,
                "baseline_risk_flag_rate": b_flag,
                "guided_risk_flag_rate": g_flag,
            }
        )
    checks["all_pass"] = all(
        bool(v)
        for k, v in checks.items()
        if k.endswith("_ok") or k in {"bad_rate_ok", "rough_rate_ok", "risk_score_ok", "risk_flag_ok"}
    )
    return checks


def guided_decision(task: str, baseline: List[Rollout], guided: List[Rollout], args: argparse.Namespace, seed_offset: int) -> Dict[str, Any]:
    if len(baseline) < args.min_episodes or len(guided) < args.min_episodes:
        return {
            "pass_vs_baseline": False,
            "reason": "Insufficient rollout count.",
            "n_baseline": len(baseline),
            "n_guided": len(guided),
        }
    ci = quality_delta_ci(baseline, guided, args, seed_offset)
    q_delta = arm_mean(guided, "quality_score") - arm_mean(baseline, "quality_score")
    nondeg = non_degradation_checks(task, baseline, guided, args)
    quality_pass = (
        ci.get("available", False)
        and q_delta >= args.min_quality_delta
        and ci.get("ci95_low", -math.inf) > 0.0
    )
    return {
        "pass_vs_baseline": bool(quality_pass and nondeg["all_pass"]),
        "quality_pass": bool(quality_pass),
        "quality_delta_mean": float(q_delta),
        "quality_delta_ci": ci,
        "non_degradation": nondeg,
        "reason": "quality improves with positive CI and task bad-rate constraints do not regress",
    }


def compare_guided_arms(rows: Dict[str, List[Rollout]], args: argparse.Namespace) -> Dict[str, Any]:
    default = rows["default_guided"]
    distilled = rows["distilled_guided"]
    ci = quality_delta_ci(default, distilled, args, 303)
    distilled_minus_default = arm_mean(distilled, "quality_score") - arm_mean(default, "quality_score")
    if ci.get("available", False) and ci.get("ci95_low", -math.inf) > 0.0:
        winner = "distilled_guided"
        reason = "distilled quality is significantly higher than task-default guided"
    elif ci.get("available", False) and ci.get("ci95_high", math.inf) < 0.0:
        winner = "default_guided"
        reason = "task-default guided quality is significantly higher than distilled"
    else:
        winner = "tie_or_underpowered"
        reason = "guided-vs-guided CI overlaps zero"
    return {
        "winner": winner,
        "reason": reason,
        "distilled_minus_default_quality_delta_mean": float(distilled_minus_default),
        "distilled_minus_default_quality_delta_ci": ci,
    }


def build_result(args: argparse.Namespace) -> Dict[str, Any]:
    built = build_rows(args)
    rows = built["rows"]
    summary = {arm: aggregate(rows[arm]) for arm in ARMS}
    decisions = {
        "default_guided": guided_decision(args.task, rows["baseline"], rows["default_guided"], args, 101),
        "distilled_guided": guided_decision(args.task, rows["baseline"], rows["distilled_guided"], args, 202),
    }
    guided_comparison = compare_guided_arms(rows, args)
    production_ablation_pass = bool(
        decisions["default_guided"]["pass_vs_baseline"]
        or decisions["distilled_guided"]["pass_vs_baseline"]
    )
    result = {
        "task": args.task,
        "baseline_dir": args.baseline_dir,
        "default_guided_dir": args.default_guided_dir,
        "distilled_guided_dir": args.distilled_guided_dir,
        "pairing_csv": args.pairing_csv,
        "metadata_csv": args.metadata_csv,
        "debug_or_underpowered": bool(any(len(rows[arm]) < args.min_episodes for arm in ARMS)),
        "reference": built["reference"],
        "summary": summary,
        "decisions": decisions,
        "guided_arm_comparison": guided_comparison,
        "production_ablation_pass": production_ablation_pass,
        "recommended_real_scorer": (
            guided_comparison["winner"]
            if guided_comparison["winner"] != "tie_or_underpowered"
            else (
                "distilled_guided"
                if decisions["distilled_guided"]["pass_vs_baseline"]
                else "default_guided"
                if decisions["default_guided"]["pass_vs_baseline"]
                else None
            )
        ),
        "decision_config": {
            "min_episodes": args.min_episodes,
            "min_quality_delta": args.min_quality_delta,
            "max_bad_rate_increase": args.max_bad_rate_increase,
            "max_success_rate_drop": args.max_success_rate_drop,
            "bootstrap_samples": args.bootstrap_samples,
            "paired": bool(args.pairing_csv),
            "seed": args.seed,
        },
        "note": "Formal three-arm scorer ablation; use after collecting recorded rollouts.",
    }
    result["all_rows"] = [row for arm in ARMS for row in rows[arm]]
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Real Rollout Scorer Ablation Gate",
        "",
        f"- task: `{result['task']}`",
        f"- production_ablation_pass: `{result['production_ablation_pass']}`",
        f"- recommended_real_scorer: `{result['recommended_real_scorer']}`",
        f"- debug_or_underpowered: `{result['debug_or_underpowered']}`",
        "",
        "## Decisions",
        "",
        "```json",
        json.dumps(result["decisions"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Guided Arm Comparison",
        "",
        "```json",
        json.dumps(result["guided_arm_comparison"], ensure_ascii=False, indent=2),
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
    parser.add_argument("--default_guided_dir", required=True)
    parser.add_argument("--distilled_guided_dir", required=True)
    parser.add_argument("--pairing_csv", default=None)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tag", default=None)
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--min_quality_delta", type=float, default=0.03)
    parser.add_argument("--max_bad_rate_increase", type=float, default=0.05)
    parser.add_argument("--max_success_rate_drop", type=float, default=0.0)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_result(args)
    tag = args.tag or f"{args.task}_baseline_vs_default_vs_distilled"
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "real_rollout_scorer_ablation_gate.json"
    md_path = out_dir / "real_rollout_scorer_ablation_gate.md"
    csv_path = out_dir / "real_rollout_scorer_ablation_episode_metrics.csv"
    rows = result.pop("all_rows")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    write_csv(rows, csv_path)
    print(
        json.dumps(
            {
                "production_ablation_pass": result["production_ablation_pass"],
                "recommended_real_scorer": result["recommended_real_scorer"],
                "guided_arm_winner": result["guided_arm_comparison"]["winner"],
                "json": str(json_path),
                "markdown": str(md_path),
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
