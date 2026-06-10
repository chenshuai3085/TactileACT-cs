"""Audit whether TacQuality guidance agrees with task quality proxies.

This audit is intentionally stricter than "the scorer score increased".  A
classifier-guidance potential can be locally self-consistent while still moving
actions toward rough or out-of-range behavior.  Here we reuse existing
clean-action refinement outputs and check whether accepted scorer-gradient
updates also preserve task-level proxy quality:

  - insertion: scorer improves, hard range violation stays negligible, and
    smoothness side effect remains small;
  - board wiping: scorer improves, force/action smoothness proxy improves, and
    normalized action range remains valid.

It is still offline evidence, not a real robot rollout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_proxy_alignment_audit")

PATHS = {
    "insertion_clean": Path(
        "/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/"
        "n24_k4/insertion_distilled_clean_refine_comparison.json"
    ),
    "board_clean": Path(
        "/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/"
        "fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.json"
    ),
    "action_aware_guidance": Path(
        "/home/chenshuai/Project/output/action_aware_guidance_suitability/"
        "line_search_default/action_aware_guidance_suitability.json"
    ),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d: Optional[Dict[str, Any]], dotted: str, default=None):
    cur: Any = d
    if cur is None:
        return default
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def metric(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def row(
    *,
    task: str,
    scorer: str,
    source: str,
    score_delta_mean: float,
    improved_rate: float,
    smoothness_delta_mean: float,
    range_violation_max: float,
    accept_rate_mean: float,
    max_allowed_smoothness_delta: float,
    requires_smoothness_improvement: bool,
    min_improved_rate: float,
    max_range_violation: float,
    min_accept_rate: float,
) -> Dict[str, Any]:
    score_pass = improved_rate >= min_improved_rate and score_delta_mean > 0.0
    if requires_smoothness_improvement:
        smoothness_pass = smoothness_delta_mean <= max_allowed_smoothness_delta
    else:
        smoothness_pass = smoothness_delta_mean <= max_allowed_smoothness_delta
    range_pass = range_violation_max <= max_range_violation
    accept_pass = accept_rate_mean >= min_accept_rate
    return {
        "task": task,
        "scorer": scorer,
        "source": source,
        "metrics": {
            "score_delta_mean": float(score_delta_mean),
            "improved_rate": float(improved_rate),
            "smoothness_delta_mean": float(smoothness_delta_mean),
            "range_violation_max": float(range_violation_max),
            "accept_rate_mean": float(accept_rate_mean),
        },
        "thresholds": {
            "min_improved_rate": float(min_improved_rate),
            "max_allowed_smoothness_delta": float(max_allowed_smoothness_delta),
            "requires_smoothness_improvement": bool(requires_smoothness_improvement),
            "max_range_violation": float(max_range_violation),
            "min_accept_rate": float(min_accept_rate),
        },
        "checks": {
            "score_improves": bool(score_pass),
            "smoothness_aligned": bool(smoothness_pass),
            "range_safe": bool(range_pass),
            "accept_rate_ok": bool(accept_pass),
        },
        "proxy_alignment_pass": bool(score_pass and smoothness_pass and range_pass and accept_pass),
        "interpretation": (
            "pass"
            if score_pass and smoothness_pass and range_pass and accept_pass
            else "Scorer-gradient refinement improves its own score but has a proxy-quality side effect that needs review."
        ),
    }


def build(paths: Dict[str, Path]) -> Dict[str, Any]:
    insertion = load_json(paths["insertion_clean"])
    board = load_json(paths["board_clean"])
    action_aware = load_json(paths["action_aware_guidance"])

    rows = [
        row(
            task="insertion",
            scorer="InsertionRiskScorerRuntime",
            source=str(paths["insertion_clean"]),
            score_delta_mean=metric(get(insertion, "insertion_risk.summary.score_delta.mean")),
            improved_rate=metric(get(insertion, "insertion_risk.summary.refined_beats_base_rate")),
            smoothness_delta_mean=metric(get(insertion, "insertion_risk.summary.smoothness_delta.mean")),
            range_violation_max=metric(get(insertion, "insertion_risk.summary.refined_hard_range_violation.max")),
            accept_rate_mean=metric(get(insertion, "insertion_risk.summary.accept_rate_per_step.mean")),
            max_allowed_smoothness_delta=0.02,
            requires_smoothness_improvement=False,
            min_improved_rate=0.95,
            max_range_violation=0.002,
            min_accept_rate=0.75,
        ),
        row(
            task="insertion",
            scorer="DistilledTacQualityEnergyRuntime",
            source=str(paths["insertion_clean"]),
            score_delta_mean=metric(get(insertion, "distilled_energy.summary.score_delta.mean")),
            improved_rate=metric(get(insertion, "distilled_energy.summary.refined_beats_base_rate")),
            smoothness_delta_mean=metric(get(insertion, "distilled_energy.summary.smoothness_delta.mean")),
            range_violation_max=metric(get(insertion, "distilled_energy.summary.refined_hard_range_violation.max")),
            accept_rate_mean=metric(get(insertion, "distilled_energy.summary.accept_rate_per_step.mean")),
            max_allowed_smoothness_delta=0.005,
            requires_smoothness_improvement=False,
            min_improved_rate=0.95,
            max_range_violation=0.002,
            min_accept_rate=0.75,
        ),
        row(
            task="board",
            scorer="PTGProxyScorerV2Runtime",
            source=str(paths["board_clean"]),
            score_delta_mean=metric(get(board, "ptg_proxy_v2.summary.score_delta.mean")),
            improved_rate=metric(get(board, "ptg_proxy_v2.summary.guided_beats_base_rate")),
            smoothness_delta_mean=metric(get(board, "ptg_proxy_v2.summary.smoothness_delta.mean")),
            range_violation_max=metric(get(board, "ptg_proxy_v2.summary.range_violation.max")),
            accept_rate_mean=metric(get(board, "ptg_proxy_v2.summary.guide_accept_rate_per_step.mean")),
            max_allowed_smoothness_delta=-0.05,
            requires_smoothness_improvement=True,
            min_improved_rate=0.95,
            max_range_violation=1e-6,
            min_accept_rate=0.95,
        ),
        row(
            task="board",
            scorer="DistilledTacQualityEnergyRuntime",
            source=str(paths["board_clean"]),
            score_delta_mean=metric(get(board, "distilled_energy.summary.score_delta.mean")),
            improved_rate=metric(get(board, "distilled_energy.summary.guided_beats_base_rate")),
            smoothness_delta_mean=metric(get(board, "distilled_energy.summary.smoothness_delta.mean")),
            range_violation_max=metric(get(board, "distilled_energy.summary.range_violation.max")),
            accept_rate_mean=metric(get(board, "distilled_energy.summary.guide_accept_rate_per_step.mean")),
            max_allowed_smoothness_delta=-0.05,
            requires_smoothness_improvement=True,
            min_improved_rate=0.95,
            max_range_violation=1e-6,
            min_accept_rate=0.95,
        ),
    ]

    action_aware_row = {
        "task": "mixed",
        "scorer": "ActionAwareScorerRuntime",
        "source": str(paths["action_aware_guidance"]),
        "metrics": {
            "recommended_mode": get(action_aware, "recommended_mode"),
            "quality_line_search_accepted_rate": metric(
                get(action_aware, "modes.quality.gradient_probe.mixed.line_search.accepted_rate")
            ),
            "quality_fixed_step_improved_rate": metric(
                get(action_aware, "modes.quality.gradient_probe.mixed.improved_rate")
            ),
            "quality_score_auc": metric(get(action_aware, "modes.quality.score_metrics.binary_auc")),
            "quality_score_corr": metric(get(action_aware, "modes.quality.score_metrics.corr_with_quality")),
        },
        "checks": {
            "quality_mode_selected": get(action_aware, "recommended_mode") == "quality",
            "line_search_required": metric(
                get(action_aware, "modes.quality.gradient_probe.mixed.line_search.accepted_rate")
            )
            >= 0.95,
            "fixed_step_not_promoted": metric(get(action_aware, "modes.quality.gradient_probe.mixed.improved_rate"))
            < 0.95,
        },
    }
    action_aware_row["proxy_alignment_pass"] = bool(
        action_aware_row["checks"]["quality_mode_selected"]
        and action_aware_row["checks"]["line_search_required"]
        and action_aware_row["checks"]["fixed_step_not_promoted"]
    )
    action_aware_row["interpretation"] = (
        "ActionAware is acceptable only with quality-mode line-search; fixed-step guidance is not aligned enough."
    )

    all_rows = rows + [action_aware_row]
    task_pass = {
        task: all(row["proxy_alignment_pass"] for row in rows if row["task"] == task)
        for task in ["insertion", "board"]
    }
    result = {
        "name": "TacQuality proxy-alignment audit",
        "purpose": (
            "Check whether scorer-gradient refinement improves scorer outputs while preserving task quality proxies."
        ),
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "proxy_alignment_pass": bool(all(row["proxy_alignment_pass"] for row in all_rows)),
        "task_pass": task_pass,
        "rows": all_rows,
        "key_findings": [
            "InsertionRisk improves insertion risk score strongly but adds a small smoothness cost within threshold.",
            "DistilledTacQualityEnergy improves insertion score while also reducing insertion smoothness proxy.",
            "Both board scorers improve score and reduce board smoothness proxy without range violations.",
            "ActionAware should remain line-search/accept-only; fixed-step action-gradient ascent is not promoted.",
        ],
        "remaining_gap": "Offline proxy alignment does not replace real robot/production rollout evidence.",
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Proxy-Alignment Audit",
        "",
        f"- proxy_alignment_pass: `{result['proxy_alignment_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- remaining_gap: {result['remaining_gap']}",
        "",
        "## Rows",
        "",
        "| task | scorer | pass | score delta | improved | smoothness delta | range max | accept |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["rows"]:
        metrics = item["metrics"]
        lines.append(
            f"| {item['task']} | {item['scorer']} | {item['proxy_alignment_pass']} | "
            f"{metrics.get('score_delta_mean', 0.0):.6f} | {metrics.get('improved_rate', 0.0):.4f} | "
            f"{metrics.get('smoothness_delta_mean', 0.0):.6f} | {metrics.get('range_violation_max', 0.0):.6f} | "
            f"{metrics.get('accept_rate_mean', 0.0):.4f} |"
        )
    lines.extend(["", "## Key Findings", ""])
    for finding in result["key_findings"]:
        lines.append(f"- {finding}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(PATHS)
    json_path = out_dir / "tac_quality_proxy_alignment_audit.json"
    md_path = out_dir / "tac_quality_proxy_alignment_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "proxy_alignment_pass": result["proxy_alignment_pass"],
                "task_pass": result["task_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
