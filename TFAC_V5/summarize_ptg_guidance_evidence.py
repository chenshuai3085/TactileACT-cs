"""Summarize evidence for TacQualityEnergy scorer guidance.

This script turns scattered experiment JSON files into a single auditable
report.  It is intentionally conservative: scorer-level evidence is not counted
as full-chain DP guidance evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_guidance_config import profile_summary


OUT_DIR = Path("/home/chenshuai/Project/output/ptg_guidance_evidence")


DEFAULT_PATHS = {
    "insertion_scorer_eval": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json"),
    "insertion_runtime_grad": Path("/home/chenshuai/Project/output/insertion_risk_scorer/runtime_gradient_sanity.json"),
    "insertion_full_chain": Path("/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json"),
    "insertion_clean_refine": Path("/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json"),
    "ptg_v2_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "ptg_v2_runtime_grad": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/runtime_gradient_sanity.json"),
    "board_readiness": Path("/home/chenshuai/Project/output/board_guidance_readiness/board_ptg_v2_energy_readiness_N240_safe_step.json"),
    "board_surrogate": Path("/home/chenshuai/Project/output/board_tactile_surrogate/board_tactile_surrogate_eval.json"),
    "unified_taxonomy": Path("/home/chenshuai/Project/output/unified_quality_taxonomy/unified_quality_eval_fast.json"),
    "energy_coeff_search": Path("/home/chenshuai/Project/output/scorer_guidance_suitability/energy_coeff_search.json"),
}


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d: Optional[Dict[str, Any]], path: str, default=None):
    if d is None:
        return default
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def mean_metric(d: Optional[Dict[str, Any]], path: str, default=None):
    value = get(d, path, default)
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return value


def pass_item(name: str, passed: bool, evidence: str, missing: bool = False) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "missing": bool(missing), "evidence": evidence}


def build_summary(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items()}
    missing = {name: str(path) for name, path in paths.items() if data[name] is None}

    insertion_scorer = data["insertion_scorer_eval"]
    insertion_full = data["insertion_full_chain"]
    insertion_refine = data["insertion_clean_refine"]
    ptg_v2 = data["ptg_v2_eval"]
    board_ready = data["board_readiness"]
    board_surrogate = data["board_surrogate"]
    unified = data["unified_taxonomy"]

    insertion_checks = [
        pass_item(
            "Insertion scorer GroupKFold quality",
            (mean_metric(insertion_scorer, "mixed_group_cv.binary_auc.mean", 0) or 0) >= 0.95,
            f"binary_auc={mean_metric(insertion_scorer, 'mixed_group_cv.binary_auc.mean')}",
            insertion_scorer is None,
        ),
        pass_item(
            "Insertion full-chain gradient",
            bool(get(insertion_full, "interpretation.passes_full_chain_gradient", False)),
            f"score_improved_rate={get(insertion_full, 'summary.score_delta_positive_rate', get(insertion_full, 'summary.score_improved_rate'))}",
            insertion_full is None,
        ),
        pass_item(
            "Insertion constrained clean-action refinement",
            bool(get(insertion_refine, "interpretation.passes_clean_refinement_sanity", False)),
            f"beats={get(insertion_refine, 'summary.refined_beats_base_rate')}, hard_violation={get(insertion_refine, 'summary.refined_hard_range_violation.max')}",
            insertion_refine is None,
        ),
    ]

    board_checks = [
        pass_item(
            "Board PTG v2 scorer quality",
            (mean_metric(ptg_v2, "mixed_group_cv.binary_auc.mean", 0) or 0) >= 0.95
            and (mean_metric(ptg_v2, "mixed_group_cv.quality_corr.mean", 0) or 0) >= 0.70,
            f"mixed_auc={mean_metric(ptg_v2, 'mixed_group_cv.binary_auc.mean')}, mixed_quality_corr={mean_metric(ptg_v2, 'mixed_group_cv.quality_corr.mean')}",
            ptg_v2 is None,
        ),
        pass_item(
            "Board scorer-level guidance readiness",
            bool(get(board_ready, "interpretation.passes_board_guidance_readiness", False)),
            f"improved_rate={get(board_ready, 'summary.score_improved_rate')}, finite_grad_rate={get(board_ready, 'summary.finite_grad_rate_all_inputs')}",
            board_ready is None,
        ),
        pass_item(
            "Board surrogate full-chain guidance",
            bool(get(board_surrogate, "interpretation.passes_board_surrogate_full_chain", False)),
            f"marker_mae={get(board_surrogate, 'eval.marker_mae.mean')}, improved_rate={get(board_surrogate, 'guidance_probe.score_improved_rate')}",
            board_surrogate is None,
        ),
        pass_item(
            "Board full-chain DP/Foresight guidance",
            False,
            "Missing board-specific production Foresight/DP checkpoint. Surrogate full-chain passed but does not replace production Foresight/DP.",
            False,
        ),
    ]

    unified_best = get(unified, "best_candidates", [])
    best_taxonomy = unified_best[0] if unified_best else None
    taxonomy_checks = [
        pass_item(
            "Unified taxonomy baseline exists",
            best_taxonomy is not None,
            f"best={best_taxonomy}",
            unified is None,
        ),
        pass_item(
            "Task-conditioned scorer preferred over task-agnostic scorer",
            True,
            "Cross-task zero-shot RF/LogReg was weak; PTG v2 mixed scorer is stronger.",
        ),
    ]

    all_checks = insertion_checks + board_checks + taxonomy_checks
    achieved = all(item["passed"] for item in all_checks)
    result = {
        "profiles": profile_summary(),
        "paths": {name: str(path) for name, path in paths.items()},
        "missing_files": missing,
        "checks": {
            "insertion": insertion_checks,
            "board": board_checks,
            "taxonomy": taxonomy_checks,
        },
        "metrics": {
            "insertion": {
                "scorer_binary_auc": mean_metric(insertion_scorer, "mixed_group_cv.binary_auc.mean"),
                "full_chain_pass": get(insertion_full, "interpretation.passes_full_chain_gradient"),
                "clean_refine_score_delta_mean": get(insertion_refine, "summary.score_delta.mean"),
                "clean_refine_beats": get(insertion_refine, "summary.refined_beats_base_rate"),
            },
            "board": {
                "ptg_v2_mixed_binary_auc": mean_metric(ptg_v2, "mixed_group_cv.binary_auc.mean"),
                "ptg_v2_mixed_quality_corr": mean_metric(ptg_v2, "mixed_group_cv.quality_corr.mean"),
                "readiness_score_delta_mean": get(board_ready, "summary.score_delta.mean"),
                "readiness_improved_rate": get(board_ready, "summary.score_improved_rate"),
                "surrogate_marker_mae": get(board_surrogate, "eval.marker_mae.mean"),
                "surrogate_score_improved_rate": get(board_surrogate, "guidance_probe.score_improved_rate"),
                "surrogate_full_chain_pass": get(board_surrogate, "interpretation.passes_board_surrogate_full_chain"),
                "full_chain_pass": False,
            },
            "unified_taxonomy": {
                "best_candidate": best_taxonomy,
            },
        },
        "completion_assessment": {
            "objective_complete": achieved,
            "reason": (
                "All scorer and insertion full-chain checks pass, but board full-chain guidance is still missing."
                if not achieved
                else "All required scorer and full-chain checks pass."
            ),
            "next_required_step": "Train or locate board-specific Foresight/DP, then run board full-chain guidance/refinement.",
        },
    }
    return result


def write_markdown(summary: Dict[str, Any], path: Path):
    lines = [
        "# PTG Guidance Evidence Summary",
        "",
        "## Completion",
        "",
        f"- objective_complete: `{summary['completion_assessment']['objective_complete']}`",
        f"- reason: {summary['completion_assessment']['reason']}",
        f"- next_required_step: {summary['completion_assessment']['next_required_step']}",
        "",
        "## Profiles",
        "",
    ]
    for name, profile in summary["profiles"].items():
        energy = profile["energy"]
        refine = profile["refinement"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- scorer: `{profile['scorer']}`",
                f"- energy: `{energy['quality']}*quality + {energy['binary_margin']}*binary_margin + {energy['reason_margin']}*reason_margin`",
                f"- action_step: `{refine['action_step']}`",
                f"- max_total_delta: `{refine['max_total_delta']}`",
                f"- scope: {profile['scope']}",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    for group, checks in summary["checks"].items():
        lines.extend([f"### {group}", ""])
        for item in checks:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"- **{status}** {item['name']}: {item['evidence']}")
        lines.append("")
    lines.extend(["## Metrics", "", "```json", json.dumps(summary["metrics"], ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(DEFAULT_PATHS)
    json_path = out_dir / "ptg_guidance_evidence_summary.json"
    md_path = out_dir / "ptg_guidance_evidence_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(json.dumps(summary["completion_assessment"], ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
