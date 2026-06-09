"""Offline production-readiness gate for PTG scorer guidance.

This script is intentionally conservative.  It does not claim real-robot
completion.  It checks whether the current socket and board evidence is strong
enough to be considered ready for a production/robot dry run:

  - scorer quality and full-chain gradient evidence exists;
  - clean-action guidance improves TacQualityEnergy on held-out samples;
  - trust-region updates keep actions inside the training range;
  - board evidence uses the full80 feature-cache DP and stronger fast100
    Foresight, not only smoke checkpoints.

The output is a JSON/Markdown gate report.  The final real-robot policy
validation remains a separate required step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/ptg_offline_production_gate")


DEFAULT_PATHS = {
    "evidence_summary": Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json"),
    "insertion_clean_refine": Path("/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json"),
    "insertion_full_chain": Path("/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json"),
    "board_foresight_fast100": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_foresight_fast100.json"),
    "board_feature_cache_dp": Path("/home/chenshuai/Project/output/board_production_chain_setup/board_dp_feature_cache_full80_fast32ema_w4096_e5.json"),
    "board_fast100_full_chain": Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast100_heldout32_K4_N64.json"),
    "controller_denoising_diagnostic": Path("/home/chenshuai/Project/output/tac_quality_controller_denoising_smoke/insertion_controller_denoising_final_s0001_K4_N4.json"),
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


def check(name: str, passed: bool, evidence: str, severity: str = "required") -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def num(d: Optional[Dict[str, Any]], dotted: str, default: float) -> float:
    value = get(d, dotted, None)
    if value is None:
        return default
    return float(value)


def build_gate(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {k: load_json(v) for k, v in paths.items()}
    missing = {k: str(v) for k, v in paths.items() if data[k] is None}

    insertion_clean = data["insertion_clean_refine"]
    insertion_full = data["insertion_full_chain"]
    board_fs100 = data["board_foresight_fast100"]
    board_dp = data["board_feature_cache_dp"]
    board_chain = data["board_fast100_full_chain"]
    denoising_diag = data["controller_denoising_diagnostic"]

    insertion_checks = [
        check(
            "Insertion full-chain gradient exists",
            bool(get(insertion_full, "interpretation.passes_full_chain_gradient", False)),
            f"score_improved_rate={get(insertion_full, 'summary.score_improved_rate')}, finite_grad_rate={get(insertion_full, 'summary.finite_grad_rate')}",
        ),
        check(
            "Insertion clean-action guidance improves score",
            bool(get(insertion_clean, "interpretation.passes_clean_refinement_sanity", False))
            and num(insertion_clean, "summary.refined_beats_base_rate", 0.0) >= 0.90
            and num(insertion_clean, "summary.score_delta.mean", 0.0) > 0.05,
            f"beats={get(insertion_clean, 'summary.refined_beats_base_rate')}, score_delta={get(insertion_clean, 'summary.score_delta.mean')}",
        ),
        check(
            "Insertion trust-region safety",
            num(insertion_clean, "summary.refined_hard_range_violation.max", 1.0) <= 1e-6
            and num(insertion_clean, "summary.action_delta_norm.p95", 999.0) <= 0.081,
            f"hard_violation={get(insertion_clean, 'summary.refined_hard_range_violation.max')}, delta_p95={get(insertion_clean, 'summary.action_delta_norm.p95')}",
        ),
    ]

    board_checks = [
        check(
            "Board stronger Foresight trained",
            bool(get(board_fs100, "interpretation.passes_board_foresight_fast100_training", False))
            and num(board_fs100, "metrics.relative_best_val_reduction_vs_fast20", 0.0) >= 0.20,
            f"best_val={get(board_fs100, 'metrics.best_val')}, reduction_vs_fast20={get(board_fs100, 'metrics.relative_best_val_reduction_vs_fast20')}",
        ),
        check(
            "Board full80 feature-cache DP trained",
            bool(get(board_dp, "interpretation.passes_feature_cache_full80_training", False))
            and num(board_dp, "n_episodes", 0.0) >= 80
            and num(board_dp, "n_windows", 0.0) >= 4096,
            f"episodes={get(board_dp, 'n_episodes')}, windows={get(board_dp, 'n_windows')}, final_loss={get(board_dp, 'final_train_loss')}",
        ),
        check(
            "Board fast100 full-chain guidance improves score",
            bool(get(board_chain, "interpretation.passes_board_dp_full_chain_smoke", False))
            and num(board_chain, "n_action_samples", 0.0) >= 128
            and num(board_chain, "summary.guided_beats_base_rate", 0.0) >= 0.95
            and num(board_chain, "summary.score_delta.mean", 0.0) > 0.02,
            f"samples={get(board_chain, 'n_action_samples')}, beats={get(board_chain, 'summary.guided_beats_base_rate')}, score_delta={get(board_chain, 'summary.score_delta.mean')}",
        ),
        check(
            "Board trust-region safety",
            num(board_chain, "summary.range_violation.max", 1.0) <= 1e-6
            and num(board_chain, "summary.norm_action_delta.p95", 999.0) <= 0.041,
            f"range_violation={get(board_chain, 'summary.range_violation.max')}, norm_delta_p95={get(board_chain, 'summary.norm_action_delta.p95')}",
        ),
        check(
            "Board smoothness is not degraded",
            num(board_chain, "summary.smoothness_delta.mean", 1.0) <= 0.0,
            f"smoothness_delta={get(board_chain, 'summary.smoothness_delta.mean')}",
        ),
    ]

    checks = insertion_checks + board_checks
    deployment_checks = [
        check(
            "Clean-action refinement is the recommended deployment mode",
            bool(get(insertion_clean, "interpretation.passes_clean_refinement_sanity", False))
            and bool(get(board_chain, "interpretation.passes_board_dp_full_chain_smoke", False)),
            "insertion_clean_refine_pass="
            f"{get(insertion_clean, 'interpretation.passes_clean_refinement_sanity')}, "
            "board_clean_refine_pass="
            f"{get(board_chain, 'interpretation.passes_board_dp_full_chain_smoke')}",
        ),
        check(
            "Every-step denoising controller is explicitly not production-ready",
            denoising_diag is not None
            and not bool(get(denoising_diag, "interpretation.passes_controller_denoising_smoke", True)),
            "passes_controller_denoising_smoke="
            f"{get(denoising_diag, 'interpretation.passes_controller_denoising_smoke')}, "
            f"beats={get(denoising_diag, 'summary.guided_beats_base_rate')}, "
            f"score_delta={get(denoising_diag, 'summary.score_delta.mean')}",
        ),
    ]
    checks += deployment_checks
    required_pass = all(c["passed"] for c in checks if c["severity"] == "required")
    result = {
        "scope": (
            "Offline gate only.  Passing this gate means the scorer/guidance stack "
            "is ready for production/robot dry-run validation, not that real-robot "
            "deployment has been completed."
        ),
        "paths": {k: str(v) for k, v in paths.items()},
        "missing_files": missing,
        "checks": {
            "insertion": insertion_checks,
            "board": board_checks,
            "deployment_policy": deployment_checks,
        },
        "metrics": {
            "insertion": {
                "clean_score_delta_mean": get(insertion_clean, "summary.score_delta.mean"),
                "clean_beats": get(insertion_clean, "summary.refined_beats_base_rate"),
                "hard_range_violation_max": get(insertion_clean, "summary.refined_hard_range_violation.max"),
            },
            "board": {
                "foresight_fast100_best_val": get(board_fs100, "metrics.best_val"),
                "foresight_fast100_reduction_vs_fast20": get(board_fs100, "metrics.relative_best_val_reduction_vs_fast20"),
                "feature_cache_dp_final_loss": get(board_dp, "final_train_loss"),
                "full_chain_score_delta_mean": get(board_chain, "summary.score_delta.mean"),
                "full_chain_beats": get(board_chain, "summary.guided_beats_base_rate"),
                "full_chain_range_violation_max": get(board_chain, "summary.range_violation.max"),
                "full_chain_smoothness_delta_mean": get(board_chain, "summary.smoothness_delta.mean"),
            },
            "deployment_policy": {
                "recommended_mode": "final_clean_action_trust_region_refinement",
                "denoising_controller_pass": get(denoising_diag, "interpretation.passes_controller_denoising_smoke"),
                "denoising_controller_beats": get(denoising_diag, "summary.guided_beats_base_rate"),
                "denoising_controller_score_delta_mean": get(denoising_diag, "summary.score_delta.mean"),
                "not_recommended_mode": "unconditional_every_ddpm_step_controller_guidance",
            },
        },
        "offline_production_gate_pass": bool(required_pass and not missing),
        "remaining_required_step": "Real robot / final production policy validation.",
        "recommended_deployment_mode": "final_clean_action_trust_region_refinement",
        "research_only_mode": "late_step_or_every_step_denoising_controller_guidance",
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# PTG Offline Production Gate",
        "",
        f"- offline_production_gate_pass: `{result['offline_production_gate_pass']}`",
        f"- scope: {result['scope']}",
        f"- remaining_required_step: {result['remaining_required_step']}",
        "",
        "## Checks",
        "",
    ]
    for group, checks in result["checks"].items():
        lines.append(f"### {group}")
        lines.append("")
        for item in checks:
            status = "PASS" if item["passed"] else "FAIL"
            lines.append(f"- **{status}** {item['name']}: {item['evidence']}")
        lines.append("")
    lines.extend(["## Metrics", "", "```json", json.dumps(result["metrics"], ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_gate(DEFAULT_PATHS)
    json_path = out_dir / "ptg_offline_production_gate.json"
    md_path = out_dir / "ptg_offline_production_gate.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "offline_production_gate_pass": result["offline_production_gate_pass"],
        "remaining_required_step": result["remaining_required_step"],
    }, ensure_ascii=False, indent=2))
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
