"""Build a deployment manifest for TacQuality DP classifier guidance.

The manifest is a machine-readable handoff artifact for connecting the current
scorer/guidance stack to a DP policy or robot dry-run.  It records:

  - selected task-conditioned scorers and checkpoints;
  - runtime/refiner/config modules;
  - evidence JSON files and their pass/fail status;
  - the exact score API and remaining validation gap.

It intentionally does not claim real-robot completion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest")


PATHS = {
    "insertion_scorer_ckpt": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"),
    "board_scorer_ckpt": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt"),
    "distilled_energy_ckpt": Path(
        "/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt"
    ),
    "scorer_selection_gate": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json"
    ),
    "runtime_contract": Path("/home/chenshuai/Project/output/tac_quality_guidance_runtime/runtime_contract_sanity.json"),
    "trust_region_guidance": Path("/home/chenshuai/Project/output/tac_quality_trust_region_guidance/trust_region_sanity.json"),
    "dp_guidance_controller": Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_sanity.json"),
    "dp_guidance_controller_real_sample": Path("/home/chenshuai/Project/output/tac_quality_dp_guidance_controller/controller_real_sample_audit.json"),
    "score_calibration": Path("/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json"),
    "evidence_summary": Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json"),
    "offline_gate": Path("/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json"),
    "insertion_full_chain": Path("/home/chenshuai/Project/output/full_chain_guidance_gradient/insertion_full_chain_energy_clipped_K8_N16.json"),
    "insertion_clean_refine": Path("/home/chenshuai/Project/output/clean_action_energy_refinement/insertion_clean_refine_constrained_K4_N40.json"),
    "insertion_distilled_clean_refine": Path(
        "/home/chenshuai/Project/output/insertion_distilled_clean_refine_comparison/"
        "n24_k4/insertion_distilled_clean_refine_comparison.json"
    ),
    "board_full_chain_fast100": Path("/home/chenshuai/Project/output/board_dp_denoising_full_chain_smoke/board_dp_feature_cache_full80_fast32ema_w4096_e5_fast100_heldout32_K4_N64.json"),
    "board_distilled_clean_refine": Path(
        "/home/chenshuai/Project/output/board_dp_distilled_clean_refine_comparison/"
        "fast20_heldout32_n64/board_dp_distilled_clean_refine_comparison.json"
    ),
    "goal_completion_audit": Path("/home/chenshuai/Project/output/tac_quality_goal_audit/tac_quality_goal_completion_audit.json"),
}


MODULES = {
    "guidance_config": Path("TFAC_V5/tac_quality_guidance_config.py"),
    "guidance_runtime": Path("TFAC_V5/tac_quality_guidance_runtime.py"),
    "trust_region_refiner": Path("TFAC_V5/tac_quality_trust_region_guidance.py"),
    "dp_guidance_controller": Path("TFAC_V5/tac_quality_dp_guidance_controller.py"),
    "real_rollout_quality_gate": Path("TFAC_V5/eval_real_rollout_quality_gate.py"),
    "real_rollout_validation_prep": Path("TFAC_V5/prepare_real_rollout_validation.py"),
    "real_rollout_sample_size_plan": Path("TFAC_V5/plan_real_rollout_sample_size.py"),
    "real_rollout_experiment_packet": Path("TFAC_V5/build_real_rollout_experiment_packet.py"),
    "scorer_selection_gate": Path("TFAC_V5/build_tac_quality_scorer_selection_gate.py"),
    "summary_builder": Path("TFAC_V5/summarize_ptg_guidance_evidence.py"),
    "goal_completion_audit": Path("TFAC_V5/audit_tac_quality_goal_completion.py"),
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


def file_info(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else None,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_manifest() -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in PATHS.items() if path.suffix == ".json"}
    missing = {name: str(path) for name, path in {**PATHS, **MODULES}.items() if not path.exists()}

    checks = [
        {
            "name": "scorer_selection_gate_pass",
            "passed": bool(get(data["scorer_selection_gate"], "selection_gate_pass", False))
            and get(data["scorer_selection_gate"], "selection.current_default_board_scorer")
            == "PTGProxyScorerV2Runtime"
            and get(data["scorer_selection_gate"], "selection.promoted_ablation_candidate")
            == "DistilledTacQualityEnergyRuntime"
            and get(data["scorer_selection_gate"], "selection.distilled_replacement_status") == "not_yet_replacement",
            "evidence": {
                "selection_gate_pass": get(data["scorer_selection_gate"], "selection_gate_pass"),
                "status": get(data["scorer_selection_gate"], "status"),
                "current_default_board_scorer": get(
                    data["scorer_selection_gate"], "selection.current_default_board_scorer"
                ),
                "promoted_ablation_candidate": get(
                    data["scorer_selection_gate"], "selection.promoted_ablation_candidate"
                ),
                "distilled_replacement_status": get(
                    data["scorer_selection_gate"], "selection.distilled_replacement_status"
                ),
            },
        },
        {
            "name": "all_manifest_files_exist",
            "passed": not missing,
            "evidence": missing,
        },
        {
            "name": "runtime_contract_pass",
            "passed": bool(get(data["runtime_contract"], "passes_runtime_contract_sanity", False)),
            "evidence": {
                "insertion_action_grad": get(data["runtime_contract"], "insertion.action_grad_norm_mean"),
                "board_joint_grad": get(data["runtime_contract"], "board.joint_action_grad_norm_mean"),
                "board_eef_grad": get(data["runtime_contract"], "board.eef_action_grad_norm_mean"),
            },
        },
        {
            "name": "trust_region_guidance_pass",
            "passed": bool(get(data["trust_region_guidance"], "passes_trust_region_guidance_sanity", False)),
            "evidence": {
                "insertion_improved": get(data["trust_region_guidance"], "insertion.improved_rate"),
                "board_improved": get(data["trust_region_guidance"], "board.improved_rate"),
            },
        },
        {
            "name": "dp_guidance_controller_pass",
            "passed": bool(get(data["dp_guidance_controller"], "passes_controller_sanity", False))
            and bool(get(data["dp_guidance_controller_real_sample"], "overall_pass", False)),
            "evidence": {
                "sanity_pass": get(data["dp_guidance_controller"], "passes_controller_sanity"),
                "real_sample_pass": get(data["dp_guidance_controller_real_sample"], "overall_pass"),
                "insertion_improved": get(data["dp_guidance_controller_real_sample"], "insertion.report.improved_rate"),
                "board_improved": get(data["dp_guidance_controller_real_sample"], "board.report.improved_rate"),
                "stale_gradient_reuse_allowed": get(data["dp_guidance_controller"], "insertion.guardrails.stale_gradient_reuse_allowed"),
            },
        },
        {
            "name": "score_calibration_pass",
            "passed": get(data["score_calibration"], "recommendation.insertion") == "energy"
            and get(data["score_calibration"], "recommendation.board") == "quality",
            "evidence": get(data["score_calibration"], "recommendation"),
        },
        {
            "name": "offline_gate_pass",
            "passed": bool(get(data["offline_gate"], "offline_production_gate_pass", False)),
            "evidence": get(data["offline_gate"], "remaining_required_step"),
        },
        {
            "name": "evidence_summary_keeps_real_robot_gap",
            "passed": (
                get(data["evidence_summary"], "completion_assessment.objective_complete", get(data["evidence_summary"], "objective_complete"))
                is False
            )
            and "real-robot" in str(
                get(data["evidence_summary"], "completion_assessment.reason", get(data["evidence_summary"], "reason", ""))
            ),
            "evidence": get(data["evidence_summary"], "completion_assessment", data["evidence_summary"]),
        },
        {
            "name": "goal_completion_audit_keeps_real_rollout_gap",
            "passed": get(data["goal_completion_audit"], "objective_complete") is False
            and get(data["goal_completion_audit"], "status") == "incomplete"
            and get(data["goal_completion_audit"], "next_required_step") is not None,
            "evidence": {
                "objective_complete": get(data["goal_completion_audit"], "objective_complete"),
                "status": get(data["goal_completion_audit"], "status"),
                "n_blockers": len(get(data["goal_completion_audit"], "blockers", []) or []),
                "next_required_step": get(data["goal_completion_audit"], "next_required_step"),
            },
        },
    ]

    manifest = {
        "name": "TacQualityEnergy DP classifier guidance manifest",
        "git_commit": git_commit(),
        "scope": (
            "Offline-ready scorer/guidance package for socket insertion and board wiping. "
            "This manifest does not claim real-robot validation."
        ),
        "deployment_policy": {
            "recommended_mode": "final_clean_action_trust_region_refinement",
            "allowed_for_robot_dry_run": [
                "Run the original DP denoising sampler first.",
                "Predict tactile consequence with the current Foresight model.",
                "Score with task-conditioned TacQualityEnergy.",
                "Apply bounded accept-only refinement on the clean/final action.",
                "Recompute action -> Foresight -> TacQualityEnergy gradient for every accepted update.",
            ],
            "research_only_modes": [
                "late_step_denoising_controller_guidance",
                "controller_in_every_ddpm_step",
            ],
            "not_recommended_yet": [
                "Unconditional guidance at every DDPM denoising step.",
                "Large guidance scale without trust-region projection.",
                "Using p_good/log_p_good alone as the guidance potential.",
                "Reusing cached or stale gradients across denoising steps.",
                "Using a single cross-task probability threshold as the final quality decision.",
            ],
            "reason": (
                "Offline scorer quality, full-chain gradients, clean-action refinement, "
                "board heldout chain, and production gate pass.  However controller-in-denoising "
                "diagnostics show local step score gains do not reliably translate into final "
                "denoised sample improvement, so final/clean-action refinement is the current "
                "safe deployment mode."
            ),
            "completion_status": "offline_ready_not_robot_validated",
        },
        "score_api": {
            "runtime": "TFAC_V5.tac_quality_guidance_runtime.TacQualityGuidanceRuntime",
            "score_call": "runtime.score(task, predicted_tactile, action, mode='profile')",
            "refiner": "TFAC_V5.tac_quality_trust_region_guidance.TacQualityTrustRegionRefiner",
            "refine_call": "refiner.refine(action, score_fn)",
            "dp_controller": "TFAC_V5.tac_quality_dp_guidance_controller.TacQualityDPGuidanceController",
            "dp_controller_call": "guided_action, report = controller.guide(action, current_score_fn)",
            "guardrail": "Do not pass cached gradients; current_score_fn must recompute action -> Foresight -> TacQuality score each guidance step.",
            "real_rollout_gate": (
                "python TFAC_V5/eval_real_rollout_quality_gate.py "
                "--task {insertion,board} --baseline_dir <baseline_hdf5_dir> "
                "--guided_dir <guided_hdf5_dir>"
            ),
            "real_rollout_validation_prep": (
                "python TFAC_V5/prepare_real_rollout_validation.py "
                "--task {insertion,board} --baseline_dir <baseline_hdf5_dir> "
                "--guided_dir <guided_hdf5_dir>"
            ),
            "real_rollout_sample_size_plan": (
                "python TFAC_V5/plan_real_rollout_sample_size.py "
                "--task {insertion,board}"
            ),
            "real_rollout_experiment_packet": (
                "python TFAC_V5/build_real_rollout_experiment_packet.py --tag formal_paired12"
            ),
        },
        "tasks": {
            "insertion": {
                "scorer": "InsertionRiskScorerRuntime",
                "checkpoint": file_info(PATHS["insertion_scorer_ckpt"]),
                "profile_energy": "0.50*quality_logit + 0.10*binary_margin",
                "calibration_mode": get(data["score_calibration"], "recommendation.insertion"),
                "trust_region": {
                    "steps": get(data["trust_region_guidance"], "insertion.config.steps"),
                    "step_size": get(data["trust_region_guidance"], "insertion.config.step_size"),
                    "max_total_delta": get(data["trust_region_guidance"], "insertion.config.max_total_delta"),
                    "accept_only_improved": get(data["trust_region_guidance"], "insertion.config.accept_only_improved"),
                },
            },
            "board": {
                "scorer": "PTGProxyScorerV2Runtime",
                "checkpoint": file_info(PATHS["board_scorer_ckpt"]),
                "ablation_candidate": {
                    "scorer": "DistilledTacQualityEnergyRuntime",
                    "checkpoint": file_info(PATHS["distilled_energy_ckpt"]),
                    "status": get(
                        data["scorer_selection_gate"],
                        "selection.distilled_replacement_status",
                        "not_evaluated",
                    ),
                    "selection_gate": str(PATHS["scorer_selection_gate"]),
                },
                "profile_energy": "0.75*quality_logit + 0.10*binary_margin",
                "calibration_mode": get(data["score_calibration"], "recommendation.board"),
                "trust_region": {
                    "steps": get(data["trust_region_guidance"], "board.config.steps"),
                    "step_size": get(data["trust_region_guidance"], "board.config.step_size"),
                    "max_total_delta": get(data["trust_region_guidance"], "board.config.max_total_delta"),
                    "accept_only_improved": get(data["trust_region_guidance"], "board.config.accept_only_improved"),
                },
            },
        },
        "modules": {name: file_info(path) for name, path in MODULES.items()},
        "evidence_files": {name: file_info(path) for name, path in PATHS.items()},
        "checks": checks,
        "deployment_manifest_pass": all(item["passed"] for item in checks),
        "remaining_required_step": "Real robot / final production policy validation.",
    }
    return manifest


def write_markdown(manifest: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Guidance Deployment Manifest",
        "",
        f"- deployment_manifest_pass: `{manifest['deployment_manifest_pass']}`",
        f"- git_commit: `{manifest['git_commit']}`",
        f"- scope: {manifest['scope']}",
        f"- remaining_required_step: {manifest['remaining_required_step']}",
        "",
        "## Deployment Policy",
        "",
        f"- recommended_mode: `{manifest['deployment_policy']['recommended_mode']}`",
        f"- completion_status: `{manifest['deployment_policy']['completion_status']}`",
        f"- reason: {manifest['deployment_policy']['reason']}",
        "",
        "### Allowed Robot Dry-Run Flow",
        "",
    ]
    for item in manifest["deployment_policy"]["allowed_for_robot_dry_run"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Research-Only Modes",
            "",
        ]
    )
    for item in manifest["deployment_policy"]["research_only_modes"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "### Not Recommended Yet",
            "",
        ]
    )
    for item in manifest["deployment_policy"]["not_recommended_yet"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## API",
            "",
            f"- score: `{manifest['score_api']['score_call']}`",
            f"- refine: `{manifest['score_api']['refine_call']}`",
            f"- real rollout gate: `{manifest['score_api']['real_rollout_gate']}`",
            "",
            "## Tasks",
            "",
        ]
    )
    for task, info in manifest["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- scorer: `{info['scorer']}`",
                f"- checkpoint: `{info['checkpoint']['path']}`",
                f"- profile_energy: `{info['profile_energy']}`",
                f"- calibration_mode: `{info['calibration_mode']}`",
                f"- trust_region: `{json.dumps(info['trust_region'], ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(["## Checks", ""])
    for item in manifest["checks"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(f"- **{status}** {item['name']}: `{json.dumps(item['evidence'], ensure_ascii=False)}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    json_path = out_dir / "tac_quality_guidance_manifest.json"
    md_path = out_dir / "tac_quality_guidance_manifest.md"
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(manifest, md_path)
    print(json.dumps({
        "deployment_manifest_pass": manifest["deployment_manifest_pass"],
        "remaining_required_step": manifest["remaining_required_step"],
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
