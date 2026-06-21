#!/usr/bin/env python3
"""Audit insertion good-margin scorer consistency across evidence artifacts.

The insertion arm has several evidence sources:

1. score-mode ablation, which shows why good_margin is preferred over p_good,
2. rollout config, which is what serving loads,
3. DDPM-step guidance sweep, which checks denoising-loop guidance,
4. final-action and denoising-step serving smokes,
5. current TacQuality scorecard, which is the human-facing summary.

This script checks that all sources agree on the same deployable insertion
scorer.  It is a cheap preflight before real insertion rollouts so the
classifier/energy guidance path is not evaluated with stale score modes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
DEFAULT_SCORE_MODE_ABLATION = Path(
    "/home/chenshuai/Project/output/tac_quality_score_mode_ablation/"
    "insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/"
    "insertion_score_mode_ablation.json"
)
DEFAULT_DDPM_STEP_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "insertion_0401_good_margin_protected_multiep8_start2_seed2_t0_s001/"
    "insertion_ddpm_step_guidance_sweep.json"
)
DEFAULT_FINAL_ACTION_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "insertion_0401_good_margin_guided_smoke_20260619/guided_server_dry_run_smoke.json"
)
DEFAULT_DENOISING_STEP_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "insertion_good_margin_denoising_step_smoke_20260619/guided_server_dry_run_smoke.json"
)
DEFAULT_SCORECARD = Path(
    "/home/chenshuai/Project/output/tac_quality_current_scorecard/"
    "current_tac_quality_scorecard.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/insertion_config_consistency")


EXPECTED_ARM = "good_margin_guided"
EXPECTED_RUNTIME = "InsertionRiskScorerRuntime"
EXPECTED_SCORE_MODE = "good_margin"
EXPECTED_ADAPTER_FINAL = "final_clean_action_trust_region_refinement"
EXPECTED_ADAPTER_DENOISING = "denoising_step_tac_quality_guidance"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} JSON root is not an object")
    return data


def get(data: Mapping[str, Any] | None, dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def stat_mean(value: Any) -> float | None:
    if isinstance(value, Mapping) and "mean" in value:
        value = value["mean"]
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def check_item(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "pass": bool(passed), "detail": detail}


def write_markdown(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Insertion Config Consistency Audit",
        "",
        f"Created: `{result['created_at']}`",
        "",
        "## Summary",
        "",
        f"- pass: `{result['pass']}`",
        f"- expected arm: `{result['expected']['arm']}`",
        f"- expected runtime: `{result['expected']['runtime']}`",
        f"- expected score mode: `{result['expected']['score_mode']}`",
        "",
        "## Checks",
        "",
        "| check | pass | detail |",
        "|---|---:|---|",
    ]
    for item in result["checks"]:
        detail = json.dumps(item["detail"], ensure_ascii=False)
        if len(detail) > 500:
            detail = detail[:497] + "..."
        lines.append(f"| `{item['name']}` | `{item['pass']}` | `{detail}` |")
    lines.extend([
        "",
        "## Evidence Boundary",
        "",
        result["evidence_boundary"],
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rollout = load_json(args.rollout_config)
    ablation = load_json(args.score_mode_ablation)
    ddpm = load_json(args.ddpm_step_sweep)
    final_smoke = load_json(args.final_action_smoke)
    denoise_smoke = load_json(args.denoising_step_smoke)
    scorecard = load_json(args.scorecard)

    arm = get(rollout, f"tasks.insertion.{args.arm}", {})
    refiner = get(arm, "refiner", {})
    energy = get(refiner, "energy", {})
    checkpoint = get(arm, "checkpoint", {})
    ckpt_path = Path(get(checkpoint, "path", ""))

    ablation_good = get(ablation, "summary.good_margin", {})
    ablation_good_margin_delta = stat_mean(get(ablation_good, "cross_scores.good_margin.delta"))
    ablation_energy_delta = stat_mean(get(ablation_good, "cross_scores.energy.delta"))
    ablation_profile_delta = stat_mean(get(ablation_good, "cross_scores.profile.delta"))
    ablation_p_good_delta = stat_mean(get(ablation_good, "cross_scores.p_good.delta"))
    ablation_good_improve = get(ablation_good, "cross_scores.good_margin.improve_rate")

    ddpm_summary = get(ddpm, "summary", {})
    ddpm_improve = get(ddpm_summary, "final_score_improve_rate")
    ddpm_finite = stat_mean(get(ddpm_summary, "finite_grad_rate"))
    ddpm_positive = stat_mean(get(ddpm_summary, "positive_grad_rate"))
    ddpm_final_accept = stat_mean(get(ddpm_summary, "final_accept_rate"))

    scorecard_ins = get(scorecard, "task_scorecards.insertion", {})
    scorecard_rec = get(scorecard, "recommendation.insertion", {})

    checks = [
        check_item(
            "rollout_recommends_good_margin_arm",
            get(rollout, "recommended_insertion_arm") == args.arm,
            {"recommended_insertion_arm": get(rollout, "recommended_insertion_arm")},
        ),
        check_item(
            "rollout_arm_exists",
            isinstance(arm, Mapping) and bool(arm),
            {"arm": args.arm},
        ),
        check_item(
            "rollout_runtime_and_score_mode",
            get(arm, "scorer_runtime") == EXPECTED_RUNTIME
            and get(refiner, "score_mode") == EXPECTED_SCORE_MODE,
            {
                "runtime": get(arm, "scorer_runtime"),
                "score_mode": get(refiner, "score_mode"),
            },
        ),
        check_item(
            "rollout_energy_is_good_margin",
            get(energy, "source") == "InsertionRiskScorerRuntime.good_margin"
            and get(energy, "definition") == "binary_logits[:, good] - binary_logits[:, bad]",
            {
                "source": get(energy, "source"),
                "definition": get(energy, "definition"),
            },
        ),
        check_item(
            "rollout_points_to_evidence",
            str(args.score_mode_ablation) == str(get(energy, "selected_by"))
            and str(args.ddpm_step_sweep) == str(get(energy, "ddpm_step_sweep")),
            {
                "selected_by": get(energy, "selected_by"),
                "ddpm_step_sweep": get(energy, "ddpm_step_sweep"),
                "expected_ablation": str(args.score_mode_ablation),
                "expected_ddpm": str(args.ddpm_step_sweep),
            },
        ),
        check_item(
            "checkpoint_exists",
            bool(get(checkpoint, "exists")) and ckpt_path.exists(),
            {
                "checkpoint": get(checkpoint, "path"),
                "rollout_exists": get(checkpoint, "exists"),
                "filesystem_exists": ckpt_path.exists(),
            },
        ),
        check_item(
            "ablation_contains_good_margin_mode",
            EXPECTED_SCORE_MODE in (get(ablation, "guidance_modes", []) or [])
            and EXPECTED_SCORE_MODE in (get(ablation, "eval_modes", []) or [])
            and isinstance(ablation_good, Mapping)
            and bool(ablation_good),
            {
                "guidance_modes": get(ablation, "guidance_modes", []),
                "eval_modes": get(ablation, "eval_modes", []),
                "has_summary_good_margin": bool(ablation_good),
            },
        ),
        check_item(
            "ablation_good_margin_not_saturated",
            ablation_good_margin_delta is not None
            and ablation_p_good_delta is not None
            and ablation_good_margin_delta > 1e-4
            and abs(ablation_p_good_delta) < 1e-8,
            {
                "good_margin_delta_mean": ablation_good_margin_delta,
                "p_good_delta_mean": ablation_p_good_delta,
            },
        ),
        check_item(
            "ablation_good_margin_is_strongest_cross_score",
            ablation_good_margin_delta is not None
            and ablation_energy_delta is not None
            and ablation_profile_delta is not None
            and ablation_good_margin_delta >= ablation_energy_delta
            and ablation_good_margin_delta >= ablation_profile_delta
            and float(ablation_good_improve or 0.0) >= 0.90,
            {
                "good_margin_delta_mean": ablation_good_margin_delta,
                "energy_delta_mean": ablation_energy_delta,
                "profile_delta_mean": ablation_profile_delta,
                "good_margin_improve_rate": ablation_good_improve,
            },
        ),
        check_item(
            "ddpm_step_sweep_passes",
            float(ddpm_improve or 0.0) >= 0.90
            and float(ddpm_finite or 0.0) >= 0.999
            and float(ddpm_positive or 0.0) >= 0.999
            and float(ddpm_final_accept or 0.0) >= 0.999,
            {
                "final_score_improve_rate": ddpm_improve,
                "finite_grad_rate_mean": ddpm_finite,
                "positive_grad_rate_mean": ddpm_positive,
                "final_accept_rate_mean": ddpm_final_accept,
                "score_delta_mean": stat_mean(get(ddpm_summary, "final_score_delta")),
                "action_delta_norm_mean": stat_mean(get(ddpm_summary, "guided_action_delta_norm")),
            },
        ),
        check_item(
            "final_action_smoke_passes",
            bool(get(final_smoke, "dry_run_guidance_smoke_pass"))
            and get(final_smoke, "task") == "insertion"
            and get(final_smoke, "arm") == args.arm
            and get(final_smoke, "report.scorer_runtime") == EXPECTED_RUNTIME
            and get(final_smoke, "report.score_mode") == EXPECTED_SCORE_MODE
            and get(final_smoke, "report.adapter_policy") == EXPECTED_ADAPTER_FINAL
            and bool(get(final_smoke, "not_reranking")),
            {
                "pass": get(final_smoke, "dry_run_guidance_smoke_pass"),
                "task": get(final_smoke, "task"),
                "arm": get(final_smoke, "arm"),
                "runtime": get(final_smoke, "report.scorer_runtime"),
                "score_mode": get(final_smoke, "report.score_mode"),
                "adapter_policy": get(final_smoke, "report.adapter_policy"),
                "not_reranking": get(final_smoke, "not_reranking"),
                "score_delta_mean": stat_mean(get(final_smoke, "report.score_delta")),
                "normalized_action_delta_mean": stat_mean(get(final_smoke, "report.normalized_action_delta")),
            },
        ),
        check_item(
            "denoising_step_smoke_passes",
            bool(get(denoise_smoke, "dry_run_guidance_smoke_pass"))
            and get(denoise_smoke, "task") == "insertion"
            and get(denoise_smoke, "arm") == args.arm
            and get(denoise_smoke, "report.scorer_runtime") == EXPECTED_RUNTIME
            and get(denoise_smoke, "report.score_mode") == EXPECTED_SCORE_MODE
            and get(denoise_smoke, "report.adapter_policy") == EXPECTED_ADAPTER_DENOISING
            and bool(get(denoise_smoke, "not_reranking"))
            and bool(get(denoise_smoke, "report.every_step_ddpm_guidance")),
            {
                "pass": get(denoise_smoke, "dry_run_guidance_smoke_pass"),
                "task": get(denoise_smoke, "task"),
                "arm": get(denoise_smoke, "arm"),
                "runtime": get(denoise_smoke, "report.scorer_runtime"),
                "score_mode": get(denoise_smoke, "report.score_mode"),
                "adapter_policy": get(denoise_smoke, "report.adapter_policy"),
                "not_reranking": get(denoise_smoke, "not_reranking"),
                "every_step_ddpm_guidance": get(denoise_smoke, "report.every_step_ddpm_guidance"),
                "score_delta_mean": stat_mean(get(denoise_smoke, "report.score_delta")),
                "normalized_action_delta_mean": stat_mean(get(denoise_smoke, "report.normalized_action_delta")),
            },
        ),
        check_item(
            "scorecard_matches_insertion_recommendation",
            get(scorecard_rec, "arm") == args.arm
            and get(scorecard_rec, "runtime") == EXPECTED_RUNTIME
            and get(scorecard_rec, "score_mode") == EXPECTED_SCORE_MODE
            and get(scorecard_ins, "arm") == args.arm
            and get(scorecard_ins, "runtime") == EXPECTED_RUNTIME
            and get(scorecard_ins, "score_mode") == EXPECTED_SCORE_MODE,
            {
                "recommendation": {
                    "arm": get(scorecard_rec, "arm"),
                    "runtime": get(scorecard_rec, "runtime"),
                    "score_mode": get(scorecard_rec, "score_mode"),
                },
                "task_scorecard": {
                    "arm": get(scorecard_ins, "arm"),
                    "runtime": get(scorecard_ins, "runtime"),
                    "score_mode": get(scorecard_ins, "score_mode"),
                },
            },
        ),
        check_item(
            "scorecard_keeps_real_rollout_gap",
            get(scorecard, "evidence_levels.goal_complete") is False
            and get(scorecard, "evidence_levels.real_paired_rollout_complete") is False,
            {
                "goal_complete": get(scorecard, "evidence_levels.goal_complete"),
                "real_paired_rollout_complete": get(scorecard, "evidence_levels.real_paired_rollout_complete"),
            },
        ),
    ]
    passed = all(item["pass"] for item in checks)

    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pass": passed,
        "task": "insertion",
        "arm": args.arm,
        "expected": {
            "arm": EXPECTED_ARM,
            "runtime": EXPECTED_RUNTIME,
            "score_mode": EXPECTED_SCORE_MODE,
            "final_action_adapter": EXPECTED_ADAPTER_FINAL,
            "denoising_step_adapter": EXPECTED_ADAPTER_DENOISING,
        },
        "checks": checks,
        "summary": {
            "ablation_good_margin_delta_mean": ablation_good_margin_delta,
            "ablation_p_good_delta_mean": ablation_p_good_delta,
            "ablation_good_margin_improve_rate": ablation_good_improve,
            "ddpm_final_score_improve_rate": ddpm_improve,
            "ddpm_score_delta_mean": stat_mean(get(ddpm_summary, "final_score_delta")),
            "ddpm_action_delta_norm_mean": stat_mean(get(ddpm_summary, "guided_action_delta_norm")),
            "final_action_smoke_score_delta_mean": stat_mean(get(final_smoke, "report.score_delta")),
            "denoising_step_smoke_score_delta_mean": stat_mean(get(denoise_smoke, "report.score_delta")),
            "scorecard_goal_complete": get(scorecard, "evidence_levels.goal_complete"),
            "real_paired_rollout_complete": get(scorecard, "evidence_levels.real_paired_rollout_complete"),
        },
        "inputs": {
            "rollout_config": file_info(args.rollout_config),
            "score_mode_ablation": file_info(args.score_mode_ablation),
            "ddpm_step_sweep": file_info(args.ddpm_step_sweep),
            "final_action_smoke": file_info(args.final_action_smoke),
            "denoising_step_smoke": file_info(args.denoising_step_smoke),
            "scorecard": file_info(args.scorecard),
        },
        "evidence_boundary": (
            "This is a consistency/preflight audit for offline and serving artifacts. "
            "It proves that insertion guidance is configured as the selected "
            "good_margin logit-margin scorer and that final-action and denoising-step "
            "serving paths load the same runtime/score mode. It does not prove real "
            "robot insertion improvement."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--score_mode_ablation", type=Path, default=DEFAULT_SCORE_MODE_ABLATION)
    parser.add_argument("--ddpm_step_sweep", type=Path, default=DEFAULT_DDPM_STEP_SWEEP)
    parser.add_argument("--final_action_smoke", type=Path, default=DEFAULT_FINAL_ACTION_SMOKE)
    parser.add_argument("--denoising_step_smoke", type=Path, default=DEFAULT_DENOISING_STEP_SMOKE)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arm", default=EXPECTED_ARM)
    args = parser.parse_args()

    result = run(args)
    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    json_path = out_dir / "insertion_config_consistency.json"
    md_path = out_dir / "insertion_config_consistency.md"
    result["paths"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "pass": result["pass"],
                "failed_checks": [item["name"] for item in result["checks"] if not item["pass"]],
                "summary": result["summary"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
