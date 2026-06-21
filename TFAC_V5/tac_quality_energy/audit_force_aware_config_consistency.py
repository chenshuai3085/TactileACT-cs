#!/usr/bin/env python3
"""Audit force-aware board scorer consistency across evidence artifacts.

The force-aware board arm has several evidence sources:

1. score-weight sweep, which selects the current score preset,
2. rollout config, which is what serving loads,
3. real-HDF5-window serving audit, which records the runtime's actual weights,
4. current TacQuality scorecard, which is the human-facing summary.

This script checks that all four agree.  It is intended as a cheap preflight
before real board rollouts so that classifier/energy guidance is not evaluated
with stale or mismatched score weights.
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
DEFAULT_WEIGHT_SWEEP = Path(
    "/home/chenshuai/Project/output/force_aware_score_weight_sweep/"
    "20260621_104503/force_aware_score_weight_sweep.json"
)
DEFAULT_SERVING_AUDIT = Path(
    "/home/chenshuai/Project/output/force_aware_serving_real_window_audit/"
    "20260621_110440/force_aware_serving_real_window_audit.json"
)
DEFAULT_SCORECARD = Path(
    "/home/chenshuai/Project/output/tac_quality_current_scorecard/"
    "current_tac_quality_scorecard.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/force_aware_config_consistency")


EXPECTED_PRESET = "margin_only"
EXPECTED_WEIGHTS = {
    "band_margin": 1.0,
    "contact_logprob": 0.0,
    "force_center": 0.0,
    "force_smooth": 0.0,
    "action_smooth": 0.0,
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} JSON root is not an object")
    return data


def get(data: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def weights_equal(a: Any, b: Any) -> bool:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False
    if set(a.keys()) != set(b.keys()):
        return False
    for key in b:
        try:
            if abs(float(a[key]) - float(b[key])) > 1e-9:
                return False
        except Exception:
            return False
    return True


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
        "# Force-Aware Config Consistency Audit",
        "",
        f"Created: `{result['created_at']}`",
        "",
        "## Summary",
        "",
        f"- pass: `{result['pass']}`",
        f"- expected preset: `{result['expected']['score_preset']}`",
        f"- expected weights: `{result['expected']['weights']}`",
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
    sweep = load_json(args.weight_sweep)
    serving = load_json(args.serving_audit)
    scorecard = load_json(args.scorecard)

    arm = get(rollout, f"tasks.board.{args.arm}", {})
    rollout_energy = get(arm, "refiner.energy", {})
    sweep_best = get(sweep, "best", {})
    scorecard_force = get(scorecard, "task_scorecards.board.force_aware_foresight_guidance", {})
    scorecard_pref = get(scorecard, "recommendation.board_scientific_preference", {})

    rollout_weights = get(rollout_energy, "weights", {})
    sweep_weights = get(sweep_best, "weights", {})
    serving_weights = get(serving, "setup.score_weights", {})
    runtime_weights = get(serving, "setup.runtime_summary.score_weights", {})
    scorecard_weights = get(scorecard_force, "score_weights", {})

    serving_path_from_scorecard = get(scorecard_force, "serving_real_window_audit.path")
    sweep_path_from_rollout = get(rollout_energy, "selected_by")
    sweep_path_from_scorecard = get(scorecard_force, "score_weight_sweep.path")

    checks = [
        check_item(
            "rollout_arm_exists",
            isinstance(arm, Mapping) and bool(arm),
            {"arm": args.arm},
        ),
        check_item(
            "rollout_runtime_is_force_aware",
            get(arm, "scorer_runtime") == "ForceAwareForesightGuidanceRuntime",
            {"runtime": get(arm, "scorer_runtime")},
        ),
        check_item(
            "rollout_score_preset",
            get(rollout_energy, "score_preset") == EXPECTED_PRESET,
            {"score_preset": get(rollout_energy, "score_preset")},
        ),
        check_item(
            "weight_sweep_best_preset",
            get(sweep_best, "name") == EXPECTED_PRESET,
            {"best": get(sweep_best, "name")},
        ),
        check_item(
            "all_weights_match_expected",
            all(
                weights_equal(w, EXPECTED_WEIGHTS)
                for w in [rollout_weights, sweep_weights, serving_weights, runtime_weights, scorecard_weights]
            ),
            {
                "rollout": rollout_weights,
                "sweep": sweep_weights,
                "serving_setup": serving_weights,
                "serving_runtime": runtime_weights,
                "scorecard": scorecard_weights,
            },
        ),
        check_item(
            "scorecard_preference_matches",
            get(scorecard_pref, "arm") == args.arm
            and get(scorecard_pref, "runtime") == "ForceAwareForesightGuidanceRuntime"
            and get(scorecard_pref, "score_preset") == EXPECTED_PRESET,
            {
                "arm": get(scorecard_pref, "arm"),
                "runtime": get(scorecard_pref, "runtime"),
                "score_preset": get(scorecard_pref, "score_preset"),
            },
        ),
        check_item(
            "serving_audit_passed",
            bool(get(serving, "summary.pass"))
            and bool(get(serving, "checks.runtime_is_force_aware"))
            and bool(get(serving, "checks.adapter_policy_ok"))
            and bool(get(serving, "checks.not_reranking")),
            {
                "pass": get(serving, "summary.pass"),
                "checks": get(serving, "checks", {}),
                "score_delta_mean": get(serving, "summary.score_delta.mean"),
                "normalized_action_delta_mean": get(serving, "summary.normalized_action_delta.mean"),
            },
        ),
        check_item(
            "scorecard_points_to_serving_audit",
            str(args.serving_audit) == str(serving_path_from_scorecard),
            {"scorecard_path": serving_path_from_scorecard, "expected": str(args.serving_audit)},
        ),
        check_item(
            "rollout_and_scorecard_point_to_weight_sweep",
            str(args.weight_sweep) == str(sweep_path_from_rollout) == str(sweep_path_from_scorecard),
            {
                "rollout_selected_by": sweep_path_from_rollout,
                "scorecard_sweep_path": sweep_path_from_scorecard,
                "expected": str(args.weight_sweep),
            },
        ),
        check_item(
            "goal_not_marked_complete_without_real_rollouts",
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
        "task": "board",
        "arm": args.arm,
        "expected": {
            "score_preset": EXPECTED_PRESET,
            "weights": EXPECTED_WEIGHTS,
        },
        "checks": checks,
        "summary": {
            "score_delta_mean": get(serving, "summary.score_delta.mean"),
            "normalized_action_delta_mean": get(serving, "summary.normalized_action_delta.mean"),
            "serving_num_windows": get(serving, "setup.num_windows"),
            "serving_labels": get(serving, "summary.labels", {}),
            "scorecard_goal_complete": get(scorecard, "evidence_levels.goal_complete"),
            "real_paired_rollout_complete": get(scorecard, "evidence_levels.real_paired_rollout_complete"),
        },
        "inputs": {
            "rollout_config": file_info(args.rollout_config),
            "weight_sweep": file_info(args.weight_sweep),
            "serving_audit": file_info(args.serving_audit),
            "scorecard": file_info(args.scorecard),
        },
        "evidence_boundary": (
            "This is a consistency/preflight audit for offline and serving artifacts. "
            "It proves that the force-aware board research arm is configured with "
            "the selected margin_only score and that the latest serving-window audit "
            "loaded the same weights. It does not prove real robot improvement."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--weight_sweep", type=Path, default=DEFAULT_WEIGHT_SWEEP)
    parser.add_argument("--serving_audit", type=Path, default=DEFAULT_SERVING_AUDIT)
    parser.add_argument("--scorecard", type=Path, default=DEFAULT_SCORECARD)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--arm", default="force_aware_guided")
    args = parser.parse_args()

    result = run(args)
    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=False)
    json_path = out_dir / "force_aware_config_consistency.json"
    md_path = out_dir / "force_aware_config_consistency.md"
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
