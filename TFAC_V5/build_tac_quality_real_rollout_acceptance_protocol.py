"""Build the TacQuality real-rollout acceptance protocol.

Offline scorer quality is not enough to finish the objective.  This protocol
turns the remaining real-rollout blockers into explicit pass/fail criteria:
what to collect, which evaluator to run, which metrics must improve, and which
artifacts prove completion.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_real_rollout_acceptance_protocol")

PATHS = {
    "experiment_packet": Path(
        "/home/chenshuai/Project/output/real_rollout_experiment_packet/"
        "formal_paired12/real_rollout_experiment_packet.json"
    ),
    "insertion_sample_plan": Path(
        "/home/chenshuai/Project/output/real_rollout_sample_size_plan/"
        "insertion_default_plan/real_rollout_sample_size_plan.json"
    ),
    "board_sample_plan": Path(
        "/home/chenshuai/Project/output/real_rollout_sample_size_plan/"
        "board_default_plan/real_rollout_sample_size_plan.json"
    ),
    "label_registry": Path(
        "/home/chenshuai/Project/output/tac_quality_label_standard_registry/"
        "tac_quality_label_standard_registry.json"
    ),
    "decision_matrix": Path(
        "/home/chenshuai/Project/output/tac_quality_scorer_decision_matrix/"
        "tac_quality_scorer_decision_matrix.json"
    ),
    "formal_launch_sheet": Path(
        "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
        "formal_paired12/tac_quality_formal_launch_sheet.json"
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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_info(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
    }


def task_protocol(task: str, packet: Dict[str, Any], plan: Dict[str, Any], registry: Dict[str, Any]) -> Dict[str, Any]:
    task_packet = get(packet, f"tasks.{task}", {}) or {}
    constraints = get(plan, "gate_aligned_constraints", {}) or {}
    rec = get(plan, "recommendation", {}) or {}
    task_standard = get(registry, f"tasks.{task}", {}) or {}
    return {
        "task": task,
        "label_standard": {
            "binary_policy": task_standard.get("binary_policy"),
            "deployment_target": task_standard.get("deployment_target"),
            "board_target_force": task_standard.get("target_force") if task == "board" else None,
            "board_force_sigma": task_standard.get("force_sigma") if task == "board" else None,
        },
        "collection": {
            "paired_design_required": bool(rec.get("paired_design_recommended", True)),
            "paired_n_pairs": int(rec.get("paired_n_pairs", task_packet.get("paired_n_pairs", 0))),
            "unpaired_n_per_group": int(rec.get("unpaired_n_per_group", task_packet.get("unpaired_n_per_group", 0))),
            "formal_arms": ["baseline", "default_guided", "distilled_guided"],
            "optional_arms": ["action_aware_guided"],
            "pairing_template": task_packet.get("pairing_template"),
            "three_arm_pairing_template": task_packet.get("three_arm_pairing_template"),
            "metadata_template": task_packet.get("metadata_template"),
            "collection_rule": (
                "Use paired trials with matched initial condition/task setup inside each pair. "
                "If pairing is impossible, collect the larger unpaired_n_per_group count and do not claim paired evidence."
            ),
        },
        "two_arm_gate": {
            "purpose": "baseline DP vs task-default TacQuality-guided DP sanity/production validation",
            "command": task_packet.get("gate_command"),
            "artifact": (
                "/home/chenshuai/Project/output/real_rollout_quality_gate/"
                f"{task}_baseline_vs_guided/real_rollout_quality_gate.json"
            ),
            "pass_conditions": {
                "min_episodes_each_arm": constraints.get("min_episodes", 10),
                "quality_delta_mean_at_least": constraints.get("min_quality_delta", 0.03),
                "bootstrap_ci95_low_must_be_positive": bool(
                    constraints.get("bootstrap_ci_lower_must_be_positive", True)
                ),
                "max_bad_rate_increase": constraints.get("max_bad_rate_increase", 0.05),
                "max_success_rate_drop": constraints.get("max_success_rate_drop", 0.0),
                "debug_or_underpowered_must_be_false": True,
            },
            "non_degradation_metrics": (
                ["risk_score", "risk_flag", "success_attr", "stopped_early_attr"]
                if task == "insertion"
                else ["too_heavy_flag", "rough_flag", "success_attr", "stopped_early_attr"]
            ),
        },
        "three_arm_ablation_gate": {
            "purpose": "select best real guided scorer among task-default and DistilledTacQualityEnergy",
            "command": task_packet.get("ablation_gate_command"),
            "artifact": (
                "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
                f"{task}_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
            ),
            "pass_conditions": {
                "production_ablation_pass": True,
                "debug_or_underpowered_must_be_false": True,
                "recommended_real_scorer_must_be_non_null": True,
                "at_least_one_guided_arm_passes_vs_baseline": True,
                "guided_vs_guided_ci_selects_winner_or_reports_tie": True,
            },
        },
    }


def build(paths: Dict[str, Path]) -> Dict[str, Any]:
    packet = load_json(paths["experiment_packet"]) or {}
    insertion_plan = load_json(paths["insertion_sample_plan"]) or {}
    board_plan = load_json(paths["board_sample_plan"]) or {}
    registry = load_json(paths["label_registry"]) or {}
    matrix = load_json(paths["decision_matrix"]) or {}
    launch = load_json(paths["formal_launch_sheet"]) or {}
    tasks = {
        "insertion": task_protocol("insertion", packet, insertion_plan, registry),
        "board": task_protocol("board", packet, board_plan, registry),
    }
    optional_action_aware = {
        "status": "optional_fourth_arm_candidate",
        "why_optional": get(
            matrix,
            "recommendation.reason",
            "ActionAware remains optional until real rollout and cross-task evidence are stronger.",
        ),
        "collection_dirs": {
            task: get(launch, f"tasks.{task}.rollout_dirs.action_aware_guided")
            for task in ["insertion", "board"]
        },
        "gate_command": (
            "python TFAC_V5/run_optional_action_aware_rollout_gate.py "
            "--packet /home/chenshuai/Project/output/real_rollout_experiment_packet/"
            "formal_paired12/real_rollout_experiment_packet.json "
            "--action_aware_pairing_dir /home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12 "
            "--run_gates"
        ),
        "acceptance": {
            "formal_gate_dependency": False,
            "must_use_quality_mode_line_search_accept_only": True,
            "compare_against_baseline_only_after_formal_three_arm_gate_is_reviewed": True,
        },
    }
    blockers_to_close = [
        {
            "name": "insertion_two_arm_real_rollout_gate",
            "artifact": tasks["insertion"]["two_arm_gate"]["artifact"],
        },
        {
            "name": "board_two_arm_real_rollout_gate",
            "artifact": tasks["board"]["two_arm_gate"]["artifact"],
        },
        {
            "name": "insertion_three_arm_real_rollout_ablation",
            "artifact": tasks["insertion"]["three_arm_ablation_gate"]["artifact"],
        },
        {
            "name": "board_three_arm_real_rollout_ablation",
            "artifact": tasks["board"]["three_arm_ablation_gate"]["artifact"],
        },
    ]
    protocol = {
        "name": "TacQuality real-rollout acceptance protocol",
        "purpose": (
            "Formal pass/fail protocol for completing TacQuality DP classifier-guidance validation "
            "on socket insertion and board wiping."
        ),
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "tasks": tasks,
        "optional_action_aware": optional_action_aware,
        "completion_blockers_to_close": blockers_to_close,
        "cannot_count_as_completion": [
            "synthetic HDF5 smoke outputs",
            "frame-level random cross validation",
            "score-only improvement without non-degradation checks",
            "server launch smoke without recorded rollout HDF5s",
            "optional ActionAware pass without formal baseline/default/distilled gates",
        ],
        "review_sequence": [
            "Collect formal HDF5 rollouts for baseline/default_guided/distilled_guided.",
            "Run TFAC_V5/run_tac_quality_post_collection_pipeline.py.",
            "If preflight is ready, rerun with --run_gates.",
            "Review two-arm gates for baseline-vs-default sanity.",
            "Review three-arm ablation gates and recommended_real_scorer.",
            "Optionally run ActionAware gate after formal gates are understood.",
            "Rerun TFAC_V5/audit_tac_quality_goal_completion.py.",
        ],
        "protocol_pass": bool(
            all(paths[name].exists() for name in ["experiment_packet", "insertion_sample_plan", "board_sample_plan", "label_registry"])
            and tasks["insertion"]["collection"]["paired_n_pairs"] >= 10
            and tasks["board"]["collection"]["paired_n_pairs"] >= 10
            and tasks["insertion"]["two_arm_gate"]["command"]
            and tasks["board"]["three_arm_ablation_gate"]["command"]
        ),
        "artifacts": {name: file_info(path) for name, path in paths.items()},
    }
    return protocol


def write_markdown(protocol: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Real-Rollout Acceptance Protocol",
        "",
        f"- protocol_pass: `{protocol['protocol_pass']}`",
        f"- scientific_evidence: `{protocol['scientific_evidence']}`",
        "",
        "## Tasks",
        "",
    ]
    for task, row in protocol["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- paired_n_pairs: `{row['collection']['paired_n_pairs']}`",
                f"- unpaired_n_per_group: `{row['collection']['unpaired_n_per_group']}`",
                f"- deployment_target: {row['label_standard']['deployment_target']}",
                "",
                "Two-arm gate:",
                "",
                "```bash",
                row["two_arm_gate"]["command"] or "",
                "```",
                "",
                "Three-arm ablation gate:",
                "",
                "```bash",
                row["three_arm_ablation_gate"]["command"] or "",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Cannot Count As Completion",
            "",
        ]
    )
    for item in protocol["cannot_count_as_completion"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Review Sequence", ""])
    for i, item in enumerate(protocol["review_sequence"], start=1):
        lines.append(f"{i}. {item}")
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
    protocol = build(PATHS)
    json_path = out_dir / "tac_quality_real_rollout_acceptance_protocol.json"
    md_path = out_dir / "tac_quality_real_rollout_acceptance_protocol.md"
    json_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(protocol, md_path)
    print(
        json.dumps(
            {
                "protocol_pass": protocol["protocol_pass"],
                "scientific_evidence": protocol["scientific_evidence"],
                "paired_n_pairs": {
                    task: row["collection"]["paired_n_pairs"]
                    for task, row in protocol["tasks"].items()
                },
                "blockers_to_close": protocol["completion_blockers_to_close"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
