#!/usr/bin/env python3
"""Audit current TacQuality scorer and gradient-guidance state.

The goal is to keep one machine-checkable answer to:

1. Which scorer is the current candidate for insertion and board?
2. Is it evaluated with a leakage-safe protocol?
3. Does it produce differentiable Foresight -> score -> action guidance?
4. Is the current server path using gradient guidance, not reranking?
5. Is real robot improvement proven yet?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE_AUDIT = Path(
    "/home/chenshuai/Project/output/tac_quality_evidence_audit_20260618/tac_quality_evidence_audit.json"
)
DEFAULT_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json"
)
DEFAULT_REAL_ROLLOUT = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_eval/current_tac_quality_pre_rollout_20260619/tac_quality_real_rollout_eval.json"
)
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_state_audit")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def get(d: dict[str, Any] | None, dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def num(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def pass_item(value: bool, detail: str) -> dict[str, Any]:
    return {"pass": bool(value), "detail": detail}


def ckpt_from_arm(rollout_config: dict[str, Any] | None, task: str, arm: str) -> str | None:
    return get(rollout_config, f"tasks.{task}.{arm}.checkpoint.path")


def runtime_from_arm(rollout_config: dict[str, Any] | None, task: str, arm: str) -> str | None:
    return get(rollout_config, f"tasks.{task}.{arm}.scorer_runtime")


def audit_insertion(evidence: dict[str, Any] | None, rollout_config: dict[str, Any] | None) -> dict[str, Any]:
    task = get(evidence, "tasks.insertion", {})
    arm = "default_guided"
    ckpt = ckpt_from_arm(rollout_config, "insertion", arm)
    grouped = task.get("grouped_cv", {}) if isinstance(task, dict) else {}
    gradient = task.get("foresight_gradient_audit", {}) if isinstance(task, dict) else {}
    dry_run = task.get("server_dry_run", {}) if isinstance(task, dict) else {}
    checks = {
        "recommended_arm_exists": pass_item(
            get(rollout_config, f"tasks.insertion.{arm}") is not None,
            f"arm={arm}",
        ),
        "runtime_matches": pass_item(
            runtime_from_arm(rollout_config, "insertion", arm) == "InsertionRiskScorerRuntime",
            str(runtime_from_arm(rollout_config, "insertion", arm)),
        ),
        "checkpoint_exists": pass_item(exists(ckpt), str(ckpt)),
        "grouped_cv_auc": pass_item(
            num(grouped.get("binary_auc_mean")) >= 0.95,
            f"AUC={grouped.get('binary_auc_mean')}",
        ),
        "grouped_cv_balanced_accuracy": pass_item(
            num(grouped.get("binary_balanced_accuracy_mean")) >= 0.90,
            f"bACC={grouped.get('binary_balanced_accuracy_mean')}",
        ),
        "quality_corr": pass_item(
            num(grouped.get("quality_corr_mean")) >= 0.50,
            f"corr={grouped.get('quality_corr_mean')}",
        ),
        "foresight_gradient": pass_item(
            bool(gradient.get("pass"))
            and num(gradient.get("finite_grad_rate_mean")) >= 0.999
            and num(gradient.get("positive_grad_rate_mean")) >= 0.999
            and num(gradient.get("improved_rate_mean")) >= 0.95
            and num(gradient.get("trust_region_pass_rate")) >= 0.999,
            json.dumps(gradient, ensure_ascii=False),
        ),
        "server_dry_run": pass_item(
            bool(dry_run.get("pass")) and bool(dry_run.get("not_reranking")),
            json.dumps(dry_run, ensure_ascii=False),
        ),
    }
    return {
        "task": "insertion",
        "recommended_arm": arm,
        "scorer": "InsertionRiskScorerRuntime",
        "checkpoint": ckpt,
        "checks": checks,
        "ready_for_real_rollout": all(item["pass"] for item in checks.values()),
    }


def audit_board(evidence: dict[str, Any] | None, rollout_config: dict[str, Any] | None) -> dict[str, Any]:
    task = get(evidence, "tasks.board", {})
    arm = get(rollout_config, "recommended_board_arm", "marker_joint_guided")
    ckpt = ckpt_from_arm(rollout_config, "board", str(arm))
    grouped = task.get("grouped_heldout", {}) if isinstance(task, dict) else {}
    alignment = task.get("foresight_alignment", {}) if isinstance(task, dict) else {}
    gradient = task.get("foresight_gradient_audit", {}) if isinstance(task, dict) else {}
    dry_run = task.get("server_dry_run", {}) if isinstance(task, dict) else {}
    energy_source = get(rollout_config, f"tasks.board.{arm}.refiner.energy.source")
    checks = {
        "recommended_arm_exists": pass_item(
            get(rollout_config, f"tasks.board.{arm}") is not None,
            f"arm={arm}",
        ),
        "runtime_matches": pass_item(
            runtime_from_arm(rollout_config, "board", str(arm)) == "ForceBandTacQualityEnergyRuntime",
            str(runtime_from_arm(rollout_config, "board", str(arm))),
        ),
        "marker_joint_source": pass_item(
            "marker_joint_action" in str(energy_source),
            str(energy_source),
        ),
        "checkpoint_exists": pass_item(exists(ckpt), str(ckpt)),
        "heldout_auc": pass_item(
            num(grouped.get("binary_auc")) >= 0.95,
            f"AUC={grouped.get('binary_auc')}",
        ),
        "heldout_balanced_accuracy": pass_item(
            num(grouped.get("binary_balanced_accuracy")) >= 0.90,
            f"bACC={grouped.get('binary_balanced_accuracy')}",
        ),
        "quality_spearman": pass_item(
            num(grouped.get("quality_spearman")) >= 0.70,
            f"spearman={grouped.get('quality_spearman')}",
        ),
        "foresight_alignment": pass_item(
            num(alignment.get("pred_auc_good")) >= 0.90
            and num(alignment.get("pred_gt_spearman")) >= 0.40
            and num(alignment.get("pred_score_force_band_quality_spearman")) >= 0.30,
            json.dumps(alignment, ensure_ascii=False),
        ),
        "foresight_gradient": pass_item(
            bool(gradient.get("pass"))
            and num(gradient.get("finite_grad_rate_mean")) >= 0.999
            and num(gradient.get("positive_grad_rate_mean")) >= 0.999
            and num(gradient.get("improved_rate_mean")) >= 0.95
            and num(gradient.get("trust_region_pass_rate")) >= 0.999,
            json.dumps(gradient, ensure_ascii=False),
        ),
        "server_dry_run": pass_item(
            bool(dry_run.get("pass")) and bool(dry_run.get("not_reranking")),
            json.dumps(dry_run, ensure_ascii=False),
        ),
    }
    return {
        "task": "board",
        "recommended_arm": arm,
        "scorer": "ForceBandTacQualityEnergyRuntime(marker_joint_action)",
        "checkpoint": ckpt,
        "checks": checks,
        "ready_for_real_rollout": all(item["pass"] for item in checks.values()),
    }


def audit_real_rollout(real_rollout: dict[str, Any] | None) -> dict[str, Any]:
    if real_rollout is None:
        return {
            "exists": False,
            "real_rollout_evidence_complete": False,
            "detail": "real rollout summary JSON missing",
        }
    return {
        "exists": True,
        "summary_json": real_rollout.get("summary_json"),
        "summary_md": real_rollout.get("summary_md"),
        "board_real_comparison_ready": bool(get(real_rollout, "board.real_comparison_ready")),
        "insertion_real_comparison_ready": bool(get(real_rollout, "insertion.real_comparison_ready")),
        "real_rollout_evidence_complete": bool(real_rollout.get("real_rollout_evidence_complete")),
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Guidance State Audit",
        "",
        "This audit checks the current scorer/guidance state against the DP classifier guidance goal.",
        "",
        "## Summary",
        "",
        f"- insertion_ready_for_real_rollout: `{result['insertion']['ready_for_real_rollout']}`",
        f"- board_ready_for_real_rollout: `{result['board']['ready_for_real_rollout']}`",
        f"- real_rollout_evidence_complete: `{result['real_rollout']['real_rollout_evidence_complete']}`",
        f"- overall_goal_complete: `{result['overall_goal_complete']}`",
        "",
        "## Recommended Scorers",
        "",
        "| task | arm | scorer | checkpoint | ready |",
        "|---|---|---|---|---|",
    ]
    for task_name in ["insertion", "board"]:
        task = result[task_name]
        lines.append(
            f"| {task_name} | `{task['recommended_arm']}` | `{task['scorer']}` | "
            f"`{task['checkpoint']}` | `{task['ready_for_real_rollout']}` |"
        )
    for task_name in ["insertion", "board"]:
        lines.extend(["", f"## {task_name.title()} Checks", "", "| check | pass | detail |", "|---|---|---|"])
        for name, item in result[task_name]["checks"].items():
            detail = str(item["detail"]).replace("\n", " ")[:500]
            lines.append(f"| `{name}` | `{item['pass']}` | {detail} |")
    lines.extend([
        "",
        "## Real Rollout Evidence",
        "",
        f"- summary_json: `{result['real_rollout'].get('summary_json')}`",
        f"- summary_md: `{result['real_rollout'].get('summary_md')}`",
        f"- board_real_comparison_ready: `{result['real_rollout'].get('board_real_comparison_ready')}`",
        f"- insertion_real_comparison_ready: `{result['real_rollout'].get('insertion_real_comparison_ready')}`",
        "",
        "## Interpretation",
        "",
        "- If insertion/board readiness is true, the current scorer is suitable to test as DP gradient guidance.",
        "- If real rollout evidence is false, the final performance claim is still missing.",
        "- Dry-run, offline CV, and Foresight gradient audits are readiness evidence only.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence_audit", default=str(DEFAULT_EVIDENCE_AUDIT))
    parser.add_argument("--rollout_config", default=str(DEFAULT_ROLLOUT_CONFIG))
    parser.add_argument("--real_rollout_summary", default=str(DEFAULT_REAL_ROLLOUT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = load_json(Path(args.evidence_audit))
    rollout_config = load_json(Path(args.rollout_config))
    real_rollout = load_json(Path(args.real_rollout_summary))

    result = {
        "inputs": {
            "evidence_audit": args.evidence_audit,
            "rollout_config": args.rollout_config,
            "real_rollout_summary": args.real_rollout_summary,
        },
        "insertion": audit_insertion(evidence, rollout_config),
        "board": audit_board(evidence, rollout_config),
        "real_rollout": audit_real_rollout(real_rollout),
    }
    result["overall_goal_complete"] = bool(
        result["insertion"]["ready_for_real_rollout"]
        and result["board"]["ready_for_real_rollout"]
        and result["real_rollout"]["real_rollout_evidence_complete"]
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_guidance_state_audit.json"
    md_path = out_dir / "tac_quality_guidance_state_audit.md"
    result["outputs"] = {"json": str(json_path), "markdown": str(md_path)}
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "insertion_ready": result["insertion"]["ready_for_real_rollout"],
        "board_ready": result["board"]["ready_for_real_rollout"],
        "real_rollout_evidence_complete": result["real_rollout"]["real_rollout_evidence_complete"],
        "overall_goal_complete": result["overall_goal_complete"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
