"""Audit completion of the TacQualityEnergy guidance objective.

This is stricter than the offline production gate.  The gate answers:
"is the current scorer/guidance stack ready for robot dry-run?"  This audit
answers the user-level objective:

  design, evaluate, and record a tactile quality classifier/scorer for DP
  classifier guidance on socket insertion and board wiping, with evidence that
  it works and is innovative enough for the intended gradient-guidance use.

The audit deliberately keeps the objective incomplete until real baseline-vs-
guided production/robot rollouts exist for both tasks.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_goal_audit")

PATHS = {
    "insertion_eval": Path("/home/chenshuai/Project/output/ptg_quality_eval/tactile_quality_model_eval.json"),
    "board_scheme_eval": Path("/home/chenshuai/Project/output/board_quality_label_schemes/w32_s16/board_quality_scheme_eval.json"),
    "ptg_proxy_eval": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_eval.json"),
    "score_calibration": Path("/home/chenshuai/Project/output/tac_quality_score_calibration/tac_quality_score_calibration.json"),
    "scale_sweep": Path("/home/chenshuai/Project/output/tac_quality_guidance_scale_sweep/tac_quality_guidance_scale_sweep.json"),
    "robustness": Path("/home/chenshuai/Project/output/tac_quality_guidance_robustness/tac_quality_guidance_robustness.json"),
    "offline_gate": Path("/home/chenshuai/Project/output/ptg_offline_production_gate/ptg_offline_production_gate.json"),
    "manifest": Path("/home/chenshuai/Project/output/tac_quality_guidance_manifest/tac_quality_guidance_manifest.json"),
    "evidence_summary": Path("/home/chenshuai/Project/output/ptg_guidance_evidence/ptg_guidance_evidence_summary.json"),
    "real_rollout_insertion": Path("/home/chenshuai/Project/output/real_rollout_quality_gate/insertion_baseline_vs_guided/real_rollout_quality_gate.json"),
    "real_rollout_board": Path("/home/chenshuai/Project/output/real_rollout_quality_gate/board_baseline_vs_guided/real_rollout_quality_gate.json"),
    "real_rollout_prep_smoke": Path("/home/chenshuai/Project/output/real_rollout_validation_ready/board_smoke_ready_with_csv/real_rollout_validation_readiness.json"),
    "record": Path("/home/chenshuai/Project/TactileACT-cs/工作记录codex.txt"),
    "eval_doc": Path("/home/chenshuai/Project/TactileACT-cs/research/PTG_触觉质量分类器评估方案与实验记录.md"),
    "deploy_doc": Path("/home/chenshuai/Project/TactileACT-cs/research/PTG_TacQualityEnergy_部署策略与运行手册.md"),
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


def file_contains(path: Path, snippets: List[str]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    return all(s in text for s in snippets)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def item(requirement: str, status: str, evidence: str, artifact: str) -> Dict[str, Any]:
    if status not in {"satisfied", "incomplete", "weak", "missing", "contradicted"}:
        raise ValueError(f"Bad audit status: {status}")
    return {
        "requirement": requirement,
        "status": status,
        "passed": status == "satisfied",
        "evidence": evidence,
        "artifact": artifact,
    }


def real_rollout_status(d: Optional[Dict[str, Any]], task: str) -> Dict[str, Any]:
    if d is None:
        return {
            "status": "missing",
            "evidence": f"No formal {task} baseline-vs-guided rollout gate report found.",
        }
    decision = get(d, "decision.production_validation_pass", False)
    debug = bool(get(d, "debug_or_underpowered", True))
    if decision and not debug:
        return {"status": "satisfied", "evidence": "production_validation_pass=true and debug_or_underpowered=false"}
    return {
        "status": "incomplete",
        "evidence": (
            f"production_validation_pass={decision}, "
            f"debug_or_underpowered={debug}, "
            f"decision={get(d, 'decision')}"
        ),
    }


def build_audit(paths: Dict[str, Path]) -> Dict[str, Any]:
    data = {name: load_json(path) for name, path in paths.items() if path.suffix == ".json"}

    insertion = data["insertion_eval"]
    board = data["board_scheme_eval"]
    ptg = data["ptg_proxy_eval"]
    scale = data["scale_sweep"]
    robust = data["robustness"]
    offline = data["offline_gate"]
    manifest = data["manifest"]
    summary = data["evidence_summary"]
    rr_ins = data["real_rollout_insertion"]
    rr_board = data["real_rollout_board"]
    prep_smoke = data["real_rollout_prep_smoke"]

    requirements = [
        item(
            "Socket insertion scorer is evaluated with episode-level generalization.",
            "satisfied"
            if (get(insertion, "group_cv.LDA.balanced_accuracy.mean", 0.0) or 0.0) >= 0.88
            and (get(insertion, "data.n_groups", 0) or 0) >= 100
            else "incomplete",
            "GroupKFold best LDA balanced_acc="
            f"{get(insertion, 'group_cv.LDA.balanced_accuracy.mean')}; "
            f"n_groups={get(insertion, 'data.n_groups')}",
            str(paths["insertion_eval"]),
        ),
        item(
            "Board wiping quality labels and scorer target are defined from force magnitude and smoothness.",
            "satisfied"
            if get(board, "best_classification.balanced_accuracy", 0.0) >= 0.88
            and get(board, "best_regression.quality_corr", 0.0) >= 0.95
            else "incomplete",
            "best_classification balanced_acc="
            f"{get(board, 'best_classification.balanced_accuracy')}; "
            f"best_regression quality_corr={get(board, 'best_regression.quality_corr')}; "
            f"scheme={get(board, 'best_classification.scheme')}",
            str(paths["board_scheme_eval"]),
        ),
        item(
            "Unified task-conditioned differentiable scorer is trained/evaluated across insertion and board.",
            "satisfied"
            if (get(ptg, "mixed_group_cv.binary_auc.mean", 0.0) or 0.0) >= 0.95
            and (get(ptg, "mixed_group_cv.quality_corr.mean", 0.0) or 0.0) >= 0.70
            and bool(get(ptg, "gradient_sanity.usable_for_feature_guidance", False))
            else "incomplete",
            "binary_auc="
            f"{get(ptg, 'mixed_group_cv.binary_auc.mean')}; "
            f"quality_corr={get(ptg, 'mixed_group_cv.quality_corr.mean')}; "
            f"gradient_sanity={get(ptg, 'gradient_sanity')}",
            str(paths["ptg_proxy_eval"]),
        ),
        item(
            "Scorer behaves as a local action-gradient energy on both tasks.",
            "satisfied"
            if bool(get(scale, "overall_pass", False))
            and bool(get(robust, "overall_pass", False))
            else "incomplete",
            "scale_sweep overall="
            f"{get(scale, 'overall_pass')}, insertion_improved={get(scale, 'insertion.recommended_improved_rate')}, "
            f"board_improved={get(scale, 'board.recommended_improved_rate')}; "
            "robustness overall="
            f"{get(robust, 'overall_pass')}, insertion_worst={get(robust, 'insertion.worst_perturbed_gradient_improved_rate')}, "
            f"board_worst={get(robust, 'board.worst_perturbed_gradient_improved_rate')}",
            f"{paths['scale_sweep']} ; {paths['robustness']}",
        ),
        item(
            "Offline production-readiness gate passes while preserving the real-robot validation gap.",
            "satisfied"
            if bool(get(offline, "offline_production_gate_pass", False))
            and get(offline, "remaining_required_step") == "Real robot / final production policy validation."
            else "incomplete",
            "offline_gate="
            f"{get(offline, 'offline_production_gate_pass')}; "
            f"remaining={get(offline, 'remaining_required_step')}",
            str(paths["offline_gate"]),
        ),
        item(
            "Deployment manifest exists and records the offline-ready but not robot-validated package.",
            "satisfied"
            if bool(get(manifest, "deployment_manifest_pass", False))
            and get(manifest, "deployment_policy.completion_status") == "offline_ready_not_robot_validated"
            else "incomplete",
            "manifest_pass="
            f"{get(manifest, 'deployment_manifest_pass')}; "
            f"completion_status={get(manifest, 'deployment_policy.completion_status')}; "
            f"manifest_git_commit={get(manifest, 'git_commit')}",
            str(paths["manifest"]),
        ),
        item(
            "Evidence summary explicitly keeps the objective incomplete until real rollout validation.",
            "satisfied"
            if get(summary, "completion_assessment.objective_complete", get(summary, "objective_complete")) is False
            and "real-robot" in str(get(summary, "completion_assessment.reason", get(summary, "reason", "")))
            else "incomplete",
            "objective_complete="
            f"{get(summary, 'completion_assessment.objective_complete', get(summary, 'objective_complete'))}; "
            f"reason={get(summary, 'completion_assessment.reason', get(summary, 'reason'))}",
            str(paths["evidence_summary"]),
        ),
        item(
            "Work and research records document design, standards, experiments, and deployment policy.",
            "satisfied"
            if file_contains(paths["record"], ["TacQualityEnergy", "GroupKFold", "offline production gate"])
            and file_contains(paths["eval_doc"], ["TacQualityEnergy", "GroupKFold", "可梯度引导"])
            and file_contains(paths["deploy_doc"], ["TacQualityEnergy", "offline production gate", "Real robot"])
            else "incomplete",
            "Required snippets found in 工作记录codex.txt, PTG evaluation doc, and deployment manual.",
            f"{paths['record']} ; {paths['eval_doc']} ; {paths['deploy_doc']}",
        ),
        item(
            "Formal rollout validation preparation tool exists and produces gate-ready templates/commands.",
            "satisfied"
            if prep_smoke is not None
            and get(prep_smoke, "task") == "board"
            and get(prep_smoke, "ready_for_quality_gate") is True
            and get(prep_smoke, "outputs.pairing_csv_template") is not None
            and get(prep_smoke, "outputs.metadata_csv_template") is not None
            and "eval_real_rollout_quality_gate.py" in str(get(prep_smoke, "gate_command", ""))
            else "incomplete",
            "prep_smoke="
            f"task={get(prep_smoke, 'task')}, "
            f"ready={get(prep_smoke, 'ready_for_quality_gate')}, "
            f"baseline_n={get(prep_smoke, 'baseline.n')}, guided_n={get(prep_smoke, 'guided.n')}, "
            f"pairing_ready={get(prep_smoke, 'pairing_csv_check.ready')}, "
            f"metadata_ready={get(prep_smoke, 'metadata_csv_check.ready')}, "
            f"pairing={get(prep_smoke, 'outputs.pairing_csv_template')}, "
            f"metadata={get(prep_smoke, 'outputs.metadata_csv_template')}",
            str(paths["real_rollout_prep_smoke"]),
        ),
    ]

    ins_rr = real_rollout_status(rr_ins, "insertion")
    board_rr = real_rollout_status(rr_board, "board")
    requirements.extend(
        [
            item(
                "Formal socket insertion baseline-vs-guided production/robot rollout validation passes.",
                ins_rr["status"],
                ins_rr["evidence"],
                str(paths["real_rollout_insertion"]),
            ),
            item(
                "Formal board wiping baseline-vs-guided production/robot rollout validation passes.",
                board_rr["status"],
                board_rr["evidence"],
                str(paths["real_rollout_board"]),
            ),
        ]
    )

    objective_complete = all(r["status"] == "satisfied" for r in requirements)
    blockers = [r for r in requirements if r["status"] != "satisfied"]
    result = {
        "objective": (
            "Design, evaluate, and record a tactile quality classifier/scorer for DP classifier guidance "
            "on socket insertion and board wiping, balancing effectiveness and novelty."
        ),
        "git_commit": git_commit(),
        "objective_complete": bool(objective_complete),
        "status": "complete" if objective_complete else "incomplete",
        "requirements": requirements,
        "blockers": blockers,
        "next_required_step": (
            "Collect formal baseline-vs-guided production/robot rollouts for insertion and board, "
            "then run TFAC_V5/eval_real_rollout_quality_gate.py with metadata/pairing if available."
            if blockers
            else None
        ),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Goal Completion Audit",
        "",
        f"- objective_complete: `{result['objective_complete']}`",
        f"- status: `{result['status']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "## Requirements",
        "",
        "| status | requirement | evidence |",
        "|---|---|---|",
    ]
    for row in result["requirements"]:
        lines.append(
            f"| {row['status']} | {row['requirement']} | {str(row['evidence']).replace('|', '/')} |"
        )
    lines.extend(["", "## Blockers", ""])
    for row in result["blockers"]:
        lines.append(f"- **{row['status']}** {row['requirement']}: {row['evidence']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_audit(PATHS)
    json_path = out_dir / "tac_quality_goal_completion_audit.json"
    md_path = out_dir / "tac_quality_goal_completion_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "objective_complete": result["objective_complete"],
                "status": result["status"],
                "n_requirements": len(result["requirements"]),
                "n_blockers": len(result["blockers"]),
                "next_required_step": result["next_required_step"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Saved {json_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
