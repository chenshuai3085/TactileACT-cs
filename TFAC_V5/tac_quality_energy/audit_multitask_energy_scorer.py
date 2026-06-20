"""Audit the current multitask TacQualityEnergy scorer evidence.

This script is intentionally read-only with respect to model checkpoints.  It
collects the existing JSON artifacts for the insertion + board manual-board
TacQualityEnergy candidate and writes a compact pass/fail audit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


DEFAULT_DESIGN = Path("/home/chenshuai/Project/output/tac_quality_scorer_design_comparison/scorer_design_comparison.json")
DEFAULT_EVAL = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy/distilled_tac_quality_energy_eval.json")
DEFAULT_RUNTIME = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy/runtime_sanity.json")
DEFAULT_SWEEP = Path(
    "/home/chenshuai/Project/output/manual_board_tac_quality_energy_action_gradient_smoke/"
    "manual_board_energy_action_gradient_sweep_summary.json"
)
DEFAULT_BRIDGE = Path(
    "/home/chenshuai/Project/output/manual_board_tac_quality_energy_foresight_bridge_smoke/"
    "manual_board_energy_foresight_bridge_smoke.json"
)
DEFAULT_REAL_GATES = {
    "insertion": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "insertion_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
    "board": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "board_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
}
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy/audit")
DEFAULT_DOC = Path("docs/2026-06-20_multitask_tac_quality_energy_audit.md")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def mean_metric(block: Dict[str, Any], key: str) -> Optional[float]:
    value = block.get(key)
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def pass_metric(value: Optional[float], threshold: float, higher_is_better: bool = True) -> bool:
    if value is None:
        return False
    return value >= threshold if higher_is_better else value <= threshold


def check(name: str, passed: bool, value: Any, threshold: Any, note: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "pass": bool(passed),
        "value": value,
        "threshold": threshold,
        "note": note,
    }


def all_pass(rows: Iterable[Dict[str, Any]]) -> bool:
    return all(bool(row.get("pass")) for row in rows)


def task_eval_checks(task: str, metrics: Dict[str, Any]) -> list[Dict[str, Any]]:
    thresholds = {
        "binary_auc": 0.90,
        "energy_binary_auc": 0.90,
        "reason_macro_f1": 0.65,
        "quality_corr": 0.60,
        "energy_quality_spearman": 0.40,
    }
    rows = []
    for key, threshold in thresholds.items():
        value = mean_metric(metrics, key)
        rows.append(check(f"{task}.{key}", pass_metric(value, threshold), value, threshold))
    return rows


def summarize_real_gate(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "production_validation_pass": False,
            "note": "missing formal real rollout gate; offline scorer evidence cannot prove robot improvement",
        }
    data = load_json(path)
    return {
        "path": str(path),
        "exists": True,
        "production_validation_pass": bool(data.get("production_validation_pass", False)),
        "quality_delta_mean": data.get("comparison", {}).get("quality_delta_mean"),
        "note": data.get("interpretation") or data.get("note", ""),
    }


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    design = load_json(args.design)
    eval_result = load_json(args.eval)
    runtime = load_json(args.runtime)
    sweep = load_json(args.sweep)
    bridge = load_json(args.bridge)

    design_recommendation = design.get("recommendation", {})
    recommended_design = design_recommendation.get("recommended_main_design")

    mixed = eval_result["mixed_episode_group_cv"]
    task_breakdown = eval_result["task_breakdown_group_cv"]

    mixed_checks = [
        check("mixed.binary_auc", pass_metric(mean_metric(mixed, "binary_auc"), 0.95), mean_metric(mixed, "binary_auc"), 0.95),
        check(
            "mixed.energy_binary_auc",
            pass_metric(mean_metric(mixed, "energy_binary_auc"), 0.95),
            mean_metric(mixed, "energy_binary_auc"),
            0.95,
        ),
        check(
            "mixed.reason_macro_f1",
            pass_metric(mean_metric(mixed, "reason_macro_f1"), 0.75),
            mean_metric(mixed, "reason_macro_f1"),
            0.75,
        ),
        check("mixed.quality_corr", pass_metric(mean_metric(mixed, "quality_corr"), 0.70), mean_metric(mixed, "quality_corr"), 0.70),
        check(
            "mixed.teacher_pred_corr",
            pass_metric(mean_metric(mixed, "teacher_pred_corr"), 0.90),
            mean_metric(mixed, "teacher_pred_corr"),
            0.90,
        ),
        check(
            "mixed.energy_teacher_spearman",
            pass_metric(mean_metric(mixed, "energy_teacher_spearman"), 0.90),
            mean_metric(mixed, "energy_teacher_spearman"),
            0.90,
        ),
        check(
            "mixed.energy_quality_spearman",
            pass_metric(mean_metric(mixed, "energy_quality_spearman"), 0.65),
            mean_metric(mixed, "energy_quality_spearman"),
            0.65,
        ),
    ]

    per_task_checks: Dict[str, list[Dict[str, Any]]] = {}
    for task in ("insertion", "board"):
        per_task_checks[task] = task_eval_checks(task, task_breakdown[task])

    runtime_checks = [
        check("runtime.all_finite", bool(runtime.get("all_finite")), runtime.get("all_finite"), True),
        check("runtime.usable_for_guidance", bool(runtime.get("usable_for_guidance")), runtime.get("usable_for_guidance"), True),
        check("runtime.left_grad_norm", float(runtime.get("left_grad_norm", 0.0)) > 0.0, runtime.get("left_grad_norm"), "> 0"),
        check("runtime.joint_grad_norm", float(runtime.get("joint_grad_norm", 0.0)) > 0.0, runtime.get("joint_grad_norm"), "> 0"),
    ]

    recommended = sweep.get("recommended_deployment", {}).get("row") or sweep.get("recommended") or {}
    sweep_checks = [
        check("sweep.mode", recommended.get("mode") == "energy_clipped", recommended.get("mode"), "energy_clipped"),
        check("sweep.scale", abs(float(recommended.get("scale", -1.0)) - 0.5) < 1e-9, recommended.get("scale"), 0.5),
        check("sweep.overall_pass", bool(recommended.get("overall_pass")), recommended.get("overall_pass"), True),
        check(
            "sweep.insertion_improved_rate",
            float(recommended.get("insertion_improved_rate", 0.0)) >= 0.95,
            recommended.get("insertion_improved_rate"),
            ">= 0.95",
        ),
        check(
            "sweep.board_improved_rate",
            float(recommended.get("board_improved_rate", 0.0)) >= 0.95,
            recommended.get("board_improved_rate"),
            ">= 0.95",
        ),
        check("sweep.insertion_delta_positive", float(recommended.get("insertion_delta_mean", 0.0)) > 0, recommended.get("insertion_delta_mean"), "> 0"),
        check("sweep.board_delta_positive", float(recommended.get("board_delta_mean", 0.0)) > 0, recommended.get("board_delta_mean"), "> 0"),
    ]

    bridge_checks = [
        check("bridge.overall_pass", bool(bridge.get("overall_pass")), bridge.get("overall_pass"), True),
        check("bridge.not_reranking", bool(bridge.get("not_reranking")), bridge.get("not_reranking"), True),
        check(
            "bridge.not_every_step_ddpm_guidance",
            bool(bridge.get("not_every_step_ddpm_guidance")),
            bridge.get("not_every_step_ddpm_guidance"),
            True,
        ),
    ]
    for task in ("insertion", "board"):
        task_bridge = bridge[task]
        grad = task_bridge["bridge_score_gradient"]
        adapter = task_bridge["adapter_report"]
        bridge_checks.extend(
            [
                check(f"bridge.{task}.passes", bool(task_bridge.get("passes")), task_bridge.get("passes"), True),
                check(f"bridge.{task}.finite_grad_rate", float(grad.get("finite_grad_rate", 0.0)) >= 0.999, grad.get("finite_grad_rate"), ">= 0.999"),
                check(f"bridge.{task}.positive_grad_rate", float(grad.get("positive_grad_rate", 0.0)) >= 0.999, grad.get("positive_grad_rate"), ">= 0.999"),
                check(f"bridge.{task}.improved_rate", float(adapter.get("improved_rate", 0.0)) >= 0.95, adapter.get("improved_rate"), ">= 0.95"),
                check(
                    f"bridge.{task}.trust_region",
                    bool(adapter.get("max_delta_within_trust_region")),
                    adapter.get("max_delta_within_trust_region"),
                    True,
                ),
            ]
        )

    real_gates = {task: summarize_real_gate(path) for task, path in args.real_gate.items()}

    category_checks = {
        "design": [
            check(
                "design.recommended_main_design",
                recommended_design == "continuous_energy_with_aux_reason_heads",
                recommended_design,
                "continuous_energy_with_aux_reason_heads",
            )
        ],
        "mixed_group_cv": mixed_checks,
        "task_breakdown_group_cv": per_task_checks["insertion"] + per_task_checks["board"],
        "runtime_gradient": runtime_checks,
        "action_gradient_sweep": sweep_checks,
        "foresight_bridge_smoke": bridge_checks,
    }
    category_pass = {name: all_pass(rows) for name, rows in category_checks.items()}
    offline_guidance_ready = all(category_pass.values())
    production_validated = all(bool(row["production_validation_pass"]) for row in real_gates.values())

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Audit current multitask manual-board TacQualityEnergy evidence for DP classifier/scorer guidance.",
        "candidate": {
            "name": "manual-board / distilled TacQualityEnergy",
            "checkpoint": "/home/chenshuai/Project/output/manual_board_tac_quality_energy/distilled_tac_quality_energy_final.pt",
            "runtime": "TFAC_V5.tac_quality_energy.runtime.DistilledTacQualityEnergyRuntime",
            "score_mode": "energy_clipped",
            "action_step_scale": 0.5,
            "design": "shared encoder + binary head + reason head + quality head + RF-teacher distillation head + free energy head",
        },
        "inputs": {
            "design": str(args.design),
            "eval": str(args.eval),
            "runtime": str(args.runtime),
            "sweep": str(args.sweep),
            "bridge": str(args.bridge),
            "real_gates": {task: str(path) for task, path in args.real_gate.items()},
        },
        "category_pass": category_pass,
        "checks": category_checks,
        "key_metrics": {
            "mixed": {
                "binary_auc": mean_metric(mixed, "binary_auc"),
                "energy_binary_auc": mean_metric(mixed, "energy_binary_auc"),
                "reason_macro_f1": mean_metric(mixed, "reason_macro_f1"),
                "quality_corr": mean_metric(mixed, "quality_corr"),
                "teacher_pred_corr": mean_metric(mixed, "teacher_pred_corr"),
                "energy_teacher_spearman": mean_metric(mixed, "energy_teacher_spearman"),
                "energy_quality_spearman": mean_metric(mixed, "energy_quality_spearman"),
            },
            "tasks": {
                task: {
                    "binary_auc": mean_metric(task_breakdown[task], "binary_auc"),
                    "energy_binary_auc": mean_metric(task_breakdown[task], "energy_binary_auc"),
                    "reason_macro_f1": mean_metric(task_breakdown[task], "reason_macro_f1"),
                    "quality_corr": mean_metric(task_breakdown[task], "quality_corr"),
                    "energy_quality_spearman": mean_metric(task_breakdown[task], "energy_quality_spearman"),
                }
                for task in ("insertion", "board")
            },
            "recommended_guidance": recommended,
            "runtime_sanity": runtime,
            "bridge": {
                task: {
                    "finite_grad_rate": bridge[task]["bridge_score_gradient"].get("finite_grad_rate"),
                    "positive_grad_rate": bridge[task]["bridge_score_gradient"].get("positive_grad_rate"),
                    "improved_rate": bridge[task]["adapter_report"].get("improved_rate"),
                    "score_delta_mean": bridge[task]["adapter_report"].get("score_delta", {}).get("mean"),
                }
                for task in ("insertion", "board")
            },
        },
        "offline_guidance_ready": bool(offline_guidance_ready),
        "production_validated": bool(production_validated),
        "real_rollout_gates": real_gates,
        "conclusion": {
            "current_best_candidate": "manual-board / distilled TacQualityEnergy is the strongest unified innovation candidate before real rollout.",
            "recommended_use_now": "Use as an ablation scorer for DP/Foresight trust-region gradient guidance with score_mode=energy_clipped and action_step_scale=0.5.",
            "not_yet_proven": "No formal baseline-vs-guided real rollout gate exists for both insertion and board, so true robot improvement remains unproven.",
            "next_required_evidence": [
                "Run paired insertion baseline-vs-guided real rollout gate.",
                "Run paired board baseline-vs-guided real rollout gate.",
                "Compare task-default guided vs distilled TacQualityEnergy guided in a three-arm rollout ablation.",
            ],
        },
    }


def render_md(result: Dict[str, Any]) -> str:
    lines = [
        "# Multitask TacQualityEnergy Audit",
        "",
        f"Generated: `{result['generated_at']}`",
        "",
        "## Candidate",
        "",
        f"- name: `{result['candidate']['name']}`",
        f"- checkpoint: `{result['candidate']['checkpoint']}`",
        f"- runtime: `{result['candidate']['runtime']}`",
        f"- score mode: `{result['candidate']['score_mode']}`",
        f"- action step scale: `{result['candidate']['action_step_scale']}`",
        "",
        "## Pass Summary",
        "",
        "| category | pass |",
        "|---|---:|",
    ]
    for name, passed in result["category_pass"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(
        [
            "",
            f"- offline guidance ready: `{result['offline_guidance_ready']}`",
            f"- production validated by real rollout gates: `{result['production_validated']}`",
            "",
            "## Key Metrics",
            "",
            "| scope | binary AUC | energy AUC | reason macro-F1 | quality corr | energy-quality rho |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    mixed = result["key_metrics"]["mixed"]
    lines.append(
        "| mixed | "
        f"{mixed['binary_auc']:.4f} | {mixed['energy_binary_auc']:.4f} | "
        f"{mixed['reason_macro_f1']:.4f} | {mixed['quality_corr']:.4f} | "
        f"{mixed['energy_quality_spearman']:.4f} |"
    )
    for task, row in result["key_metrics"]["tasks"].items():
        lines.append(
            f"| {task} | {row['binary_auc']:.4f} | {row['energy_binary_auc']:.4f} | "
            f"{row['reason_macro_f1']:.4f} | {row['quality_corr']:.4f} | "
            f"{row['energy_quality_spearman']:.4f} |"
        )

    rec = result["key_metrics"]["recommended_guidance"]
    lines.extend(
        [
            "",
            "## Recommended Guidance Setting",
            "",
            f"- mode: `{rec.get('mode')}`",
            f"- action step scale: `{rec.get('scale')}`",
            f"- insertion improved rate: `{rec.get('insertion_improved_rate')}`",
            f"- board improved rate: `{rec.get('board_improved_rate')}`",
            f"- insertion score delta mean: `{rec.get('insertion_delta_mean')}`",
            f"- board score delta mean: `{rec.get('board_delta_mean')}`",
            "",
            "## Foresight-Bridge Smoke",
            "",
            "| task | finite grad | positive grad | improved rate | score delta mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for task, row in result["key_metrics"]["bridge"].items():
        lines.append(
            f"| {task} | {row['finite_grad_rate']} | {row['positive_grad_rate']} | "
            f"{row['improved_rate']} | {row['score_delta_mean']} |"
        )

    lines.extend(["", "## Real Rollout Gates", "", "| task | exists | pass | path |", "|---|---:|---:|---|"])
    for task, row in result["real_rollout_gates"].items():
        lines.append(f"| {task} | {row['exists']} | {row['production_validation_pass']} | `{row['path']}` |")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- {result['conclusion']['current_best_candidate']}",
            f"- {result['conclusion']['recommended_use_now']}",
            f"- {result['conclusion']['not_yet_proven']}",
            "",
            "Next required evidence:",
        ]
    )
    for item in result["conclusion"]["next_required_evidence"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument("--real_insertion_gate", type=Path, default=DEFAULT_REAL_GATES["insertion"])
    parser.add_argument("--real_board_gate", type=Path, default=DEFAULT_REAL_GATES["board"])
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    args.real_gate = {"insertion": args.real_insertion_gate, "board": args.real_board_gate}
    return args


def main() -> None:
    args = parse_args()
    result = audit(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "multitask_energy_scorer_audit.json"
    md_path = args.out_dir / "multitask_energy_scorer_audit.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    md = render_md(result)
    md_path.write_text(md)
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(md)
    print(json.dumps({"json": str(json_path), "md": str(md_path), "doc": str(args.doc), "offline_guidance_ready": result["offline_guidance_ready"], "production_validated": result["production_validated"]}, indent=2))


if __name__ == "__main__":
    main()
