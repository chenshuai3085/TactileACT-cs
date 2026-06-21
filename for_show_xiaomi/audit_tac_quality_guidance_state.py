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
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json"
)
DEFAULT_REAL_ROLLOUT = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_eval/"
    "current_forceaware_goodmargin_tac_quality/tac_quality_real_rollout_eval.json"
)
DEFAULT_INSERTION_EVAL = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json")
DEFAULT_INSERTION_GRADIENT = Path(
    "/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/guidance_gradient_audit.json"
)
DEFAULT_INSERTION_DDPM_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "insertion_0401_good_margin_protected_multiep8_start2_seed2_t0_s001/"
    "insertion_ddpm_step_guidance_sweep.json"
)
DEFAULT_BOARD_TRAIN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/train_result.json")
DEFAULT_BOARD_ALIGNMENT = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "foresight_alignment_quality/foresight_score_alignment.json"
)
DEFAULT_BOARD_GRADIENT = Path(
    "/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/"
    "guidance_gradient_audit_quality/guidance_gradient_audit.json"
)
DEFAULT_INSERTION_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "insertion_0401_good_margin_guided_smoke_20260619/guided_server_dry_run_smoke.json"
)
DEFAULT_INSERTION_DENOISE_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "insertion_good_margin_denoising_step_smoke_20260619/guided_server_dry_run_smoke.json"
)
DEFAULT_BOARD_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "board_force_aware_guided_smoke_20260621/guided_server_dry_run_smoke.json"
)
DEFAULT_BOARD_DENOISE_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "board_force_aware_denoising_step_smoke_20260621/guided_server_dry_run_smoke.json"
)
DEFAULT_ROLLOUT_MANIFEST = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/"
    "current_forceaware_goodmargin_manifest/tac_quality_rollout_manifest.json"
)
DEFAULT_MANIFEST_APPLY = Path(
    "/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/"
    "current_forceaware_goodmargin_manifest/manifest_metadata_apply_result.json"
)
DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guidance_state_audit")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data.setdefault("_source_path", str(path))
    return data


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


def required_checks_pass(checks: dict[str, dict[str, Any]]) -> bool:
    return all(item["pass"] for item in checks.values() if item.get("required", True))


def file_info(path: str | None) -> dict[str, Any]:
    if not path:
        return {"path": path, "exists": False}
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    stat = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "bytes": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def metric_mean(container: dict[str, Any], key: str) -> Any:
    value = container.get(key)
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return value


def first_present(container: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in container:
            return container[key]
    return None


def ckpt_from_arm(rollout_config: dict[str, Any] | None, task: str, arm: str) -> str | None:
    return get(rollout_config, f"tasks.{task}.{arm}.checkpoint.path")


def runtime_from_arm(rollout_config: dict[str, Any] | None, task: str, arm: str) -> str | None:
    return get(rollout_config, f"tasks.{task}.{arm}.scorer_runtime")


def smoke_pass(
    smoke: dict[str, Any] | None,
    *,
    task: str,
    arm: str,
    runtime: str,
    guidance_location: str,
    adapter_policy: str,
    every_step: bool | None = None,
) -> dict[str, Any]:
    report = get(smoke, "report", default={})
    ok = (
        bool(get(smoke, "dry_run_guidance_smoke_pass"))
        and get(smoke, "task") == task
        and get(smoke, "arm") == arm
        and bool(get(smoke, "not_reranking"))
        and get(smoke, "guidance_location") == guidance_location
        and get(report, "adapter_policy") == adapter_policy
        and get(report, "scorer_runtime") == runtime
        and num(get(report, "finite_grad_rate")) >= 0.999
        and num(get(report, "positive_grad_rate")) >= 0.999
        and num(get(report, "accept_rate")) > 0.0
    )
    if every_step is not None:
        actual_every_step = bool(get(report, "every_step_ddpm_guidance", False))
        ok = ok and (actual_every_step is every_step)
    detail = {
        "path": get(smoke, "_source_path"),
        "pass": get(smoke, "dry_run_guidance_smoke_pass"),
        "task": get(smoke, "task"),
        "arm": get(smoke, "arm"),
        "not_reranking": get(smoke, "not_reranking"),
        "guidance_location": get(smoke, "guidance_location"),
        "adapter_policy": get(report, "adapter_policy"),
        "every_step_ddpm_guidance": get(report, "every_step_ddpm_guidance"),
        "scorer_runtime": get(report, "scorer_runtime"),
        "score_mode": get(report, "score_mode"),
        "finite_grad_rate": get(report, "finite_grad_rate"),
        "positive_grad_rate": get(report, "positive_grad_rate"),
        "accept_rate": get(report, "accept_rate"),
        "score_delta_mean": get(report, "score_delta.mean"),
    }
    return pass_item(ok, json.dumps(detail, ensure_ascii=False))


def audit_insertion(
    evidence: dict[str, Any] | None,
    rollout_config: dict[str, Any] | None,
    insertion_eval: dict[str, Any] | None,
    gradient_audit: dict[str, Any] | None,
    smoke: dict[str, Any] | None,
    denoise_smoke: dict[str, Any] | None,
    ddpm_sweep: dict[str, Any] | None,
) -> dict[str, Any]:
    task = get(evidence, "tasks.insertion", {})
    arm = get(rollout_config, "recommended_insertion_arm", "good_margin_guided")
    ckpt = ckpt_from_arm(rollout_config, "insertion", arm)
    grouped = get(insertion_eval, "mixed_group_cv", default=None)
    if not isinstance(grouped, dict):
        grouped = task.get("grouped_cv", {}) if isinstance(task, dict) else {}
    gradient = get(gradient_audit, "summary", default=None)
    if not isinstance(gradient, dict):
        gradient = task.get("foresight_gradient_audit", {}) if isinstance(task, dict) else {}
    ddpm = get(ddpm_sweep, "summary", default={})
    checks = {
        "recommended_arm_exists": pass_item(
            get(rollout_config, f"tasks.insertion.{arm}") is not None,
            f"arm={arm}",
        ),
        "runtime_matches": pass_item(
            runtime_from_arm(rollout_config, "insertion", arm) == "InsertionRiskScorerRuntime",
            str(runtime_from_arm(rollout_config, "insertion", arm)),
        ),
        "score_mode_is_good_margin": pass_item(
            get(rollout_config, f"tasks.insertion.{arm}.refiner.score_mode") == "good_margin",
            str(get(rollout_config, f"tasks.insertion.{arm}.refiner.score_mode")),
        ),
        "checkpoint_exists": pass_item(exists(ckpt), str(ckpt)),
        "grouped_cv_auc": pass_item(
            num(metric_mean(grouped, "binary_auc")) >= 0.95 or num(grouped.get("binary_auc_mean")) >= 0.95,
            f"AUC={metric_mean(grouped, 'binary_auc') or grouped.get('binary_auc_mean')}",
        ),
        "grouped_cv_balanced_accuracy": pass_item(
            num(metric_mean(grouped, "binary_balanced_accuracy")) >= 0.90 or num(grouped.get("binary_balanced_accuracy_mean")) >= 0.90,
            f"bACC={metric_mean(grouped, 'binary_balanced_accuracy') or grouped.get('binary_balanced_accuracy_mean')}",
        ),
        "quality_corr": pass_item(
            num(metric_mean(grouped, "quality_corr")) >= 0.50 or num(grouped.get("quality_corr_mean")) >= 0.50,
            f"corr={metric_mean(grouped, 'quality_corr') or grouped.get('quality_corr_mean')}",
        ),
        "clean_action_foresight_gradient": pass_item(
            bool(gradient.get("pass"))
            and num(gradient.get("finite_grad_rate_mean")) >= 0.999
            and num(gradient.get("positive_grad_rate_mean")) >= 0.999
            and num(gradient.get("improved_rate_mean")) >= 0.95
            and num(gradient.get("trust_region_pass_rate")) >= 0.999,
            json.dumps(gradient, ensure_ascii=False),
        ),
        "good_margin_ddpm_step_sweep": pass_item(
            num(ddpm.get("final_score_improve_rate")) >= 0.90
            and num(get(ddpm, "final_score_delta.min")) >= 0.0
            and num(get(ddpm, "finite_grad_rate.mean")) >= 0.999
            and num(get(ddpm, "positive_grad_rate.mean")) >= 0.999,
            json.dumps(ddpm, ensure_ascii=False),
        ),
        "server_final_action_dry_run": smoke_pass(
            smoke,
            task="insertion",
            arm=str(arm),
            runtime="InsertionRiskScorerRuntime",
            guidance_location="after DP clean action chunk",
            adapter_policy="final_clean_action_trust_region_refinement",
            every_step=False,
        ),
        "server_denoising_step_dry_run": smoke_pass(
            denoise_smoke,
            task="insertion",
            arm=str(arm),
            runtime="InsertionRiskScorerRuntime",
            guidance_location="inside DP denoising loop on predicted clean action x0",
            adapter_policy="denoising_step_tac_quality_guidance",
            every_step=True,
        ),
    }
    return {
        "task": "insertion",
        "recommended_arm": arm,
        "scorer": "InsertionRiskScorerRuntime",
        "checkpoint": ckpt,
        "checks": checks,
        "ready_for_real_rollout": required_checks_pass(checks),
    }


def audit_board(
    evidence: dict[str, Any] | None,
    rollout_config: dict[str, Any] | None,
    train_result: dict[str, Any] | None,
    alignment_result: dict[str, Any] | None,
    gradient_audit: dict[str, Any] | None,
    smoke: dict[str, Any] | None,
    denoise_smoke: dict[str, Any] | None,
) -> dict[str, Any]:
    task = get(evidence, "tasks.board", {})
    arm = get(rollout_config, "recommended_board_arm", "marker_joint_guided")
    ckpt = ckpt_from_arm(rollout_config, "board", str(arm))
    runtime = runtime_from_arm(rollout_config, "board", str(arm))
    if arm == "force_aware_guided":
        force_energy = get(rollout_config, "tasks.board.force_aware_guided.refiner.energy", {})
        checks = {
            "recommended_arm_exists": pass_item(
                get(rollout_config, "tasks.board.force_aware_guided") is not None,
                f"arm={arm}",
            ),
            "runtime_matches": pass_item(
                runtime == "ForceAwareForesightGuidanceRuntime",
                str(runtime),
            ),
            "score_preset_margin_only": pass_item(
                force_energy.get("score_preset") == "margin_only",
                json.dumps(force_energy, ensure_ascii=False),
            ),
            "checkpoint_exists": pass_item(exists(ckpt), str(ckpt)),
            "heldout_band_balanced_accuracy": pass_item(
                num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.band_balanced_acc")) >= 0.90,
                f"bACC={get(rollout_config, 'tasks.board.force_aware_guided.refiner.energy.heldout_audit.band_balanced_acc')}",
            ),
            "heldout_contact_accuracy": pass_item(
                num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.contact_acc")) >= 0.85,
                f"contact_acc={get(rollout_config, 'tasks.board.force_aware_guided.refiner.energy.heldout_audit.contact_acc')}",
            ),
            "score_good_bad_auc": pass_item(
                num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.score_good_bad_auc")) >= 0.95,
                f"AUC={get(rollout_config, 'tasks.board.force_aware_guided.refiner.energy.heldout_audit.score_good_bad_auc')}",
            ),
            "gradient_signal": pass_item(
                num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.finite_grad_rate")) >= 0.999
                and num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.positive_grad_rate")) >= 0.999
                and num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.improved_rate")) >= 0.90
                and num(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit.score_delta_mean")) >= 0.25,
                json.dumps(get(rollout_config, "tasks.board.force_aware_guided.refiner.energy.heldout_audit"), ensure_ascii=False),
            ),
            "server_final_action_dry_run": smoke_pass(
                smoke,
                task="board",
                arm=str(arm),
                runtime="ForceAwareForesightGuidanceRuntime",
                guidance_location="after DP clean action chunk",
                adapter_policy="force_aware_foresight_trust_region_refinement",
                every_step=False,
            ),
            "server_denoising_step_dry_run": {
                **smoke_pass(
                    denoise_smoke,
                    task="board",
                    arm=str(arm),
                    runtime="ForceAwareForesightGuidanceRuntime",
                    guidance_location="inside DP denoising loop on predicted clean action x0",
                    adapter_policy="denoising_step_force_aware_tac_quality_guidance",
                    every_step=True,
                ),
                "required": False,
            },
        }
        return {
            "task": "board",
            "recommended_arm": arm,
            "scorer": "ForceAwareForesightGuidanceRuntime(force-aware Foresight)",
            "checkpoint": ckpt,
            "checks": checks,
            "ready_for_real_rollout": required_checks_pass(checks),
            "integrated_fallback_arm": get(rollout_config, "integrated_fallback_board_arm"),
        }

    grouped = get(train_result, "best.val", default=None)
    if not isinstance(grouped, dict):
        grouped = task.get("grouped_heldout", {}) if isinstance(task, dict) else {}
    alignment = get(alignment_result, "summary", default=None)
    if not isinstance(alignment, dict):
        alignment = task.get("foresight_alignment", {}) if isinstance(task, dict) else {}
    gradient = get(gradient_audit, "summary", default=None)
    if not isinstance(gradient, dict):
        gradient = task.get("foresight_gradient_audit", {}) if isinstance(task, dict) else {}
    energy_source = get(rollout_config, f"tasks.board.{arm}.refiner.energy.source")
    force_band_alignment = first_present(
        alignment,
        "pred_score_vs_force_band_quality_spearman",
        "pred_score_force_band_quality_spearman",
    )
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
            and num(force_band_alignment) >= 0.30,
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
        "server_final_action_dry_run": smoke_pass(
            smoke,
            task="board",
            arm=str(arm),
            runtime="ForceBandTacQualityEnergyRuntime",
            guidance_location="after DP clean action chunk",
            adapter_policy="final_clean_action_trust_region_refinement",
            every_step=False,
        ),
        "server_denoising_step_dry_run": smoke_pass(
            denoise_smoke,
            task="board",
            arm=str(arm),
            runtime="ForceBandTacQualityEnergyRuntime",
            guidance_location="inside DP denoising loop on predicted clean action x0",
            adapter_policy="denoising_step_tac_quality_guidance",
            every_step=True,
        ),
    }
    return {
        "task": "board",
        "recommended_arm": arm,
        "scorer": "ForceBandTacQualityEnergyRuntime(marker_joint_action)",
        "checkpoint": ckpt,
        "checks": checks,
        "ready_for_real_rollout": required_checks_pass(checks),
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


def audit_manifest_pipeline(
    manifest: dict[str, Any] | None,
    manifest_apply: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = manifest.get("rows", []) if isinstance(manifest, dict) else []
    tasks = manifest.get("tasks", []) if isinstance(manifest, dict) else []
    pair_ids_by_task: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        task = str(row.get("task") or "unknown")
        group = str(row.get("group") or "unknown")
        pair_id = str(row.get("pair_id") or "")
        counts.setdefault(task, {})
        counts[task][group] = counts[task].get(group, 0) + 1
        if pair_id:
            pair_ids_by_task.setdefault(task, set()).add(pair_id)
    expected_counts_ok = (
        counts.get("board", {}).get("baseline") == 3
        and counts.get("board", {}).get("guided") == 3
        and counts.get("insertion", {}).get("baseline") == 3
        and counts.get("insertion", {}).get("guided") == 3
    )
    paired_ids_ok = (
        len(pair_ids_by_task.get("board", set())) == 3
        and len(pair_ids_by_task.get("insertion", set())) == 3
    )
    apply_ok = bool(get(manifest_apply, "ok", False))
    no_ambiguity = int(get(manifest_apply, "ambiguous", 999) or 0) == 0
    no_mismatch = int(get(manifest_apply, "metadata_mismatch", 999) or 0) == 0
    n_manifest_rows = int(get(manifest_apply, "n_manifest_rows", len(rows)) or 0)
    return {
        "manifest_exists": manifest is not None,
        "manifest_apply_exists": manifest_apply is not None,
        "manifest_path": get(manifest, "_source_path"),
        "manifest_apply_path": get(manifest_apply, "_source_path"),
        "tasks": tasks,
        "n_rows": len(rows),
        "counts": {task: dict(group_counts) for task, group_counts in counts.items()},
        "pair_id_counts": {task: len(values) for task, values in pair_ids_by_task.items()},
        "checks": {
            "expected_3_pairs_per_task": pass_item(
                expected_counts_ok and paired_ids_ok,
                json.dumps({
                    "counts": counts,
                    "pair_id_counts": {task: len(values) for task, values in pair_ids_by_task.items()},
                }, ensure_ascii=False),
            ),
            "apply_dry_run_ok": pass_item(
                apply_ok and no_ambiguity and no_mismatch and n_manifest_rows == len(rows),
                json.dumps({
                    "ok": get(manifest_apply, "ok"),
                    "n_manifest_rows": n_manifest_rows,
                    "missing": get(manifest_apply, "missing"),
                    "ambiguous": get(manifest_apply, "ambiguous"),
                    "metadata_mismatch": get(manifest_apply, "metadata_mismatch"),
                }, ensure_ascii=False),
            ),
        },
        "ready_for_real_metadata_binding": bool(
            manifest is not None
            and manifest_apply is not None
            and expected_counts_ok
            and paired_ids_ok
            and apply_ok
            and no_ambiguity
            and no_mismatch
        ),
        "note": "missing rows are allowed before real robot trials; after trials, missing should become 0.",
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
        f"- real_evidence_pipeline_ready: `{result['real_evidence_pipeline_ready']}`",
        f"- overall_goal_complete: `{result['overall_goal_complete']}`",
        f"- denoising_step_serving_ready: `{result['denoising_step_serving_ready']}`",
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
    lines.extend([
        "",
        "## Serving Guidance Paths",
        "",
        "| task | final-action smoke | denoising-step smoke |",
        "|---|---:|---:|",
        f"| insertion | `{result['insertion']['checks']['server_final_action_dry_run']['pass']}` | `{result['insertion']['checks']['server_denoising_step_dry_run']['pass']}` |",
        f"| board | `{result['board']['checks']['server_final_action_dry_run']['pass']}` | `{result['board']['checks']['server_denoising_step_dry_run']['pass']}` |",
        "",
        "Denoising-step smoke proves the service path can apply TacQuality gradients inside the DP denoising loop on predicted clean action `x0`; it is not real robot outcome evidence.",
    ])
    lines.extend([
        "",
        "## Real Evidence Pipeline",
        "",
        f"- manifest exists: `{result['manifest_pipeline']['manifest_exists']}`",
        f"- manifest apply exists: `{result['manifest_pipeline']['manifest_apply_exists']}`",
        f"- ready_for_real_metadata_binding: `{result['manifest_pipeline']['ready_for_real_metadata_binding']}`",
        f"- n_rows: `{result['manifest_pipeline']['n_rows']}`",
        f"- counts: `{json.dumps(result['manifest_pipeline']['counts'], ensure_ascii=False)}`",
        f"- pair_id_counts: `{json.dumps(result['manifest_pipeline']['pair_id_counts'], ensure_ascii=False)}`",
        "",
        "| check | pass | detail |",
        "|---|---:|---|",
    ])
    for name, item in result["manifest_pipeline"]["checks"].items():
        detail = str(item["detail"]).replace("\n", " ")[:500]
        lines.append(f"| `{name}` | `{item['pass']}` | {detail} |")
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
    parser.add_argument("--insertion_eval", default=str(DEFAULT_INSERTION_EVAL))
    parser.add_argument("--insertion_gradient", default=str(DEFAULT_INSERTION_GRADIENT))
    parser.add_argument("--insertion_ddpm_sweep", default=str(DEFAULT_INSERTION_DDPM_SWEEP))
    parser.add_argument("--board_train", default=str(DEFAULT_BOARD_TRAIN))
    parser.add_argument("--board_alignment", default=str(DEFAULT_BOARD_ALIGNMENT))
    parser.add_argument("--board_gradient", default=str(DEFAULT_BOARD_GRADIENT))
    parser.add_argument("--insertion_smoke", default=str(DEFAULT_INSERTION_SMOKE))
    parser.add_argument("--insertion_denoise_smoke", default=str(DEFAULT_INSERTION_DENOISE_SMOKE))
    parser.add_argument("--board_smoke", default=str(DEFAULT_BOARD_SMOKE))
    parser.add_argument("--board_denoise_smoke", default=str(DEFAULT_BOARD_DENOISE_SMOKE))
    parser.add_argument("--rollout_manifest", default=str(DEFAULT_ROLLOUT_MANIFEST))
    parser.add_argument("--manifest_apply", default=str(DEFAULT_MANIFEST_APPLY))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = load_json(Path(args.evidence_audit))
    rollout_config = load_json(Path(args.rollout_config))
    real_rollout = load_json(Path(args.real_rollout_summary))
    insertion_eval = load_json(Path(args.insertion_eval))
    insertion_gradient = load_json(Path(args.insertion_gradient))
    insertion_ddpm_sweep = load_json(Path(args.insertion_ddpm_sweep))
    board_train = load_json(Path(args.board_train))
    board_alignment = load_json(Path(args.board_alignment))
    board_gradient = load_json(Path(args.board_gradient))
    insertion_smoke = load_json(Path(args.insertion_smoke))
    insertion_denoise_smoke = load_json(Path(args.insertion_denoise_smoke))
    board_smoke = load_json(Path(args.board_smoke))
    board_denoise_smoke = load_json(Path(args.board_denoise_smoke))
    rollout_manifest = load_json(Path(args.rollout_manifest))
    manifest_apply = load_json(Path(args.manifest_apply))

    result = {
        "inputs": {
            "evidence_audit": args.evidence_audit,
            "rollout_config": args.rollout_config,
            "real_rollout_summary": args.real_rollout_summary,
            "insertion_eval": args.insertion_eval,
            "insertion_gradient": args.insertion_gradient,
            "insertion_ddpm_sweep": args.insertion_ddpm_sweep,
            "board_train": args.board_train,
            "board_alignment": args.board_alignment,
            "board_gradient": args.board_gradient,
            "insertion_smoke": args.insertion_smoke,
            "insertion_denoise_smoke": args.insertion_denoise_smoke,
            "board_smoke": args.board_smoke,
            "board_denoise_smoke": args.board_denoise_smoke,
            "rollout_manifest": args.rollout_manifest,
            "manifest_apply": args.manifest_apply,
        },
        "insertion": audit_insertion(
            evidence,
            rollout_config,
            insertion_eval,
            insertion_gradient,
            insertion_smoke,
            insertion_denoise_smoke,
            insertion_ddpm_sweep,
        ),
        "board": audit_board(
            evidence,
            rollout_config,
            board_train,
            board_alignment,
            board_gradient,
            board_smoke,
            board_denoise_smoke,
        ),
        "real_rollout": audit_real_rollout(real_rollout),
        "manifest_pipeline": audit_manifest_pipeline(rollout_manifest, manifest_apply),
    }
    result["denoising_step_serving_ready"] = bool(
        result["insertion"]["checks"]["server_denoising_step_dry_run"]["pass"]
        and result["board"]["checks"]["server_denoising_step_dry_run"]["pass"]
    )
    result["real_rollout_evidence_complete"] = bool(result["real_rollout"]["real_rollout_evidence_complete"])
    result["checkpoint_files"] = {
        "insertion": file_info(result["insertion"].get("checkpoint")),
        "board": file_info(result["board"].get("checkpoint")),
    }
    result["real_evidence_pipeline_ready"] = bool(result["manifest_pipeline"]["ready_for_real_metadata_binding"])
    result["overall_goal_complete"] = bool(
        result["insertion"]["ready_for_real_rollout"]
        and result["board"]["ready_for_real_rollout"]
        and result["real_rollout_evidence_complete"]
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
        "denoising_step_serving_ready": result["denoising_step_serving_ready"],
        "real_evidence_pipeline_ready": result["real_evidence_pipeline_ready"],
        "real_rollout_evidence_complete": result["real_rollout"]["real_rollout_evidence_complete"],
        "overall_goal_complete": result["overall_goal_complete"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
