#!/usr/bin/env python3
"""Build the current TacQuality scorer readiness matrix.

The generated document is deliberately evidence-bound: it reads the latest
offline scorer audits, Foresight/guidance audits, real-rollout status, and the
active DP training monitor instead of relying on stale hand-written paths.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


DEFAULT_EVIDENCE = Path("/home/chenshuai/Project/output/tac_quality_evidence_audit_20260618/tac_quality_evidence_audit.json")
DEFAULT_STATE = Path("/home/chenshuai/Project/output/tac_quality_guidance_state_audit/tac_quality_guidance_state_audit.json")
DEFAULT_REAL = Path("/home/chenshuai/Project/output/tac_quality_real_rollout_eval/smoke_no_real_rollouts_current/tac_quality_real_rollout_eval.json")
DEFAULT_BOARD_TRAIN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json")
DEFAULT_BOARD_ALIGN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json")
DEFAULT_BOARD_GRAD = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json")
DEFAULT_BOARD_S12_TRAIN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/train_result.json")
DEFAULT_BOARD_S12_ALIGN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/foresight_alignment_quality/foresight_score_alignment.json")
DEFAULT_BOARD_S12_GRAD = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/guidance_gradient_audit_quality/guidance_gradient_audit.json")
DEFAULT_BOARD_OLD_INCLUDE260617_ALIGN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality_include260617_sameset/foresight_score_alignment.json")
DEFAULT_BOARD_OLD_INCLUDE260617_GRAD = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality_include260617_sameset/guidance_gradient_audit.json")
DEFAULT_BOARD_SMOKE = Path("/home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_20260619_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json")
DEFAULT_BOARD_S12_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "board_260617_marker_joint_s12_guided_smoke_current_20260619/"
    "guided_server_dry_run_smoke.json"
)
DEFAULT_INSERT_EVAL = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json")
DEFAULT_INSERT_GRAD = Path("/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json")
DEFAULT_INSERT_GRAD_0209 = Path("/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0209_matched_20260619/guidance_gradient_audit.json")
DEFAULT_INSERT_GRAD_0401 = Path("/home/chenshuai/Project/output/insertion_guidance_gradient_audit_0401_matched_20260619/guidance_gradient_audit.json")
DEFAULT_INSERT_SMOKE = Path("/home/chenshuai/Project/output/tac_quality_guided_server_packet/insertion_0401_default_guided_smoke_20260619/guided_server_dry_run_smoke.json")
DEFAULT_INSERT_GOOD_MARGIN_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "insertion_0401_good_margin_guided_smoke_20260619/"
    "guided_server_dry_run_smoke.json"
)
DEFAULT_BOARD_GATE_SKIP_SMOKE = Path("/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_marker_joint_board_contact_gate_skip_20260619/guided_server_dry_run_smoke.json")
DEFAULT_BOARD_NOISY_ACTION_AUDIT = Path("/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/board_marker_joint_s12_260617_fast4/noisy_action_guidance_audit.json")
DEFAULT_INSERT_NOISY_ACTION_AUDIT = Path("/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_profile_current_fast4/noisy_action_guidance_audit.json")
DEFAULT_INSERT_NOISY_ACTION_AUDIT_0209 = Path("/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0209_matched_fast4/noisy_action_guidance_audit.json")
DEFAULT_INSERT_NOISY_ACTION_AUDIT_0401 = Path("/home/chenshuai/Project/output/tac_quality_noisy_action_guidance_audit/insertion_0401_matched_fast4/noisy_action_guidance_audit.json")
DEFAULT_INSERT_DDPM_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "insertion_0401_default_protected_multiep8_start2_seed2_t0_s001/"
    "insertion_ddpm_step_guidance_sweep.json"
)
DEFAULT_INSERT_PGOOD_DDPM_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "insertion_0401_p_good_protected_multiep8_start2_seed2_t0_s001/"
    "insertion_ddpm_step_guidance_sweep.json"
)
DEFAULT_INSERT_SCORE_MODE_ABLATION = Path(
    "/home/chenshuai/Project/output/tac_quality_score_mode_ablation/"
    "insertion_0401_profile_pgood_energy_goodmargin_cross_score_20260619/"
    "insertion_score_mode_ablation.json"
)
DEFAULT_BOARD_DDPM_AUDITS = [
    Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260619_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json"),
    Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_20260618ext_ep2_s80_t0_s001_seed1_4/ddpm_step_guidance_audit.json"),
    Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_real_chain_smoke/ddpm_step_guidance_audit.json"),
    Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s001_seed1/ddpm_step_guidance_audit.json"),
    Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_t0_s0005_seed1/ddpm_step_guidance_audit.json"),
    Path("/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/board_marker_joint_260617_steps8_t0_s001_seed1/ddpm_step_guidance_audit.json"),
]
DEFAULT_BOARD_DDPM_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "board_marker_joint_260617_20260619_protected_multiep6_start2_seed2_t0_s001/"
    "board_ddpm_step_guidance_sweep.json"
)
DEFAULT_BOARD_S12_DDPM_SWEEP = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "board_marker_joint_s12_260617_20260619_protected_multiep6_start2_seed2_t0_s001/"
    "board_ddpm_step_guidance_sweep.json"
)
DEFAULT_SEMANTIC_DIRECTION = Path(
    "/home/chenshuai/Project/output/tac_quality_semantic_direction_audit/"
    "tac_quality_semantic_direction_audit.json"
)
DEFAULT_ROLLOUT_CONFIG = Path("/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260619_semantic_pgood_s12.json")
DEFAULT_GOOD_MARGIN_ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs_marker_joint_20260619_insertion_good_margin.json"
)
DEFAULT_DP_RUN = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/"
    "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_"
    "20260619_full_noearly_tmux"
)
DEFAULT_OUTPUT_MD = Path("docs/2026-06-18_tac_quality_guidance_readiness_matrix.md")
DEFAULT_OUTPUT_JSON = Path("/home/chenshuai/Project/output/tac_quality_current_readiness_matrix/tac_quality_current_readiness_matrix.json")
CURRENT_BOARD_ARM = "marker_joint_s12_guided"
CURRENT_BOARD_ROOT = Path("/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer")
CURRENT_INSERTION_ARM = "good_margin_guided"
CURRENT_INSERTION_ROOT = Path("/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer")
CURRENT_BOARD_SCORER = "ForceBandTacQualityEnergyRuntime(marker_joint_action,s12)"
CURRENT_BOARD_CHECKPOINT = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt")
CURRENT_INSERTION_SCORE_MODE = "good_margin"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("_source_path", str(path))
        return data
    except Exception as exc:
        return {"_error": repr(exc), "_path": str(path)}


def get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value: Any, ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def exists(path: str | Path | None) -> bool:
    if not path:
        return False
    return Path(path).exists()


def dp_best_path(run_dir: Path) -> Path:
    return run_dir / "dp_best.pth"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    evidence = load_json(args.evidence)
    state = load_json(args.state)
    real = load_json(args.real_rollout)
    board_train = load_json(args.board_train)
    board_align = load_json(args.board_alignment)
    board_grad = load_json(args.board_gradient)
    board_s12_train = load_json(args.board_s12_train)
    board_s12_align = load_json(args.board_s12_alignment)
    board_s12_grad = load_json(args.board_s12_gradient)
    board_old_include260617_align = load_json(args.board_old_include260617_alignment)
    board_old_include260617_grad = load_json(args.board_old_include260617_gradient)
    board_smoke = load_json(args.board_smoke)
    board_s12_smoke = load_json(args.board_s12_smoke)
    insert_eval = load_json(args.insertion_eval)
    insert_grad = load_json(args.insertion_gradient)
    insert_grad_0209 = load_json(args.insertion_gradient_0209)
    insert_grad_0401 = load_json(args.insertion_gradient_0401)
    insert_smoke = load_json(args.insertion_smoke)
    insert_good_margin_smoke = load_json(args.insertion_good_margin_smoke)
    board_gate_skip_smoke = load_json(args.board_gate_skip_smoke)
    board_noisy_action_audit = load_json(args.board_noisy_action_audit)
    insert_noisy_action_audit = load_json(args.insertion_noisy_action_audit)
    insert_noisy_action_audit_0209 = load_json(args.insertion_noisy_action_audit_0209)
    insert_noisy_action_audit_0401 = load_json(args.insertion_noisy_action_audit_0401)
    insert_ddpm_step_sweep = load_json(args.insertion_ddpm_step_sweep)
    insert_pgood_ddpm_step_sweep = load_json(args.insertion_pgood_ddpm_step_sweep)
    insert_score_mode_ablation = load_json(args.insertion_score_mode_ablation)
    board_ddpm_step_audits = [load_json(path) for path in args.board_ddpm_step_audits]
    board_ddpm_step_sweep = load_json(args.board_ddpm_step_sweep)
    board_s12_ddpm_step_sweep = load_json(args.board_s12_ddpm_step_sweep)
    semantic_direction = load_json(args.semantic_direction)
    rollout_config = load_json(args.rollout_config)
    dp_status = load_json(args.dp_run / "training_status_latest.json")
    dp_stop = load_json(args.dp_run / "early_stop_summary.json")

    insertion = get(evidence, "tasks", "insertion", default={})
    board = get(evidence, "tasks", "board", default={})

    return {
        "created_at": f"{dt.datetime.now():%F %T}",
        "paths": {
            "evidence": str(args.evidence),
            "state": str(args.state),
            "real_rollout": str(args.real_rollout),
            "board_train": str(args.board_train),
            "board_alignment": str(args.board_alignment),
            "board_gradient": str(args.board_gradient),
            "board_s12_train": str(args.board_s12_train),
            "board_s12_alignment": str(args.board_s12_alignment),
            "board_s12_gradient": str(args.board_s12_gradient),
            "board_old_include260617_alignment": str(args.board_old_include260617_alignment),
            "board_old_include260617_gradient": str(args.board_old_include260617_gradient),
            "board_smoke": str(args.board_smoke),
            "board_s12_smoke": str(args.board_s12_smoke),
            "insertion_eval": str(args.insertion_eval),
            "insertion_gradient": str(args.insertion_gradient),
            "insertion_gradient_0209": str(args.insertion_gradient_0209),
            "insertion_gradient_0401": str(args.insertion_gradient_0401),
            "insertion_smoke": str(args.insertion_smoke),
            "insertion_good_margin_smoke": str(args.insertion_good_margin_smoke),
            "board_gate_skip_smoke": str(args.board_gate_skip_smoke),
            "board_noisy_action_audit": str(args.board_noisy_action_audit),
            "insertion_noisy_action_audit": str(args.insertion_noisy_action_audit),
            "insertion_noisy_action_audit_0209": str(args.insertion_noisy_action_audit_0209),
            "insertion_noisy_action_audit_0401": str(args.insertion_noisy_action_audit_0401),
            "insertion_ddpm_step_sweep": str(args.insertion_ddpm_step_sweep),
            "insertion_pgood_ddpm_step_sweep": str(args.insertion_pgood_ddpm_step_sweep),
            "insertion_score_mode_ablation": str(args.insertion_score_mode_ablation),
            "board_ddpm_step_audits": [str(path) for path in args.board_ddpm_step_audits],
            "board_ddpm_step_sweep": str(args.board_ddpm_step_sweep),
            "board_s12_ddpm_step_sweep": str(args.board_s12_ddpm_step_sweep),
            "semantic_direction": str(args.semantic_direction),
            "rollout_config": str(args.rollout_config),
            "good_margin_rollout_config": str(args.good_margin_rollout_config),
            "dp_run": str(args.dp_run),
        },
        "insertion": {
            "recommended_arm": CURRENT_INSERTION_ARM,
            "scorer": get(insertion, "scorer", default="InsertionRiskScorerRuntime"),
            "checkpoint": get(insertion, "checkpoint", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"),
            "score_mode": CURRENT_INSERTION_SCORE_MODE,
            "cv": get(insert_eval, "mixed_group_cv", default=get(insertion, "grouped_cv", default={})),
            "gradient": get(insert_grad, "summary", default=get(insertion, "foresight_gradient_audit", default={})),
            "gradient_0209": get(insert_grad_0209, "summary", default={}),
            "gradient_0401": get(insert_grad_0401, "summary", default={}),
            "server_smoke": insert_smoke,
            "good_margin_server_smoke": insert_good_margin_smoke,
            "noisy_action_audit": insert_noisy_action_audit,
            "noisy_action_audit_0209": insert_noisy_action_audit_0209,
            "noisy_action_audit_0401": insert_noisy_action_audit_0401,
            "ddpm_step_sweep": insert_ddpm_step_sweep,
            "pgood_ddpm_step_sweep": insert_pgood_ddpm_step_sweep,
            "score_mode_ablation": insert_score_mode_ablation,
            "ready_for_real_rollout": get(state, "insertion", "ready_for_real_rollout", default=False),
        },
        "board": {
            "recommended_arm": CURRENT_BOARD_ARM,
            "scorer": CURRENT_BOARD_SCORER,
            "checkpoint": str(CURRENT_BOARD_CHECKPOINT),
            "score_mode": get(board, "score_mode", default="quality"),
            "train_result": board_train,
            "alignment": board_align,
            "gradient": get(board_grad, "summary", default=get(board, "foresight_gradient_audit", default={})),
            "s12_train_result": board_s12_train,
            "s12_alignment": board_s12_align,
            "s12_gradient": get(board_s12_grad, "summary", default={}),
            "old_include260617_alignment": board_old_include260617_align,
            "old_include260617_gradient": get(board_old_include260617_grad, "summary", default={}),
            "server_smoke": board_smoke,
            "s12_server_smoke": board_s12_smoke,
            "contact_gate_skip_smoke": board_gate_skip_smoke,
            "noisy_action_audit": board_noisy_action_audit,
            "ddpm_step_audits": board_ddpm_step_audits,
            "ddpm_step_sweep": board_ddpm_step_sweep,
            "s12_ddpm_step_sweep": board_s12_ddpm_step_sweep,
            "ready_for_real_rollout": get(state, "board", "ready_for_real_rollout", default=False),
        },
        "semantic_direction": semantic_direction,
        "rollout_config": rollout_config,
        "real_rollout": real,
        "dp": {
            "run_dir": str(args.dp_run),
            "status": dp_status,
            "early_stop_summary": dp_stop,
            "recommended_ckpt": str(dp_best_path(args.dp_run)),
            "recommended_ckpt_exists": dp_best_path(args.dp_run).exists(),
        },
        "conclusion": {
            "offline_ready": bool(get(evidence, "gates", "insertion_offline_ready", default=False))
            and bool(get(evidence, "gates", "board_offline_ready", default=False)),
            "gradient_ready": bool(get(evidence, "gates", "insertion_gradient_ready", default=False))
            and bool(get(evidence, "gates", "board_gradient_ready", default=False)),
            "real_rollout_proven": bool(get(evidence, "gates", "real_rollout_proven", default=False))
            and bool(get(real, "real_rollout_evidence_complete", default=False)),
        },
    }


def cv_value(cv: dict[str, Any], key: str) -> Any:
    value = cv.get(key)
    if isinstance(value, dict) and "mean" in value:
        return value["mean"]
    return value


def noisy_levels_summary(audit: dict[str, Any]) -> str:
    by_level = audit.get("by_noise_level")
    if not isinstance(by_level, dict) or not by_level:
        return "NA"
    parts = []
    for key in sorted(by_level, key=lambda x: float(x)):
        value = by_level[key]
        if not isinstance(value, dict):
            continue
        parts.append(
            f"{key}:improve={fmt(value.get('score_improve_rate'))},delta={fmt(get(value, 'final_minus_noisy', 'mean'))}"
        )
    return "; ".join(parts) if parts else "NA"


def ddpm_audit_name(audit: dict[str, Any]) -> str:
    guidance = audit.get("guidance", {}) if isinstance(audit, dict) else {}
    steps = guidance.get("num_inference_steps", "NA")
    guided = guidance.get("guidance_steps", "NA")
    scale = guidance.get("guidance_scale", "NA")
    return f"{steps}inf/{guided}guide/scale={scale}"


def render_md(summary: dict[str, Any]) -> str:
    ins = summary["insertion"]
    board = summary["board"]
    board_best = get(board, "train_result", "best", "val", default={})
    board_align = get(board, "alignment", "summary", default={})
    board_s12_best = get(board, "s12_train_result", "best", "val", default={})
    board_s12_align = get(board, "s12_alignment", "summary", default={})
    board_s12_grad = board.get("s12_gradient", {})
    board_old_include260617_align = get(board, "old_include260617_alignment", "summary", default={})
    board_old_include260617_grad = board.get("old_include260617_gradient", {})
    dp_status = get(summary, "dp", "status", default={})
    dp_stop = get(summary, "dp", "early_stop_summary", default={})
    dp_latest = get(dp_status, "latest", default={})
    dp_best = get(dp_status, "best_val_epoch", default={})
    dp_trend = get(dp_status, "trend", default={})
    dp_stopped = (
        bool(dp_status.get("stopped"))
        or (
            isinstance(dp_stop, dict)
            and not dp_stop.get("_missing", False)
            and ("stopped_at" in dp_stop or "status" in dp_stop or "created_at" in dp_stop)
        )
    )
    real = summary["real_rollout"]
    conclusion = summary["conclusion"]

    insert_cv = ins["cv"]
    insert_grad = ins["gradient"]
    insert_grad_0209 = ins.get("gradient_0209", {})
    insert_grad_0401 = ins.get("gradient_0401", {})
    board_grad = board["gradient"]
    insert_smoke = ins["server_smoke"]
    insert_good_margin_smoke = ins.get("good_margin_server_smoke", {})
    board_smoke = board["server_smoke"]
    board_s12_smoke = board.get("s12_server_smoke", {})
    board_gate_skip_smoke = board["contact_gate_skip_smoke"]
    insert_noisy = ins["noisy_action_audit"]
    insert_noisy_0209 = ins.get("noisy_action_audit_0209", {})
    insert_noisy_0401 = ins.get("noisy_action_audit_0401", {})
    insert_ddpm_sweep = ins.get("ddpm_step_sweep", {})
    insert_pgood_ddpm_sweep = ins.get("pgood_ddpm_step_sweep", {})
    insert_score_mode_ablation = ins.get("score_mode_ablation", {})
    board_noisy = board["noisy_action_audit"]
    board_ddpm_audits = board["ddpm_step_audits"]
    board_ddpm_sweep = board["ddpm_step_sweep"]
    board_s12_ddpm_sweep = board.get("s12_ddpm_step_sweep", {})
    semantic_direction = summary.get("semantic_direction", {})
    insertion_recommended_score_mode = CURRENT_INSERTION_SCORE_MODE
    insertion_profile_score_mode = get(insert_smoke, "report", "score_mode", default="profile")
    insertion_good_margin_score_mode = get(insert_good_margin_smoke, "report", "score_mode", default="good_margin")
    board_score_mode = get(board_smoke, "report", "score_mode", default=board["score_mode"])
    board_s12_score_mode = get(board_s12_smoke, "report", "score_mode", default="quality")
    insertion_runtime = get(insert_smoke, "report", "scorer_runtime",
                            default=get(insert_smoke, "report", "profile", "scorer", default=ins["scorer"]))
    insertion_good_margin_runtime = get(insert_good_margin_smoke, "report", "scorer_runtime", default=ins["scorer"])
    board_runtime = get(board_smoke, "report", "scorer_runtime", default="ForceBandTacQualityEnergyRuntime")
    board_s12_runtime = get(board_s12_smoke, "report", "scorer_runtime", default="ForceBandTacQualityEnergyRuntime")

    lines: list[str] = []
    lines.append("# 2026-06-18 TacQuality Guidance Readiness Matrix")
    lines.append("")
    lines.append(f"Generated at: `{summary['created_at']}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This document tracks the current TacQuality classifier/energy scorers for DP classifier guidance:")
    lines.append("")
    lines.append("```text")
    lines.append("DP clean action")
    lines.append("  -> task Foresight predicts future tactile consequence")
    lines.append("  -> TacQuality scorer gives a differentiable score")
    lines.append("  -> trust-region gradient update on the action chunk")
    lines.append("```")
    lines.append("")
    lines.append("This is clean-action classifier/energy guidance. It is not reranking.")
    lines.append("")
    lines.append("## Current Recommendation")
    lines.append("")
    lines.append("| task | recommended arm | scorer | checkpoint | score mode | rollout readiness |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| insertion | `{ins['recommended_arm']}` | `{ins['scorer']}` | `{ins['checkpoint']}` | `{insertion_recommended_score_mode}` | {fmt(ins['ready_for_real_rollout'])} |"
    )
    lines.append(
        f"| board | `{board['recommended_arm']}` | `{board['scorer']}` | `{board['checkpoint']}` | `{board_score_mode}` | {fmt(board['ready_for_real_rollout'])} |"
    )
    lines.append("")
    lines.append("Board note: a stronger offline s12 candidate exists at")
    lines.append("`/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt`.")
    lines.append("It improves held-out predicted-domain metrics and now also has stronger semantic bad-to-good guidance geometry plus a stronger protected DDPM-step sweep.")
    lines.append("It is the next board A/B candidate, but still not a real-robot improvement claim.")
    lines.append("See `docs/2026-06-19_board_scorer_s12_predicted_domain_comparison.md`.")
    lines.append("A simple old/s12 ensemble sweep found a tiny offline gain for rank-normalized `0.85*old + 0.15*s12`, but the gain is too small to justify deployment complexity before real rollouts.")
    lines.append("See `docs/2026-06-19_board_scorer_ensemble_sweep.md`.")
    lines.append("Semantic direction evidence is summarized in `docs/2026-06-19_tac_quality_semantic_direction_audit.md`.")
    lines.append("")
    if not semantic_direction.get("_missing"):
        lines.append("## Semantic Direction Evidence")
        lines.append("")
        lines.append("This audit checks whether score gradients point from bad tactile outcomes toward good tactile outcomes, not just whether the classifier separates labels.")
        lines.append("")
        lines.append("| task | deployed mode | semantic recommended mode | correction pass | strict pass | best bad-to-good projection | evidence |")
        lines.append("|---|---|---|---:|---:|---:|---|")
        for task_name, sem in semantic_direction.get("summary", {}).items():
            recommended = sem.get("recommended", {})
            lines.append(
                f"| {task_name} | `{sem.get('deployed_guidance_mode')}` | `{sem.get('recommended_mode_by_semantic_direction')}` | "
                f"{fmt(sem.get('correction_pass'))} | {fmt(sem.get('strict_pass'))} | "
                f"{fmt(recommended.get('bad_to_good_projection_mean'))} | `{summary['paths']['semantic_direction']}` |"
            )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("")
        lines.append("- Insertion `p_good` has better semantic direction geometry than `profile`, but the DDPM-step sweep below shows it saturates at score 1.0 and gives no sampler improvement.")
        lines.append("- A follow-up insertion cross-score ablation shows that the unsaturated `good_margin` logit margin avoids this saturation and is the stronger next insertion A/B candidate.")
        lines.append("- Board s12 `quality` passes bad-to-good correction geometry and is a stronger board A/B candidate than the old/default scorer.")
        lines.append("- Strict pass is still false, so accept-only and final fallback remain required.")
        lines.append("")
    lines.append("## Offline Scorer Evidence")
    lines.append("")
    lines.append("| task | protocol | AUC | bACC | reason F1 | quality corr / Spearman | evidence |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    lines.append(
        f"| insertion | GroupKFold over insertion windows | {fmt(cv_value(insert_cv, 'binary_auc'))} | {fmt(cv_value(insert_cv, 'binary_balanced_accuracy'))} | {fmt(cv_value(insert_cv, 'reason_macro_f1'))} | {fmt(cv_value(insert_cv, 'quality_corr'))} | `{summary['paths']['insertion_eval']}` |"
    )
    lines.append(
        f"| board | grouped held-out deploy features, `marker_joint_action` | {fmt(board_best.get('binary_auc'))} | {fmt(board_best.get('binary_balanced_accuracy'))} | {fmt(board_best.get('reason_macro_f1'))} | {fmt(board_best.get('quality_spearman'))} | `{summary['paths']['board_train']}` |"
    )
    if not board.get("s12_train_result", {}).get("_missing"):
        lines.append(
            f"| board s12 candidate | grouped held-out predicted-domain deploy features, `marker_joint_action` | {fmt(board_s12_best.get('binary_auc'))} | {fmt(board_s12_best.get('binary_balanced_accuracy'))} | {fmt(board_s12_best.get('reason_macro_f1'))} | {fmt(board_s12_best.get('quality_spearman'))} | `{summary['paths']['board_s12_train']}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Insertion has strong binary risk separation and usable continuous quality correlation.")
    lines.append("- Board uses deploy-aligned features: Foresight-predicted marker proxy plus candidate joint-action proxy. It does not use unavailable future `eef_abs`.")
    lines.append("")
    lines.append("## Foresight-Chain Alignment")
    lines.append("")
    lines.append("| task | score mode | samples | pred AUC(good) | GT AUC(good) | pred/GT Spearman | score vs force quality | evidence |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| insertion | `{insertion_recommended_score_mode}` | NA | NA | NA | NA | NA | gradient audit below |"
    )
    lines.append(
        f"| board | `{board['score_mode']}` | {fmt(board_align.get('n'), 0)} | {fmt(board_align.get('pred_auc_good'))} | {fmt(board_align.get('gt_auc_good'))} | {fmt(board_align.get('pred_gt_spearman'))} | {fmt(board_align.get('pred_score_vs_force_band_quality_spearman'))} | `{summary['paths']['board_alignment']}` |"
    )
    if not board.get("old_include260617_alignment", {}).get("_missing"):
        lines.append(
            f"| board old default + 260617 | `{board['score_mode']}` | {fmt(board_old_include260617_align.get('n'), 0)} | {fmt(board_old_include260617_align.get('pred_auc_good'))} | {fmt(board_old_include260617_align.get('gt_auc_good'))} | {fmt(board_old_include260617_align.get('pred_gt_spearman'))} | {fmt(board_old_include260617_align.get('pred_score_vs_force_band_quality_spearman'))} | `{summary['paths']['board_old_include260617_alignment']}` |"
        )
    if not board.get("s12_alignment", {}).get("_missing"):
        lines.append(
            f"| board s12 candidate + 260617 | `{board['score_mode']}` | {fmt(board_s12_align.get('n'), 0)} | {fmt(board_s12_align.get('pred_auc_good'))} | {fmt(board_s12_align.get('gt_auc_good'))} | {fmt(board_s12_align.get('pred_gt_spearman'))} | {fmt(board_s12_align.get('pred_score_vs_force_band_quality_spearman'))} | `{summary['paths']['board_s12_alignment']}` |"
        )
    lines.append("")
    lines.append("The board Foresight-chain score is no longer saturated: positive labels score much higher than too-small / too-large / oscillatory contact in `quality` mode.")
    lines.append("")
    lines.append("## Guidance Gradient Evidence")
    lines.append("")
    lines.append("| task | samples | pass | finite grad | positive grad | improved | accept | trust-region | score delta mean | action delta norm mean | evidence |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    lines.append(
        f"| insertion | {fmt(get(insert_grad, 'score_delta', 'n'), 0)} | {fmt(insert_grad.get('pass'))} | {fmt(insert_grad.get('finite_grad_rate_mean'))} | {fmt(insert_grad.get('positive_grad_rate_mean'))} | {fmt(insert_grad.get('improved_rate_mean'))} | {fmt(insert_grad.get('accept_rate_mean'))} | {fmt(insert_grad.get('trust_region_pass_rate'))} | {fmt(get(insert_grad, 'score_delta', 'mean'))} | {fmt(get(insert_grad, 'action_delta_norm', 'mean'))} | `{summary['paths']['insertion_gradient']}` |"
    )
    if not ins.get("gradient_0209", {}).get("_missing"):
        lines.append(
            f"| insertion matched 0209 | {fmt(get(insert_grad_0209, 'score_delta', 'n'), 0)} | {fmt(insert_grad_0209.get('pass'))} | {fmt(insert_grad_0209.get('finite_grad_rate_mean'))} | {fmt(insert_grad_0209.get('positive_grad_rate_mean'))} | {fmt(insert_grad_0209.get('improved_rate_mean'))} | {fmt(insert_grad_0209.get('accept_rate_mean'))} | {fmt(insert_grad_0209.get('trust_region_pass_rate'))} | {fmt(get(insert_grad_0209, 'score_delta', 'mean'))} | {fmt(get(insert_grad_0209, 'action_delta_norm', 'mean'))} | `{summary['paths']['insertion_gradient_0209']}` |"
        )
    if not ins.get("gradient_0401", {}).get("_missing"):
        lines.append(
            f"| insertion matched 0401 | {fmt(get(insert_grad_0401, 'score_delta', 'n'), 0)} | {fmt(insert_grad_0401.get('pass'))} | {fmt(insert_grad_0401.get('finite_grad_rate_mean'))} | {fmt(insert_grad_0401.get('positive_grad_rate_mean'))} | {fmt(insert_grad_0401.get('improved_rate_mean'))} | {fmt(insert_grad_0401.get('accept_rate_mean'))} | {fmt(insert_grad_0401.get('trust_region_pass_rate'))} | {fmt(get(insert_grad_0401, 'score_delta', 'mean'))} | {fmt(get(insert_grad_0401, 'action_delta_norm', 'mean'))} | `{summary['paths']['insertion_gradient_0401']}` |"
        )
    lines.append(
        f"| board | {fmt(get(board_grad, 'score_delta', 'n'), 0)} | {fmt(board_grad.get('pass'))} | {fmt(board_grad.get('finite_grad_rate_mean'))} | {fmt(board_grad.get('positive_grad_rate_mean'))} | {fmt(board_grad.get('improved_rate_mean'))} | {fmt(board_grad.get('accept_rate_mean'))} | {fmt(board_grad.get('trust_region_pass_rate'))} | {fmt(get(board_grad, 'score_delta', 'mean'))} | {fmt(get(board_grad, 'action_delta_norm', 'mean'))} | `{summary['paths']['board_gradient']}` |"
    )
    if not board.get("old_include260617_gradient", {}).get("_missing"):
        lines.append(
            f"| board old default + 260617 | {fmt(get(board_old_include260617_grad, 'score_delta', 'n'), 0)} | {fmt(board_old_include260617_grad.get('pass'))} | {fmt(board_old_include260617_grad.get('finite_grad_rate_mean'))} | {fmt(board_old_include260617_grad.get('positive_grad_rate_mean'))} | {fmt(board_old_include260617_grad.get('improved_rate_mean'))} | {fmt(board_old_include260617_grad.get('accept_rate_mean'))} | {fmt(board_old_include260617_grad.get('trust_region_pass_rate'))} | {fmt(get(board_old_include260617_grad, 'score_delta', 'mean'))} | {fmt(get(board_old_include260617_grad, 'action_delta_norm', 'mean'))} | `{summary['paths']['board_old_include260617_gradient']}` |"
        )
    if not board.get("s12_gradient", {}).get("_missing"):
        lines.append(
            f"| board s12 candidate | {fmt(get(board_s12_grad, 'score_delta', 'n'), 0)} | {fmt(board_s12_grad.get('pass'))} | {fmt(board_s12_grad.get('finite_grad_rate_mean'))} | {fmt(board_s12_grad.get('positive_grad_rate_mean'))} | {fmt(board_s12_grad.get('improved_rate_mean'))} | {fmt(board_s12_grad.get('accept_rate_mean'))} | {fmt(board_s12_grad.get('trust_region_pass_rate'))} | {fmt(get(board_s12_grad, 'score_delta', 'mean'))} | {fmt(get(board_s12_grad, 'action_delta_norm', 'mean'))} | `{summary['paths']['board_s12_gradient']}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Both tasks have finite, non-zero action gradients through Foresight and the scorer.")
    lines.append("- Board uses a deliberately small trust-region step, so score/action deltas are much smaller than insertion.")
    lines.append("- These audits prove differentiability and bounded refinement. They do not prove real robot improvement.")
    lines.append("")
    lines.append("## Noisy-Action Robustness Audit")
    lines.append("")
    lines.append("This audit perturbs recorded action chunks by fractions of the Foresight action standard deviation, then checks whether the scorer/Foresight chain still gives finite positive gradients and locally improves the score.")
    lines.append("")
    lines.append("It is evidence for noisy-action guidance readiness, but it is still not a true DDPM-step guidance benchmark and not real robot evidence.")
    lines.append("")
    lines.append("| task | samples | noise levels(action std) | overall pass | Foresight kind | missing / unexpected keys | per-noise improve/delta | evidence |")
    lines.append("|---|---:|---|---|---|---:|---|---|")
    lines.append(
        f"| insertion | {fmt(insert_noisy.get('n_samples'), 0)} | `{insert_noisy.get('noise_levels_action_std')}` | {fmt(insert_noisy.get('overall_pass'))} | `{get(insert_noisy, 'foresight', 'kind')}` | {fmt(get(insert_noisy, 'foresight', 'missing'), 0)} / {fmt(get(insert_noisy, 'foresight', 'unexpected'), 0)} | {noisy_levels_summary(insert_noisy)} | `{summary['paths']['insertion_noisy_action_audit']}` |"
    )
    if not insert_noisy_0209.get("_missing"):
        lines.append(
            f"| insertion matched 0209 | {fmt(insert_noisy_0209.get('n_samples'), 0)} | `{insert_noisy_0209.get('noise_levels_action_std')}` | {fmt(insert_noisy_0209.get('overall_pass'))} | `{get(insert_noisy_0209, 'foresight', 'kind')}` | {fmt(get(insert_noisy_0209, 'foresight', 'missing'), 0)} / {fmt(get(insert_noisy_0209, 'foresight', 'unexpected'), 0)} | {noisy_levels_summary(insert_noisy_0209)} | `{summary['paths']['insertion_noisy_action_audit_0209']}` |"
        )
    if not insert_noisy_0401.get("_missing"):
        lines.append(
            f"| insertion matched 0401 | {fmt(insert_noisy_0401.get('n_samples'), 0)} | `{insert_noisy_0401.get('noise_levels_action_std')}` | {fmt(insert_noisy_0401.get('overall_pass'))} | `{get(insert_noisy_0401, 'foresight', 'kind')}` | {fmt(get(insert_noisy_0401, 'foresight', 'missing'), 0)} / {fmt(get(insert_noisy_0401, 'foresight', 'unexpected'), 0)} | {noisy_levels_summary(insert_noisy_0401)} | `{summary['paths']['insertion_noisy_action_audit_0401']}` |"
        )
    lines.append(
        f"| board | {fmt(board_noisy.get('n_samples'), 0)} | `{board_noisy.get('noise_levels_action_std')}` | {fmt(board_noisy.get('overall_pass'))} | `{get(board_noisy, 'foresight', 'kind')}` | {fmt(get(board_noisy, 'foresight', 'missing'), 0)} / {fmt(get(board_noisy, 'foresight', 'unexpected'), 0)} | {noisy_levels_summary(board_noisy)} | `{summary['paths']['board_noisy_action_audit']}` |"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Board passes all tested perturbation levels with the deploy-aligned `marker_joint_s12_guided` scorer, but score deltas are intentionally tiny because the trust-region step is small.")
    lines.append("- Insertion now has matched 0209 and 0401 Foresight audits with 0 missing / 0 unexpected keys; the older `latent_foresight_full` audit remains historical caveat evidence only.")
    lines.append("- These results support moving from final clean-action refinement toward denoising-time guidance, but a true DP denoising-step implementation still needs its own audit.")
    lines.append("")
    if not insert_ddpm_sweep.get("_missing"):
        sweep_summary = insert_ddpm_sweep.get("summary", {})
        lines.append("## Insertion DDPM-Step Multi-Episode Sweep")
        lines.append("")
        lines.append("This sweep evaluates matched `latent_foresight_0401` late-step `t=0` guidance across multiple real insertion observations.")
        lines.append("")
        lines.append("| task | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | evidence |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.append(
            f"| insertion | {fmt(sweep_summary.get('n_points'), 0)} | {fmt(sweep_summary.get('n_rows'), 0)} | "
            f"{fmt(sweep_summary.get('final_score_improve_rate'))} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'mean'), 6)} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'min'), 6)} | "
            f"{fmt(get(sweep_summary, 'accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'final_accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'finite_grad_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'guided_action_delta_norm', 'mean'), 6)} | "
            f"`{summary['paths']['insertion_ddpm_step_sweep']}` |"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("")
        lines.append("- Matched insertion Foresight/scorer gradients are finite across all tested rows, and most late-step updates improve the scorer.")
        lines.append("- This protected sweep uses step-level accept-only updates plus final fallback to the base action when the scorer would get worse.")
        lines.append("- With the protected setting, the final score delta minimum is non-negative. It is still offline sampler evidence, not robot outcome evidence.")
        lines.append("")
    if not insert_pgood_ddpm_sweep.get("_missing"):
        sweep_summary = insert_pgood_ddpm_sweep.get("summary", {})
        lines.append("## Insertion p_good DDPM-Step Ablation")
        lines.append("")
        lines.append("This ablation tests the score mode recommended by the semantic direction audit.")
        lines.append("")
        lines.append("| mode | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | evidence |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.append(
            f"| p_good | {fmt(sweep_summary.get('n_points'), 0)} | {fmt(sweep_summary.get('n_rows'), 0)} | "
            f"{fmt(sweep_summary.get('final_score_improve_rate'))} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'mean'), 6)} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'min'), 6)} | "
            f"{fmt(get(sweep_summary, 'accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'final_accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'finite_grad_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'guided_action_delta_norm', 'mean'), 6)} | "
            f"`{summary['paths']['insertion_pgood_ddpm_step_sweep']}` |"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("")
        lines.append("- `p_good` has good offline semantic geometry but saturates in the matched DDPM/Foresight chain: base scores are already near 1.0 and final score deltas are exactly zero.")
        lines.append("- Therefore `p_good` is not recommended as the current insertion DDPM-step guidance score, despite the semantic direction audit.")
        lines.append("- Keep insertion DDPM-step evidence on the protected `profile` sweep unless a less-saturated calibrated score is trained.")
        lines.append("")
    if not insert_score_mode_ablation.get("_missing"):
        ab_summary = insert_score_mode_ablation.get("summary", {})
        lines.append("## Insertion Score-Mode Cross-Score Ablation")
        lines.append("")
        lines.append("This ablation uses each insertion score mode as the DDPM-step guidance objective, then re-scores the same base/guided actions with all candidate heads. This avoids judging a mode only by the score it optimized.")
        lines.append("")
        lines.append("| guidance mode | rows | final accept | action norm | own delta | own improve | profile delta | energy delta | good margin delta | quality logit delta | min quality delta | evidence |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        for mode in ["profile", "p_good", "energy", "good_margin"]:
            mode_summary = ab_summary.get(mode, {})
            cross = mode_summary.get("cross_scores", {})
            own_mode = mode if mode in cross else "profile"
            own = cross.get(own_mode, {})
            quality_delta = get(cross, "quality_logit", "delta", default={})
            lines.append(
                f"| `{mode}` | {fmt(mode_summary.get('n_rows'), 0)} | "
                f"{fmt(mode_summary.get('final_accept_rate'))} | "
                f"{fmt(get(mode_summary, 'guided_action_delta_norm', 'mean'), 6)} | "
                f"{fmt(get(own, 'delta', 'mean'), 6)} | "
                f"{fmt(own.get('improve_rate'))} | "
                f"{fmt(get(cross, 'profile', 'delta', 'mean'), 6)} | "
                f"{fmt(get(cross, 'energy', 'delta', 'mean'), 6)} | "
                f"{fmt(get(cross, 'good_margin', 'delta', 'mean'), 6)} | "
                f"{fmt(get(quality_delta, 'mean'), 6)} | "
                f"{fmt(get(quality_delta, 'min'), 6)} | "
                f"`{summary['paths']['insertion_score_mode_ablation']}` |"
            )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("")
        lines.append("- `p_good` remains saturated: own-score delta is exactly zero under the matched DDPM/Foresight chain.")
        lines.append("- `good_margin` is the strongest unsaturated insertion candidate: it gives the largest own-score gain while keeping `profile`, `energy`, and `quality_logit` non-negative in this protected sweep.")
        lines.append("- This does not replace real robot evidence; it only upgrades the next insertion A/B candidate from bounded probability `p_good` to logit-margin `good_margin`.")
        lines.append("")
    lines.append("## DDPM-Step Guidance Audit")
    lines.append("")
    lines.append("This audit inserts the current board TacQuality scorer into the DP denoising loop and scores the predicted clean action estimate `x0` through Foresight.")
    lines.append("")
    lines.append("| task | setting | samples | final improve | final score delta | per-step score delta | finite grad | action delta norm | evidence |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for audit in board_ddpm_audits:
        audit_summary = audit.get("summary", {}) if isinstance(audit, dict) else {}
        final_delta = get(audit_summary, "final_score_delta", "mean")
        step_delta = get(audit_summary, "per_step_score_delta_mean", "mean")
        finite = get(audit_summary, "finite_grad_rate", "mean")
        action_delta = get(audit_summary, "guided_action_delta_norm", "mean")
        lines.append(
            f"| board | `{ddpm_audit_name(audit)}` | {fmt(audit_summary.get('n_samples'), 0)} | {fmt(audit_summary.get('final_score_improve_rate'))} | {fmt(final_delta, 6)} | {fmt(step_delta, 6)} | {fmt(finite)} | {fmt(action_delta, 6)} | `{audit.get('_source_path', '')}` |"
        )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- The current board scorer has usable gradients inside the sampler, but guidance timing matters.")
    lines.append("- In the 260617 smoke sample, guiding the last two denoising steps reduced final score; guiding only the final `t=0` step produced small positive score gains.")
    lines.append("- Current recommendation: keep production on final clean-action trust-region guidance, and treat true DDPM-step guidance as experimental until a larger sweep confirms late-step-only settings.")
    lines.append("")
    if not board_ddpm_sweep.get("_missing"):
        sweep_summary = board_ddpm_sweep.get("summary", {})
        lines.append("## Board DDPM-Step Multi-Episode Sweep")
        lines.append("")
        lines.append("This sweep reuses one loaded DP/Foresight/scorer stack and evaluates late-step `t=0` guidance across multiple real 260617 board observations.")
        lines.append("")
        lines.append("| task | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | contact gate mean | evidence |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.append(
            f"| board | {fmt(sweep_summary.get('n_points'), 0)} | {fmt(sweep_summary.get('n_rows'), 0)} | "
            f"{fmt(sweep_summary.get('final_score_improve_rate'))} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'mean'), 6)} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'min'), 6)} | "
            f"{fmt(get(sweep_summary, 'accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'final_accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'finite_grad_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'guided_action_delta_norm', 'mean'), 6)} | "
            f"{fmt(get(sweep_summary, 'contact_gate_value', 'mean'))} | "
            f"`{summary['paths']['board_ddpm_step_sweep']}` |"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("")
        lines.append("- The multi-episode sweep is stronger than the single-frame smoke: it covers 6 valid episodes, 12 contact-phase start points, and 24 seed/start rows.")
        lines.append("- This protected sweep uses step-level accept-only updates plus final fallback; all tested rows had finite gradients and positive final score deltas under late-step `t=0` guidance.")
        lines.append("- This supports the scorer as a stable local gradient source, but it is still offline sampler evidence, not real robot improvement.")
        lines.append("")
    if not board_s12_ddpm_sweep.get("_missing"):
        sweep_summary = board_s12_ddpm_sweep.get("summary", {})
        lines.append("## Board s12 DDPM-Step Multi-Episode Sweep")
        lines.append("")
        lines.append("This sweep uses the semantic-direction-favored `marker_joint_s12_guided` board scorer.")
        lines.append("")
        lines.append("| task | eval points | rows | improve | score delta mean | score delta min | step accept | final accept | finite grad | action delta norm | contact gate mean | evidence |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
        lines.append(
            f"| board s12 | {fmt(sweep_summary.get('n_points'), 0)} | {fmt(sweep_summary.get('n_rows'), 0)} | "
            f"{fmt(sweep_summary.get('final_score_improve_rate'))} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'mean'), 6)} | "
            f"{fmt(get(sweep_summary, 'final_score_delta', 'min'), 6)} | "
            f"{fmt(get(sweep_summary, 'accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'final_accept_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'finite_grad_rate', 'mean'))} | "
            f"{fmt(get(sweep_summary, 'guided_action_delta_norm', 'mean'), 6)} | "
            f"{fmt(get(sweep_summary, 'contact_gate_value', 'mean'))} | "
            f"`{summary['paths']['board_s12_ddpm_step_sweep']}` |"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("")
        lines.append("- Board s12 improves every tested row and has larger mean score gain than the old/default board protected sweep, with similar or smaller action update norm.")
        lines.append("- This makes s12 the better next board A/B candidate, but still only offline sampler evidence.")
        lines.append("")
    lines.append("## Server Entrypoint Smoke")
    lines.append("")
    lines.append("| task | pass | scorer runtime | score mode | contact gate | score delta | finite grad | positive grad | accept | evidence |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---|")
    lines.append(
        f"| insertion profile | {fmt(insert_smoke.get('dry_run_guidance_smoke_pass'))} | `{insertion_runtime}` | `{insertion_profile_score_mode}` | NA | "
        f"{fmt(get(insert_smoke, 'report', 'score_delta', 'mean'))} | {fmt(get(insert_smoke, 'report', 'finite_grad_rate'))} | "
        f"{fmt(get(insert_smoke, 'report', 'positive_grad_rate'))} | {fmt(get(insert_smoke, 'report', 'accept_rate'))} | `{summary['paths']['insertion_smoke']}` |"
    )
    if not insert_good_margin_smoke.get("_missing"):
        lines.append(
            f"| insertion good_margin | {fmt(insert_good_margin_smoke.get('dry_run_guidance_smoke_pass'))} | `{insertion_good_margin_runtime}` | `{insertion_good_margin_score_mode}` | NA | "
            f"{fmt(get(insert_good_margin_smoke, 'report', 'score_delta', 'mean'))} | {fmt(get(insert_good_margin_smoke, 'report', 'finite_grad_rate'))} | "
            f"{fmt(get(insert_good_margin_smoke, 'report', 'positive_grad_rate'))} | {fmt(get(insert_good_margin_smoke, 'report', 'accept_rate'))} | `{summary['paths']['insertion_good_margin_smoke']}` |"
        )
    lines.append(
        f"| board | {fmt(board_smoke.get('dry_run_guidance_smoke_pass'))} | `{board_runtime}` | `{board_score_mode}` | {fmt(get(board_smoke, 'contact_gate', 'contact_gate_value'))} | "
        f"{fmt(get(board_smoke, 'report', 'score_delta', 'mean'))} | {fmt(get(board_smoke, 'report', 'finite_grad_rate'))} | "
        f"{fmt(get(board_smoke, 'report', 'positive_grad_rate'))} | {fmt(get(board_smoke, 'report', 'accept_rate'))} | `{summary['paths']['board_smoke']}` |"
    )
    if not board_s12_smoke.get("_missing"):
        lines.append(
            f"| board s12 | {fmt(board_s12_smoke.get('dry_run_guidance_smoke_pass'))} | `{board_s12_runtime}` | `{board_s12_score_mode}` | {fmt(get(board_s12_smoke, 'contact_gate', 'contact_gate_value'))} | "
            f"{fmt(get(board_s12_smoke, 'report', 'score_delta', 'mean'))} | {fmt(get(board_s12_smoke, 'report', 'finite_grad_rate'))} | "
            f"{fmt(get(board_s12_smoke, 'report', 'positive_grad_rate'))} | {fmt(get(board_s12_smoke, 'report', 'accept_rate'))} | `{summary['paths']['board_s12_smoke']}` |"
        )
    lines.append("")
    if not insert_good_margin_smoke.get("_missing"):
        lines.append("")
        lines.append("Insertion good_margin serving command/config:")
        lines.append("")
        lines.append(f"- config: `{summary['paths']['good_margin_rollout_config']}`")
        lines.append("- arm: `good_margin_guided`")
        lines.append("- boundary: dry-run serving smoke only; real insertion success/bounce needs paired robot rollouts.")
        lines.append("")
        lines.append("| shape | value |")
        lines.append("|---|---|")
        lines.append(f"| obs_cond | `{get(insert_good_margin_smoke, 'obs_cond_shape')}` |")
        lines.append(f"| action_norm | `{get(insert_good_margin_smoke, 'action_norm_shape')}` |")
        lines.append(f"| guided_norm | `{get(insert_good_margin_smoke, 'guided_norm_shape')}` |")
        lines.append("")
    if not board_s12_smoke.get("_missing"):
        lines.append("Board s12 serving command/config:")
        lines.append("")
        lines.append(f"- config: `{summary['paths']['rollout_config']}`")
        lines.append("- arm: `marker_joint_s12_guided`")
        lines.append("- boundary: dry-run serving smoke only; real board force improvement needs paired robot rollouts with server-side force traces.")
        lines.append("")
        lines.append("| shape / gate | value |")
        lines.append("|---|---|")
        lines.append(f"| obs_cond | `{get(board_s12_smoke, 'obs_cond_shape')}` |")
        lines.append(f"| action_norm | `{get(board_s12_smoke, 'action_norm_shape')}` |")
        lines.append(f"| guided_norm | `{get(board_s12_smoke, 'guided_norm_shape')}` |")
        lines.append(f"| contact_gate_metric | `{fmt(get(board_s12_smoke, 'contact_gate', 'contact_gate_metric'))}` |")
        lines.append(f"| contact_gate_value | `{fmt(get(board_s12_smoke, 'contact_gate', 'contact_gate_value'))}` |")
        lines.append("")
    lines.append("")
    lines.append("Board contact-gate skip check:")
    lines.append("")
    lines.append("| pass | marker metric | gate value | skipped | raw action delta | evidence |")
    lines.append("|---|---:|---:|---|---:|---|")
    lines.append(
        f"| {fmt(board_gate_skip_smoke.get('dry_run_guidance_smoke_pass'))} | {fmt(get(board_gate_skip_smoke, 'contact_gate', 'contact_gate_metric'))} | {fmt(get(board_gate_skip_smoke, 'contact_gate', 'contact_gate_value'))} | {fmt(get(board_gate_skip_smoke, 'report', 'contact_gate_skipped'))} | {fmt(get(board_gate_skip_smoke, 'report', 'raw_action_delta', 'mean'))} | `{summary['paths']['board_gate_skip_smoke']}` |"
    )
    lines.append("")
    lines.append("## 260617-only Board DP Context")
    lines.append("")
    lines.append(f"- Run: `{summary['dp']['run_dir']}`")
    lines.append(f"- Run status: `{'stopped' if dp_stopped else 'active_or_unknown'}`")
    lines.append(f"- Recommended checkpoint for real tests: `{summary['dp']['recommended_ckpt']}`")
    lines.append(f"- Recommended checkpoint exists: `{fmt(summary['dp']['recommended_ckpt_exists'])}`")
    if dp_stopped:
        stopped_at = dp_stop.get("stopped_at", dp_stop.get("created_at", "NA"))
        requested_epochs = dp_stop.get("requested_epochs", dp_stop.get("total_requested_epochs", dp_latest.get("total")))
        last_epoch = (
            dp_status.get("last_complete_epoch")
            or dp_stop.get("last_complete_epoch")
            or dp_stop.get("stopped_after_epoch")
            or dp_latest.get("epoch")
        )
        last_train = dp_stop.get("last_complete_train", dp_stop.get("last_train_loss"))
        last_val = dp_stop.get("last_complete_val", dp_stop.get("last_val_loss"))
        best_epoch = dp_stop.get("best_epoch", dp_stop.get("best_epoch_reported_or_inferred", dp_best.get("epoch")))
        best_val = dp_stop.get("best_val", dp_stop.get("best_val_loss_reported_by_trainer", dp_best.get("val")))
        lines.append(f"- Stop reason: `{dp_status.get('stop_reason', dp_stop.get('stop_reason', 'NA'))}`")
        lines.append(f"- Stopped at: `{stopped_at}`")
        lines.append(f"- Last complete epoch: `{fmt(last_epoch, 0)}/{fmt(requested_epochs, 0)}`")
        lines.append(f"- Last complete train/val: `{fmt(last_train, 6)}` / `{fmt(last_val, 6)}`")
        lines.append(f"- Best epoch/val: `{fmt(best_epoch, 0)}` / `{fmt(best_val, 6)}`")
        lines.append(f"- Epochs since best: `{fmt(dp_stop.get('epochs_since_best', dp_trend.get('epochs_since_best')), 0)}`")
        lines.append(f"- Early-stop summary: `{summary['dp']['run_dir']}/early_stop_summary.json`")
    else:
        lines.append(f"- Latest epoch: `{fmt(dp_latest.get('epoch'), 0)}/{fmt(dp_latest.get('total'), 0)}`")
        lines.append(f"- Latest train/val: `{fmt(dp_latest.get('train'), 6)}` / `{fmt(dp_latest.get('val'), 6)}`")
        lines.append(f"- Best epoch/val: `{fmt(dp_best.get('epoch'), 0)}` / `{fmt(dp_best.get('val'), 6)}`")
        lines.append(f"- Trend warning: `{dp_trend.get('warning', 'NA')}`")
        lines.append(f"- Epochs since best: `{fmt(dp_trend.get('epochs_since_best'), 0)}`")
    lines.append("")
    lines.append("Deployment/testing should use `dp_best.pth`, not `dp_latest.pth`, unless intentionally testing late-overfit behavior.")
    lines.append("")
    lines.append("## Board Real-Rollout Command Packet")
    lines.append("")
    lines.append("Current copy-paste command sheet:")
    lines.append("")
    lines.append("- `for_show_xiaomi/guide_forshow.sh`")
    lines.append("")
    lines.append("Current board rollout config:")
    lines.append("")
    lines.append(f"- `{summary['paths']['rollout_config']}`")
    lines.append(f"- guided arm: `{CURRENT_BOARD_ARM}`")
    lines.append("- baseline guidance flag: `--disable_guidance`")
    lines.append("- guided scorer runtime: `ForceBandTacQualityEnergyRuntime`")
    lines.append("- guided score mode: `quality`")
    lines.append(f"- expected server-side rollout root: `{CURRENT_BOARD_ROOT}`")
    lines.append("")
    lines.append("Expected real-rollout layout:")
    lines.append("")
    lines.append("```text")
    lines.append(str(CURRENT_BOARD_ROOT) + "/")
    lines.append("  baseline/<trial>/force_trace.csv")
    lines.append("  baseline/<trial>/force_trace.npz")
    lines.append("  baseline/<trial>/force_curve.png")
    lines.append("  baseline/<trial>/metadata.json")
    lines.append("  guided/<trial>/force_trace.csv")
    lines.append("  guided/<trial>/force_trace.npz")
    lines.append("  guided/<trial>/force_curve.png")
    lines.append("  guided/<trial>/metadata.json")
    lines.append("```")
    lines.append("")
    lines.append("After real robot trials, evaluate with:")
    lines.append("")
    lines.append("```bash")
    lines.append("conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \\")
    lines.append(f"  --root {CURRENT_BOARD_ROOT} \\")
    lines.append("  --tag board_260617_marker_joint_s12_scorer \\")
    lines.append("  --expected_baseline_arm baseline \\")
    lines.append(f"  --expected_guided_arm {CURRENT_BOARD_ARM}")
    lines.append("```")
    lines.append("")
    lines.append("## Insertion Real-Rollout Command Packet")
    lines.append("")
    lines.append("Current insertion rollout config:")
    lines.append("")
    lines.append(f"- `{summary['paths']['good_margin_rollout_config']}`")
    lines.append(f"- guided arm: `{CURRENT_INSERTION_ARM}`")
    lines.append("- baseline guidance flag: `--disable_guidance`")
    lines.append("- guided scorer runtime: `InsertionRiskScorerRuntime`")
    lines.append(f"- guided score mode: `{CURRENT_INSERTION_SCORE_MODE}`")
    lines.append(f"- expected server-side rollout root: `{CURRENT_INSERTION_ROOT}`")
    lines.append("")
    lines.append("Expected real-rollout layout:")
    lines.append("")
    lines.append("```text")
    lines.append(str(CURRENT_INSERTION_ROOT) + "/")
    lines.append("  baseline/<trial>/force_trace.csv")
    lines.append("  baseline/<trial>/force_trace.npz")
    lines.append("  baseline/<trial>/force_curve.png")
    lines.append("  baseline/<trial>/metadata.json")
    lines.append("  guided/<trial>/force_trace.csv")
    lines.append("  guided/<trial>/force_trace.npz")
    lines.append("  guided/<trial>/force_curve.png")
    lines.append("  guided/<trial>/metadata.json")
    lines.append("```")
    lines.append("")
    lines.append("After real robot trials, evaluate with:")
    lines.append("")
    lines.append("```bash")
    lines.append("conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_insertion_rollouts.py \\")
    lines.append(f"  --root {CURRENT_INSERTION_ROOT} \\")
    lines.append("  --tag insertion_good_margin_risk_scorer \\")
    lines.append("  --expected_baseline_arm baseline \\")
    lines.append(f"  --expected_guided_arm {CURRENT_INSERTION_ARM}")
    lines.append("```")
    lines.append("")
    lines.append("## Remaining Real-Rollout Evidence Gap")
    lines.append("")
    lines.append(f"- Board real force rollout ready: `{fmt(not get(real, 'board', 'missing_force_trace', default=True))}`")
    lines.append(f"- Insertion real force rollout ready: `{fmt(not get(real, 'insertion', 'missing_force_trace', default=True))}`")
    lines.append(f"- Overall real rollout evidence complete: `{fmt(get(real, 'real_rollout_evidence_complete', default=False))}`")
    lines.append("")
    lines.append("Missing evidence:")
    lines.append("")
    lines.append("- insertion: baseline vs guided real rollouts with success/bounce/retry outcomes;")
    lines.append("- board: matched baseline/guided real rollouts with server-side `force_trace.csv`;")
    lines.append("- board contact-phase metrics: force-in-band ratio, too-low/too-high ratio, force derivative, marker smoothness, and task completion/coverage.")
    lines.append("")
    lines.append("## Current Gates")
    lines.append("")
    lines.append("| gate | status |")
    lines.append("|---|---|")
    lines.append(f"| offline scorer quality | {fmt(conclusion['offline_ready'])} |")
    lines.append(f"| Foresight gradient readiness | {fmt(conclusion['gradient_ready'])} |")
    lines.append(f"| real rollout improvement proven | {fmt(conclusion['real_rollout_proven'])} |")
    lines.append("")
    lines.append("Bottom line: insertion and board scorers are ready for controlled real-rollout testing, but the full project goal is not proven until matched real robot results show improved contact outcomes.")
    lines.append("")
    lines.append("## Source Inputs")
    lines.append("")
    for key, value in summary["paths"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--real_rollout", type=Path, default=DEFAULT_REAL)
    parser.add_argument("--board_train", type=Path, default=DEFAULT_BOARD_TRAIN)
    parser.add_argument("--board_alignment", type=Path, default=DEFAULT_BOARD_ALIGN)
    parser.add_argument("--board_gradient", type=Path, default=DEFAULT_BOARD_GRAD)
    parser.add_argument("--board_s12_train", type=Path, default=DEFAULT_BOARD_S12_TRAIN)
    parser.add_argument("--board_s12_alignment", type=Path, default=DEFAULT_BOARD_S12_ALIGN)
    parser.add_argument("--board_s12_gradient", type=Path, default=DEFAULT_BOARD_S12_GRAD)
    parser.add_argument("--board_old_include260617_alignment", type=Path, default=DEFAULT_BOARD_OLD_INCLUDE260617_ALIGN)
    parser.add_argument("--board_old_include260617_gradient", type=Path, default=DEFAULT_BOARD_OLD_INCLUDE260617_GRAD)
    parser.add_argument("--board_smoke", type=Path, default=DEFAULT_BOARD_SMOKE)
    parser.add_argument("--board_s12_smoke", type=Path, default=DEFAULT_BOARD_S12_SMOKE)
    parser.add_argument("--board_gate_skip_smoke", type=Path, default=DEFAULT_BOARD_GATE_SKIP_SMOKE)
    parser.add_argument("--insertion_eval", type=Path, default=DEFAULT_INSERT_EVAL)
    parser.add_argument("--insertion_gradient", type=Path, default=DEFAULT_INSERT_GRAD)
    parser.add_argument("--insertion_gradient_0209", type=Path, default=DEFAULT_INSERT_GRAD_0209)
    parser.add_argument("--insertion_gradient_0401", type=Path, default=DEFAULT_INSERT_GRAD_0401)
    parser.add_argument("--insertion_smoke", type=Path, default=DEFAULT_INSERT_SMOKE)
    parser.add_argument("--insertion_good_margin_smoke", type=Path, default=DEFAULT_INSERT_GOOD_MARGIN_SMOKE)
    parser.add_argument("--board_noisy_action_audit", type=Path, default=DEFAULT_BOARD_NOISY_ACTION_AUDIT)
    parser.add_argument("--insertion_noisy_action_audit", type=Path, default=DEFAULT_INSERT_NOISY_ACTION_AUDIT)
    parser.add_argument("--insertion_noisy_action_audit_0209", type=Path, default=DEFAULT_INSERT_NOISY_ACTION_AUDIT_0209)
    parser.add_argument("--insertion_noisy_action_audit_0401", type=Path, default=DEFAULT_INSERT_NOISY_ACTION_AUDIT_0401)
    parser.add_argument("--insertion_ddpm_step_sweep", type=Path, default=DEFAULT_INSERT_DDPM_SWEEP)
    parser.add_argument("--insertion_pgood_ddpm_step_sweep", type=Path, default=DEFAULT_INSERT_PGOOD_DDPM_SWEEP)
    parser.add_argument("--insertion_score_mode_ablation", type=Path, default=DEFAULT_INSERT_SCORE_MODE_ABLATION)
    parser.add_argument("--board_ddpm_step_audits", type=Path, nargs="*", default=DEFAULT_BOARD_DDPM_AUDITS)
    parser.add_argument("--board_ddpm_step_sweep", type=Path, default=DEFAULT_BOARD_DDPM_SWEEP)
    parser.add_argument("--board_s12_ddpm_step_sweep", type=Path, default=DEFAULT_BOARD_S12_DDPM_SWEEP)
    parser.add_argument("--semantic_direction", type=Path, default=DEFAULT_SEMANTIC_DIRECTION)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
    parser.add_argument("--good_margin_rollout_config", type=Path, default=DEFAULT_GOOD_MARGIN_ROLLOUT_CONFIG)
    parser.add_argument("--dp_run", type=Path, default=DEFAULT_DP_RUN)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    summary = build_summary(args)
    md = render_md(summary)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md, encoding="utf-8")
    args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "output_md": str(args.output_md),
        "output_json": str(args.output_json),
        "real_rollout_proven": summary["conclusion"]["real_rollout_proven"],
        "dp_warning": get(summary, "dp", "status", "trend", "warning", default=None),
    }, indent=2))


if __name__ == "__main__":
    main()
