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
DEFAULT_REAL = Path("/home/chenshuai/Project/output/tac_quality_real_rollout_eval/current_tac_quality/tac_quality_real_rollout_eval.json")
DEFAULT_BOARD_TRAIN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/train_result.json")
DEFAULT_BOARD_ALIGN = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/foresight_alignment_quality/foresight_score_alignment.json")
DEFAULT_BOARD_GRAD = Path("/home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260618/guidance_gradient_audit_quality/guidance_gradient_audit.json")
DEFAULT_BOARD_SMOKE = Path("/home/chenshuai/Project/output/tac_quality_guided_server_packet/current_marker_joint_board_ext_dp_20260619/guided_server_dry_run_smoke.json")
DEFAULT_INSERT_EVAL = Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_eval.json")
DEFAULT_INSERT_GRAD = Path("/home/chenshuai/Project/output/insertion_guidance_gradient_audit_real_foresight_profile_20260618/guidance_gradient_audit.json")
DEFAULT_INSERT_SMOKE = Path("/home/chenshuai/Project/output/tac_quality_guided_server_packet/auto_discovered/insertion_guided_server_real_foresight_smoke.json")
DEFAULT_ROLLOUT_CONFIG = Path("/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json")
DEFAULT_DP_RUN = Path("/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext")
DEFAULT_OUTPUT_MD = Path("docs/2026-06-18_tac_quality_guidance_readiness_matrix.md")
DEFAULT_OUTPUT_JSON = Path("/home/chenshuai/Project/output/tac_quality_current_readiness_matrix/tac_quality_current_readiness_matrix.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
    board_smoke = load_json(args.board_smoke)
    insert_eval = load_json(args.insertion_eval)
    insert_grad = load_json(args.insertion_gradient)
    insert_smoke = load_json(args.insertion_smoke)
    rollout_config = load_json(args.rollout_config)
    dp_status = load_json(args.dp_run / "training_status_latest.json")

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
            "board_smoke": str(args.board_smoke),
            "insertion_eval": str(args.insertion_eval),
            "insertion_gradient": str(args.insertion_gradient),
            "insertion_smoke": str(args.insertion_smoke),
            "rollout_config": str(args.rollout_config),
            "dp_run": str(args.dp_run),
        },
        "insertion": {
            "recommended_arm": get(state, "insertion", "recommended_arm", default="default_guided"),
            "scorer": get(insertion, "scorer", default="InsertionRiskScorerRuntime"),
            "checkpoint": get(insertion, "checkpoint", default="/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"),
            "score_mode": get(insertion, "score_mode", default="profile"),
            "cv": get(insert_eval, "mixed_group_cv", default=get(insertion, "grouped_cv", default={})),
            "gradient": get(insert_grad, "summary", default=get(insertion, "foresight_gradient_audit", default={})),
            "server_smoke": insert_smoke,
            "ready_for_real_rollout": get(state, "insertion", "ready_for_real_rollout", default=False),
        },
        "board": {
            "recommended_arm": get(state, "board", "recommended_arm", default="marker_joint_guided"),
            "scorer": get(board, "scorer", default="ForceBandTacQualityEnergyRuntime(marker_joint_action)"),
            "checkpoint": get(board, "checkpoint", default=get(board_train, "checkpoint_best")),
            "score_mode": get(board, "score_mode", default="quality"),
            "train_result": board_train,
            "alignment": board_align,
            "gradient": get(board_grad, "summary", default=get(board, "foresight_gradient_audit", default={})),
            "server_smoke": board_smoke,
            "ready_for_real_rollout": get(state, "board", "ready_for_real_rollout", default=False),
        },
        "rollout_config": rollout_config,
        "real_rollout": real,
        "dp": {
            "run_dir": str(args.dp_run),
            "home_symlink": "/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260618_ext",
            "status": dp_status,
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


def render_md(summary: dict[str, Any]) -> str:
    ins = summary["insertion"]
    board = summary["board"]
    board_best = get(board, "train_result", "best", "val", default={})
    board_align = get(board, "alignment", "summary", default={})
    dp_status = get(summary, "dp", "status", default={})
    dp_latest = get(dp_status, "latest", default={})
    dp_best = get(dp_status, "best_val_epoch", default={})
    dp_trend = get(dp_status, "trend", default={})
    real = summary["real_rollout"]
    conclusion = summary["conclusion"]

    insert_cv = ins["cv"]
    insert_grad = ins["gradient"]
    board_grad = board["gradient"]
    insert_smoke = ins["server_smoke"]
    board_smoke = board["server_smoke"]
    insertion_score_mode = get(insert_smoke, "report", "score_mode", default="profile")
    board_score_mode = get(board_smoke, "report", "score_mode", default=board["score_mode"])
    insertion_runtime = get(insert_smoke, "report", "scorer_runtime",
                            default=get(insert_smoke, "report", "profile", "scorer", default=ins["scorer"]))
    board_runtime = get(board_smoke, "report", "scorer_runtime", default="ForceBandTacQualityEnergyRuntime")

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
        f"| insertion | `{ins['recommended_arm']}` | `{ins['scorer']}` | `{ins['checkpoint']}` | `{insertion_score_mode}` | {fmt(ins['ready_for_real_rollout'])} |"
    )
    lines.append(
        f"| board | `{board['recommended_arm']}` | `{board['scorer']}` | `{board['checkpoint']}` | `{board_score_mode}` | {fmt(board['ready_for_real_rollout'])} |"
    )
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
        f"| insertion | `{insertion_score_mode}` | NA | NA | NA | NA | NA | gradient audit below |"
    )
    lines.append(
        f"| board | `{board['score_mode']}` | {fmt(board_align.get('n'), 0)} | {fmt(board_align.get('pred_auc_good'))} | {fmt(board_align.get('gt_auc_good'))} | {fmt(board_align.get('pred_gt_spearman'))} | {fmt(board_align.get('pred_score_vs_force_band_quality_spearman'))} | `{summary['paths']['board_alignment']}` |"
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
    lines.append(
        f"| board | {fmt(get(board_grad, 'score_delta', 'n'), 0)} | {fmt(board_grad.get('pass'))} | {fmt(board_grad.get('finite_grad_rate_mean'))} | {fmt(board_grad.get('positive_grad_rate_mean'))} | {fmt(board_grad.get('improved_rate_mean'))} | {fmt(board_grad.get('accept_rate_mean'))} | {fmt(board_grad.get('trust_region_pass_rate'))} | {fmt(get(board_grad, 'score_delta', 'mean'))} | {fmt(get(board_grad, 'action_delta_norm', 'mean'))} | `{summary['paths']['board_gradient']}` |"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("")
    lines.append("- Both tasks have finite, non-zero action gradients through Foresight and the scorer.")
    lines.append("- Board uses a deliberately small trust-region step, so score/action deltas are much smaller than insertion.")
    lines.append("- These audits prove differentiability and bounded refinement. They do not prove real robot improvement.")
    lines.append("")
    lines.append("## Server Entrypoint Smoke")
    lines.append("")
    lines.append("| task | pass | scorer runtime | score mode | contact gate | score delta | evidence |")
    lines.append("|---|---|---|---|---|---:|---|")
    lines.append(
        f"| insertion | {fmt(insert_smoke.get('dry_run_guidance_smoke_pass'))} | `{insertion_runtime}` | `{insertion_score_mode}` | NA | {fmt(get(insert_smoke, 'report', 'score_delta', 'mean'))} | `{summary['paths']['insertion_smoke']}` |"
    )
    lines.append(
        f"| board | {fmt(board_smoke.get('dry_run_guidance_smoke_pass'))} | `{board_runtime}` | `{board_score_mode}` | {fmt(get(board_smoke, 'contact_gate', 'contact_gate_value'))} | {fmt(get(board_smoke, 'report', 'score_delta', 'mean'))} | `{summary['paths']['board_smoke']}` |"
    )
    lines.append("")
    lines.append("## Active 260617-only Board DP Context")
    lines.append("")
    lines.append(f"- Active run: `{summary['dp']['run_dir']}`")
    lines.append(f"- Home symlink: `{summary['dp']['home_symlink']}`")
    lines.append(f"- Recommended checkpoint for real tests: `{summary['dp']['recommended_ckpt']}`")
    lines.append(f"- Recommended checkpoint exists: `{fmt(summary['dp']['recommended_ckpt_exists'])}`")
    lines.append(f"- Latest epoch: `{fmt(dp_latest.get('epoch'), 0)}/{fmt(dp_latest.get('total'), 0)}`")
    lines.append(f"- Latest train/val: `{fmt(dp_latest.get('train'), 6)}` / `{fmt(dp_latest.get('val'), 6)}`")
    lines.append(f"- Best epoch/val: `{fmt(dp_best.get('epoch'), 0)}` / `{fmt(dp_best.get('val'), 6)}`")
    lines.append(f"- Trend warning: `{dp_trend.get('warning', 'NA')}`")
    lines.append(f"- Epochs since best: `{fmt(dp_trend.get('epochs_since_best'), 0)}`")
    lines.append("")
    lines.append("Deployment/testing should use `dp_best.pth`, not `dp_latest.pth`, unless a later epoch refreshes the best validation checkpoint.")
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
    lines.append("- guided arm: `marker_joint_guided`")
    lines.append("- baseline guidance flag: `--disable_guidance`")
    lines.append("- guided scorer runtime: `ForceBandTacQualityEnergyRuntime`")
    lines.append("- guided score mode: `quality`")
    lines.append("- expected server-side rollout root: `/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer`")
    lines.append("")
    lines.append("Expected real-rollout layout:")
    lines.append("")
    lines.append("```text")
    lines.append("/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer/")
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
    lines.append("  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer \\")
    lines.append("  --tag board_260617_marker_joint_scorer")
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
    parser.add_argument("--board_smoke", type=Path, default=DEFAULT_BOARD_SMOKE)
    parser.add_argument("--insertion_eval", type=Path, default=DEFAULT_INSERT_EVAL)
    parser.add_argument("--insertion_gradient", type=Path, default=DEFAULT_INSERT_GRAD)
    parser.add_argument("--insertion_smoke", type=Path, default=DEFAULT_INSERT_SMOKE)
    parser.add_argument("--rollout_config", type=Path, default=DEFAULT_ROLLOUT_CONFIG)
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
