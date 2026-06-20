#!/usr/bin/env python3
"""Run and summarize real-rollout evaluators for TacQuality guidance.

This is the final evidence aggregator for the current TacQuality guidance
story.  It does not claim task improvement by itself; it reports whether the
server-side baseline/guided rollout logs contain enough evidence to support a
real robot comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BOARD_ROOT = "/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer"
DEFAULT_INSERTION_ROOT = "/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer"
DEFAULT_OUTPUT_DIR = "/home/chenshuai/Project/output/tac_quality_real_rollout_eval"
DEFAULT_BOARD_BASELINE_ARM = "baseline"
DEFAULT_BOARD_GUIDED_ARM = "marker_joint_s12_guided"
DEFAULT_INSERTION_BASELINE_ARM = "baseline"
DEFAULT_INSERTION_GUIDED_ARM = "good_margin_guided"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_cmd(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode == 0, proc.stdout


def group_n(result: dict[str, Any], name: str) -> int:
    return int(result.get("group_summary", {}).get(name, {}).get("n_trials", 0) or 0)


def metric_mean(result: dict[str, Any], group: str, metric: str) -> float | None:
    item = result.get("group_summary", {}).get(group, {}).get(metric)
    if not isinstance(item, dict) or "mean" not in item:
        return None
    try:
        return float(item["mean"])
    except (TypeError, ValueError):
        return None


def delta(result: dict[str, Any], metric: str, *, guided_minus_baseline: bool = True) -> float | None:
    guided = metric_mean(result, "guided", metric)
    baseline = metric_mean(result, "baseline", metric)
    if guided is None or baseline is None:
        return None
    return guided - baseline if guided_minus_baseline else baseline - guided


def coverage_pair(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "baseline_trials": group_n(result, "baseline"),
        "guided_trials": group_n(result, "guided"),
        "has_baseline_and_guided": group_n(result, "baseline") > 0 and group_n(result, "guided") > 0,
    }


def row_group(row: dict[str, Any]) -> str:
    group = str(row.get("group") or "").lower()
    if group:
        return group
    trial = str(row.get("trial_dir") or "").lower()
    if "baseline" in trial:
        return "baseline"
    if "guided" in trial:
        return "guided"
    return "unknown"


def pair_key(row: dict[str, Any], idx: int) -> str:
    for key in ["pair_id", "trial", "episode"]:
        value = row.get(key)
        if value not in {None, "", "nan"}:
            return f"{key}:{value}"
    return f"order:{idx:04d}"


def paired_rows(rows: list[dict[str, Any]]) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], str]:
    baseline = [row for row in rows if row_group(row) == "baseline"]
    guided = [row for row in rows if row_group(row) == "guided"]
    baseline = sorted(baseline, key=lambda r: str(r.get("trial_dir", "")))
    guided = sorted(guided, key=lambda r: str(r.get("trial_dir", "")))
    if not baseline or not guided:
        return [], "missing_baseline_or_guided"

    explicit = any(row.get("pair_id") not in {None, "", "nan"} for row in baseline + guided)
    if explicit:
        bmap = {str(row.get("pair_id")): row for row in baseline if row.get("pair_id") not in {None, "", "nan"}}
        gmap = {str(row.get("pair_id")): row for row in guided if row.get("pair_id") not in {None, "", "nan"}}
        keys = sorted(set(bmap) & set(gmap))
        return [(bmap[k], gmap[k]) for k in keys], "explicit_pair_id"

    n = min(len(baseline), len(guided))
    if len(baseline) != len(guided):
        return [(baseline[i], guided[i]) for i in range(n)], "order_pair_truncated"
    return [(baseline[i], guided[i]) for i in range(n)], "order_pair"


def mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def paired_metric_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    metric: str,
    *,
    guided_minus_baseline: bool = True,
) -> dict[str, Any]:
    deltas = []
    details = []
    for i, (base, guided) in enumerate(pairs):
        if metric not in base or metric not in guided:
            continue
        try:
            b = float(base[metric])
            g = float(guided[metric])
        except (TypeError, ValueError):
            continue
        d = g - b if guided_minus_baseline else b - g
        deltas.append(d)
        details.append({
            "pair_index": i,
            "baseline_trial": base.get("trial_dir"),
            "guided_trial": guided.get("trial_dir"),
            "baseline": b,
            "guided": g,
            "delta": d,
        })
    return {"n": len(deltas), "mean": mean(deltas), "deltas": deltas, "details": details}


def load_trial_metadata(row: dict[str, Any]) -> dict[str, Any]:
    trial_dir = row.get("trial_dir")
    if not trial_dir:
        return {}
    path = Path(str(trial_dir)) / "metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "synthetic"}


def row_is_synthetic(row: dict[str, Any]) -> bool:
    meta = load_trial_metadata(row)
    server_meta = meta.get("server_metadata") if isinstance(meta.get("server_metadata"), dict) else {}
    trial_text = str(row.get("trial_dir", "")).lower()
    return bool(
        boolish(meta.get("synthetic_smoke"))
        or boolish(meta.get("not_real_robot_evidence"))
        or str(meta.get("log_side", "")).lower() == "synthetic"
        or boolish(server_meta.get("synthetic_smoke"))
        or "synthetic" in str(server_meta.get("protocol", "")).lower()
        or "/synthetic_" in trial_text
        or "synthetic_pair" in trial_text
    )


def synthetic_summary(result: dict[str, Any] | None) -> dict[str, Any]:
    rows = result.get("rows", []) if isinstance(result, dict) else []
    synthetic_rows = [row for row in rows if row_is_synthetic(row)]
    return {
        "contains_synthetic": bool(synthetic_rows),
        "synthetic_trial_count": int(len(synthetic_rows)),
        "total_trial_count": int(len(rows)),
        "synthetic_examples": [row.get("trial_dir") for row in synthetic_rows[:5]],
    }


def metric_pass(item: dict[str, Any] | None, *, min_n: int, min_mean: float) -> bool:
    if not isinstance(item, dict):
        return False
    try:
        n = int(item.get("n", 0) or 0)
        value = item.get("mean")
        if value is None:
            return False
        return n >= int(min_n) and float(value) >= float(min_mean)
    except (TypeError, ValueError):
        return False


def build_acceptance(
    *,
    task: str,
    paired: dict[str, Any],
    min_pairs: int,
    metric_thresholds: dict[str, float],
    extra_ready: bool = True,
) -> dict[str, Any]:
    n_pairs = int(paired.get("n_pairs", 0) or 0)
    complete_pair_count = bool(paired.get("complete_pair_count"))
    metric_checks = {}
    for metric, threshold in metric_thresholds.items():
        item = paired.get(metric)
        metric_checks[metric] = {
            "n": item.get("n") if isinstance(item, dict) else 0,
            "mean": item.get("mean") if isinstance(item, dict) else None,
            "min_mean": float(threshold),
            "pass": metric_pass(item, min_n=min_pairs, min_mean=float(threshold)),
        }
    any_metric_pass = any(check["pass"] for check in metric_checks.values())
    passed = bool(n_pairs >= min_pairs and complete_pair_count and extra_ready and any_metric_pass)
    return {
        "task": task,
        "pass": passed,
        "min_pairs": int(min_pairs),
        "n_pairs": n_pairs,
        "complete_pair_count": complete_pair_count,
        "extra_ready": bool(extra_ready),
        "any_metric_pass": bool(any_metric_pass),
        "metric_checks": metric_checks,
        "note": "At least one paired task metric must improve in the expected direction.",
    }


def board_paired_summary(result: dict[str, Any]) -> dict[str, Any]:
    pairs, method = paired_rows(result.get("rows", []))
    return {
        "method": method,
        "n_pairs": len(pairs),
        "complete_pair_count": bool(pairs) and method in {"explicit_pair_id", "order_pair"},
        "quality_force_in_band_guided_minus_baseline": paired_metric_delta(
            pairs, "quality_force_in_band_ratio"
        ),
        "quality_force_smooth_guided_minus_baseline": paired_metric_delta(
            pairs, "quality_force_smooth_score"
        ),
        "quality_force_abs_error_baseline_minus_guided": paired_metric_delta(
            pairs, "quality_force_abs_error_mean", guided_minus_baseline=False
        ),
    }


def insertion_paired_summary(result: dict[str, Any]) -> dict[str, Any]:
    pairs, method = paired_rows(result.get("rows", []))
    return {
        "method": method,
        "n_pairs": len(pairs),
        "complete_pair_count": bool(pairs) and method in {"explicit_pair_id", "order_pair"},
        "success_guided_minus_baseline": paired_metric_delta(pairs, "success"),
        "bounce_baseline_minus_guided": paired_metric_delta(
            pairs, "bounce_count", guided_minus_baseline=False
        ),
        "retry_baseline_minus_guided": paired_metric_delta(
            pairs, "retry_count", guided_minus_baseline=False
        ),
    }


def summarize_board_with_thresholds(
    result: dict[str, Any] | None,
    ok: bool,
    output: str,
    *,
    min_pairs: int,
    in_band_min: float,
    smooth_min: float,
    abs_error_min: float,
    allow_synthetic: bool,
) -> dict[str, Any]:
    if not ok or result is None:
        missing = "No force_trace.csv" in output
        return {
            "evaluator_ok": False,
            "missing_force_trace": bool(missing),
            "baseline_trials": 0,
            "guided_trials": 0,
            "has_baseline_and_guided": False,
            "paired_summary": {
                "method": "missing_force_trace" if missing else "evaluator_failed",
                "n_pairs": 0,
                "complete_pair_count": False,
            },
            "data_pairing_ready": False,
            "acceptance": build_acceptance(
                task="board",
                paired={"n_pairs": 0, "complete_pair_count": False},
                min_pairs=min_pairs,
                metric_thresholds={
                    "quality_force_in_band_guided_minus_baseline": in_band_min,
                    "quality_force_smooth_guided_minus_baseline": smooth_min,
                    "quality_force_abs_error_baseline_minus_guided": abs_error_min,
                },
                extra_ready=False,
            ),
            "real_comparison_ready": False,
            "detail": "No board force_trace.csv found yet; run baseline/guided robot tests first."
            if missing else "board evaluator failed",
            "error_output": output[-4000:],
        }
    coverage = coverage_pair(result)
    synthetic = synthetic_summary(result)
    force_delta = delta(result, "quality_force_in_band_ratio")
    smooth_delta = delta(result, "quality_force_smooth_score")
    error_delta = delta(result, "quality_force_abs_error_mean", guided_minus_baseline=False)
    paired = board_paired_summary(result)
    synthetic_allowed_for_acceptance = bool(allow_synthetic or not synthetic["contains_synthetic"])
    acceptance = build_acceptance(
        task="board",
        paired=paired,
        min_pairs=min_pairs,
        metric_thresholds={
            "quality_force_in_band_guided_minus_baseline": in_band_min,
            "quality_force_smooth_guided_minus_baseline": smooth_min,
            "quality_force_abs_error_baseline_minus_guided": abs_error_min,
        },
        extra_ready=synthetic_allowed_for_acceptance,
    )
    if synthetic["contains_synthetic"]:
        detail = (
            "Synthetic rollout logs detected; accepted only for pipeline smoke, not real robot evidence."
            if allow_synthetic else
            "Synthetic rollout logs detected; refusing to count them as real robot evidence."
        )
    else:
        detail = None
    return {
        "evaluator_ok": True,
        **coverage,
        **synthetic,
        "allow_synthetic": bool(allow_synthetic),
        "summary_json": result.get("summary_json"),
        "summary_md": result.get("summary_md"),
        "overview_plot": result.get("overview_plot"),
        "group_curves_plot": result.get("group_curves_plot"),
        "quality_force_in_band_guided_minus_baseline": force_delta,
        "quality_force_smooth_guided_minus_baseline": smooth_delta,
        "quality_force_abs_error_baseline_minus_guided": error_delta,
        "paired_summary": paired,
        "data_pairing_ready": bool(
            coverage["has_baseline_and_guided"] and paired.get("complete_pair_count")
        ),
        "acceptance": acceptance,
        "real_comparison_ready": bool(acceptance["pass"] and not synthetic["contains_synthetic"]),
        "detail": detail,
    }


def summarize_insertion_with_thresholds(
    result: dict[str, Any] | None,
    ok: bool,
    output: str,
    *,
    min_pairs: int,
    success_min: float,
    bounce_min: float,
    retry_min: float,
    allow_synthetic: bool,
) -> dict[str, Any]:
    if not ok or result is None:
        missing = "No force_trace.csv" in output
        return {
            "evaluator_ok": False,
            "missing_force_trace": bool(missing),
            "baseline_trials": 0,
            "guided_trials": 0,
            "has_baseline_and_guided": False,
            "metadata_success_and_stopped_early_complete": False,
            "paired_summary": {
                "method": "missing_force_trace" if missing else "evaluator_failed",
                "n_pairs": 0,
                "complete_pair_count": False,
            },
            "data_pairing_ready": False,
            "acceptance": build_acceptance(
                task="insertion",
                paired={"n_pairs": 0, "complete_pair_count": False},
                min_pairs=min_pairs,
                metric_thresholds={
                    "success_guided_minus_baseline": success_min,
                    "bounce_baseline_minus_guided": bounce_min,
                    "retry_baseline_minus_guided": retry_min,
                },
                extra_ready=False,
            ),
            "real_comparison_ready": False,
            "detail": "No insertion force_trace.csv found yet; run baseline/guided robot tests first."
            if missing else "insertion evaluator failed",
            "error_output": output[-4000:],
        }
    coverage = coverage_pair(result)
    synthetic = synthetic_summary(result)
    meta = result.get("metadata_coverage", {})
    success_delta = delta(result, "success")
    bounce_delta = delta(result, "bounce_count", guided_minus_baseline=False)
    retry_delta = delta(result, "retry_count", guided_minus_baseline=False)
    paired = insertion_paired_summary(result)
    metadata_ready = bool(meta.get("success_and_stopped_early_complete"))
    synthetic_allowed_for_acceptance = bool(allow_synthetic or not synthetic["contains_synthetic"])
    acceptance = build_acceptance(
        task="insertion",
        paired=paired,
        min_pairs=min_pairs,
        metric_thresholds={
            "success_guided_minus_baseline": success_min,
            "bounce_baseline_minus_guided": bounce_min,
            "retry_baseline_minus_guided": retry_min,
        },
        extra_ready=bool(metadata_ready and synthetic_allowed_for_acceptance),
    )
    if synthetic["contains_synthetic"]:
        detail = (
            "Synthetic rollout logs detected; accepted only for pipeline smoke, not real robot evidence."
            if allow_synthetic else
            "Synthetic rollout logs detected; refusing to count them as real robot evidence."
        )
    else:
        detail = None
    return {
        "evaluator_ok": True,
        **coverage,
        **synthetic,
        "allow_synthetic": bool(allow_synthetic),
        "summary_json": str(Path(result.get("summary_csv", "")).with_name("insertion_rollout_summary.json")),
        "summary_md": str(Path(result.get("summary_csv", "")).with_name("insertion_rollout_summary.md")),
        "overview_plot": result.get("overview_plot"),
        "metadata_template_csv": result.get("metadata_template_csv"),
        "metadata_success_and_stopped_early_complete": metadata_ready,
        "success_guided_minus_baseline": success_delta,
        "bounce_baseline_minus_guided": bounce_delta,
        "retry_baseline_minus_guided": retry_delta,
        "paired_summary": paired,
        "data_pairing_ready": bool(
            coverage["has_baseline_and_guided"] and metadata_ready and paired.get("complete_pair_count")
        ),
        "acceptance": acceptance,
        "real_comparison_ready": bool(acceptance["pass"] and not synthetic["contains_synthetic"]),
        "detail": detail,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    board = result["board"]
    insertion = result["insertion"]
    cfg = result.get("acceptance_config", {})
    lines = [
        "# TacQuality Real Rollout Evaluation",
        "",
        "This report aggregates server-side real-rollout evaluators for the current TacQuality guidance candidates.",
        "It separates readiness from real robot evidence: no baseline/guided rollout pair means no performance claim.",
        "",
        "## Overall",
        "",
        f"- real_rollout_evidence_complete: `{result['real_rollout_evidence_complete']}`",
        f"- board_real_comparison_ready: `{board.get('real_comparison_ready')}`",
        f"- insertion_real_comparison_ready: `{insertion.get('real_comparison_ready')}`",
        f"- board_acceptance_pass: `{(board.get('acceptance') or {}).get('pass')}`",
        f"- insertion_acceptance_pass: `{(insertion.get('acceptance') or {}).get('pass')}`",
        f"- allow_synthetic_smoke: `{result.get('allow_synthetic_smoke')}`",
        f"- min_board_pairs: `{cfg.get('min_board_pairs')}`",
        f"- min_insertion_pairs: `{cfg.get('min_insertion_pairs')}`",
        "",
        "## Board",
        "",
        f"- evaluator_ok: `{board.get('evaluator_ok')}`",
        f"- detail: `{board.get('detail')}`",
        f"- contains_synthetic: `{board.get('contains_synthetic')}`",
        f"- synthetic_trial_count: `{board.get('synthetic_trial_count')}`",
        f"- baseline_trials: `{board.get('baseline_trials', 0)}`",
        f"- guided_trials: `{board.get('guided_trials', 0)}`",
        f"- summary_md: `{board.get('summary_md')}`",
        f"- overview_plot: `{board.get('overview_plot')}`",
        f"- force in-band guided-baseline: `{fmt(board.get('quality_force_in_band_guided_minus_baseline'))}`",
        f"- force smooth guided-baseline: `{fmt(board.get('quality_force_smooth_guided_minus_baseline'))}`",
        f"- force abs-error baseline-guided: `{fmt(board.get('quality_force_abs_error_baseline_minus_guided'))}`",
        f"- paired method: `{(board.get('paired_summary') or {}).get('method')}`",
        f"- paired n: `{(board.get('paired_summary') or {}).get('n_pairs')}`",
        f"- acceptance: `{(board.get('acceptance') or {}).get('pass')}`",
        "",
        "## Insertion",
        "",
        f"- evaluator_ok: `{insertion.get('evaluator_ok')}`",
        f"- detail: `{insertion.get('detail')}`",
        f"- contains_synthetic: `{insertion.get('contains_synthetic')}`",
        f"- synthetic_trial_count: `{insertion.get('synthetic_trial_count')}`",
        f"- baseline_trials: `{insertion.get('baseline_trials', 0)}`",
        f"- guided_trials: `{insertion.get('guided_trials', 0)}`",
        f"- metadata complete: `{insertion.get('metadata_success_and_stopped_early_complete')}`",
        f"- summary_md: `{insertion.get('summary_md')}`",
        f"- metadata_template_csv: `{insertion.get('metadata_template_csv')}`",
        f"- success guided-baseline: `{fmt(insertion.get('success_guided_minus_baseline'))}`",
        f"- bounce baseline-guided: `{fmt(insertion.get('bounce_baseline_minus_guided'))}`",
        f"- retry baseline-guided: `{fmt(insertion.get('retry_baseline_minus_guided'))}`",
        f"- paired method: `{(insertion.get('paired_summary') or {}).get('method')}`",
        f"- paired n: `{(insertion.get('paired_summary') or {}).get('n_pairs')}`",
        f"- acceptance: `{(insertion.get('acceptance') or {}).get('pass')}`",
        "",
        "## Paired Summary",
        "",
        "| task | method | n | key paired deltas |",
        "|---|---|---:|---|",
        f"| board | `{(board.get('paired_summary') or {}).get('method')}` | "
        f"{(board.get('paired_summary') or {}).get('n_pairs')} | "
        f"in_band={fmt(((board.get('paired_summary') or {}).get('quality_force_in_band_guided_minus_baseline') or {}).get('mean'))}, "
        f"smooth={fmt(((board.get('paired_summary') or {}).get('quality_force_smooth_guided_minus_baseline') or {}).get('mean'))}, "
        f"abs_error={fmt(((board.get('paired_summary') or {}).get('quality_force_abs_error_baseline_minus_guided') or {}).get('mean'))} |",
        f"| insertion | `{(insertion.get('paired_summary') or {}).get('method')}` | "
        f"{(insertion.get('paired_summary') or {}).get('n_pairs')} | "
        f"success={fmt(((insertion.get('paired_summary') or {}).get('success_guided_minus_baseline') or {}).get('mean'))}, "
        f"bounce={fmt(((insertion.get('paired_summary') or {}).get('bounce_baseline_minus_guided') or {}).get('mean'))}, "
        f"retry={fmt(((insertion.get('paired_summary') or {}).get('retry_baseline_minus_guided') or {}).get('mean'))} |",
        "",
        "## Acceptance Gate",
        "",
        "| task | pass | min pairs | n pairs | metric pass |",
        "|---|---:|---:|---:|---:|",
        f"| board | `{(board.get('acceptance') or {}).get('pass')}` | "
        f"{(board.get('acceptance') or {}).get('min_pairs')} | "
        f"{(board.get('acceptance') or {}).get('n_pairs')} | "
        f"`{(board.get('acceptance') or {}).get('any_metric_pass')}` |",
        f"| insertion | `{(insertion.get('acceptance') or {}).get('pass')}` | "
        f"{(insertion.get('acceptance') or {}).get('min_pairs')} | "
        f"{(insertion.get('acceptance') or {}).get('n_pairs')} | "
        f"`{(insertion.get('acceptance') or {}).get('any_metric_pass')}` |",
        "",
        "Acceptance thresholds:",
        "",
        f"- board_in_band_min_delta: `{cfg.get('board_in_band_min_delta')}`",
        f"- board_smooth_min_delta: `{cfg.get('board_smooth_min_delta')}`",
        f"- board_abs_error_min_delta: `{cfg.get('board_abs_error_min_delta')}`",
        f"- insertion_success_min_delta: `{cfg.get('insertion_success_min_delta')}`",
        f"- insertion_bounce_min_delta: `{cfg.get('insertion_bounce_min_delta')}`",
        f"- insertion_retry_min_delta: `{cfg.get('insertion_retry_min_delta')}`",
        "",
        "## Evidence Boundary",
        "",
        "- Board force/marker/action curves are real evidence only after server-side rollout logs exist for both baseline and guided.",
        "- Insertion success/bounce/retry metrics are real evidence only after metadata is complete for every trial.",
        "- Real comparison ready requires enough paired trials and at least one task metric improving in the expected direction.",
        "- Synthetic smoke logs can test evaluator wiring, but they never make `real_comparison_ready` true.",
        "- Offline scorer metrics, Foresight gradient audits, and dry-runs remain readiness evidence, not task improvement.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board_root", default=DEFAULT_BOARD_ROOT)
    parser.add_argument("--insertion_root", default=DEFAULT_INSERTION_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="current_s12_good_margin_tac_quality")
    parser.add_argument("--skip_board", action="store_true")
    parser.add_argument("--skip_insertion", action="store_true")
    parser.add_argument("--board_expected_baseline_arm", default=DEFAULT_BOARD_BASELINE_ARM)
    parser.add_argument("--board_expected_guided_arm", default=DEFAULT_BOARD_GUIDED_ARM)
    parser.add_argument("--insertion_expected_baseline_arm", default=DEFAULT_INSERTION_BASELINE_ARM)
    parser.add_argument("--insertion_expected_guided_arm", default=DEFAULT_INSERTION_GUIDED_ARM)
    parser.add_argument("--min_board_pairs", type=int, default=3)
    parser.add_argument("--min_insertion_pairs", type=int, default=3)
    parser.add_argument("--board_in_band_min_delta", type=float, default=0.0)
    parser.add_argument("--board_smooth_min_delta", type=float, default=0.0)
    parser.add_argument("--board_abs_error_min_delta", type=float, default=0.0)
    parser.add_argument("--insertion_success_min_delta", type=float, default=0.0)
    parser.add_argument("--insertion_bounce_min_delta", type=float, default=0.0)
    parser.add_argument("--insertion_retry_min_delta", type=float, default=0.0)
    parser.add_argument("--allow_synthetic_smoke", action="store_true",
                        help="Allow synthetic smoke logs to exercise acceptance checks. They still never count as real robot evidence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    board_result = None
    board_ok = False
    board_output = ""
    if not args.skip_board:
        board_eval_dir = out_dir / "board_eval"
        board_cmd = [
            sys.executable,
            "for_show_xiaomi/eval_board_force_rollouts.py",
            "--root",
            args.board_root,
            "--output_dir",
            str(board_eval_dir),
            "--tag",
            "board",
            "--expected_baseline_arm",
            args.board_expected_baseline_arm,
            "--expected_guided_arm",
            args.board_expected_guided_arm,
        ]
        board_ok, board_output = run_cmd(board_cmd)
        board_json = board_eval_dir / "board" / "board_force_rollout_summary.json"
        if board_ok and board_json.exists():
            board_result = load_json(board_json)

    insertion_result = None
    insertion_ok = False
    insertion_output = ""
    if not args.skip_insertion:
        insertion_eval_dir = out_dir / "insertion_eval"
        insertion_cmd = [
            sys.executable,
            "for_show_xiaomi/eval_insertion_rollouts.py",
            "--root",
            args.insertion_root,
            "--output_dir",
            str(insertion_eval_dir),
            "--tag",
            "insertion",
            "--expected_baseline_arm",
            args.insertion_expected_baseline_arm,
            "--expected_guided_arm",
            args.insertion_expected_guided_arm,
        ]
        insertion_ok, insertion_output = run_cmd(insertion_cmd)
        insertion_json = insertion_eval_dir / "insertion" / "insertion_rollout_summary.json"
        if insertion_ok and insertion_json.exists():
            insertion_result = load_json(insertion_json)

    result = {
        "board_root": args.board_root,
        "insertion_root": args.insertion_root,
        "output_dir": str(out_dir),
        "expected_arms": {
            "board_baseline": args.board_expected_baseline_arm,
            "board_guided": args.board_expected_guided_arm,
            "insertion_baseline": args.insertion_expected_baseline_arm,
            "insertion_guided": args.insertion_expected_guided_arm,
        },
        "acceptance_config": {
            "min_board_pairs": int(args.min_board_pairs),
            "min_insertion_pairs": int(args.min_insertion_pairs),
            "board_in_band_min_delta": float(args.board_in_band_min_delta),
            "board_smooth_min_delta": float(args.board_smooth_min_delta),
            "board_abs_error_min_delta": float(args.board_abs_error_min_delta),
            "insertion_success_min_delta": float(args.insertion_success_min_delta),
            "insertion_bounce_min_delta": float(args.insertion_bounce_min_delta),
            "insertion_retry_min_delta": float(args.insertion_retry_min_delta),
        },
        "allow_synthetic_smoke": bool(args.allow_synthetic_smoke),
        "board": summarize_board_with_thresholds(
            board_result,
            board_ok,
            board_output,
            min_pairs=int(args.min_board_pairs),
            in_band_min=float(args.board_in_band_min_delta),
            smooth_min=float(args.board_smooth_min_delta),
            abs_error_min=float(args.board_abs_error_min_delta),
            allow_synthetic=bool(args.allow_synthetic_smoke),
        ),
        "insertion": summarize_insertion_with_thresholds(
            insertion_result,
            insertion_ok,
            insertion_output,
            min_pairs=int(args.min_insertion_pairs),
            success_min=float(args.insertion_success_min_delta),
            bounce_min=float(args.insertion_bounce_min_delta),
            retry_min=float(args.insertion_retry_min_delta),
            allow_synthetic=bool(args.allow_synthetic_smoke),
        ),
    }
    result["real_rollout_evidence_complete"] = bool(
        result["board"].get("real_comparison_ready") and result["insertion"].get("real_comparison_ready")
    )

    json_path = out_dir / "tac_quality_real_rollout_eval.json"
    md_path = out_dir / "tac_quality_real_rollout_eval.md"
    result["summary_json"] = str(json_path)
    result["summary_md"] = str(md_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "summary_json": str(json_path),
        "summary_md": str(md_path),
        "real_rollout_evidence_complete": result["real_rollout_evidence_complete"],
        "board_ready": result["board"].get("real_comparison_ready"),
        "insertion_ready": result["insertion"].get("real_comparison_ready"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
