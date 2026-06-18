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


DEFAULT_BOARD_ROOT = "/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer"
DEFAULT_INSERTION_ROOT = "/home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer"
DEFAULT_OUTPUT_DIR = "/home/chenshuai/Project/output/tac_quality_real_rollout_eval"


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


def summarize_board(result: dict[str, Any] | None, ok: bool, output: str) -> dict[str, Any]:
    if not ok or result is None:
        missing = "No force_trace.csv" in output
        return {
            "evaluator_ok": False,
            "missing_force_trace": bool(missing),
            "detail": "No board force_trace.csv found yet; run baseline/guided robot tests first."
            if missing else "board evaluator failed",
            "error_output": output[-4000:],
        }
    coverage = coverage_pair(result)
    force_delta = delta(result, "quality_force_in_band_ratio")
    smooth_delta = delta(result, "quality_force_smooth_score")
    error_delta = delta(result, "quality_force_abs_error_mean", guided_minus_baseline=False)
    return {
        "evaluator_ok": True,
        **coverage,
        "summary_json": result.get("summary_json"),
        "summary_md": result.get("summary_md"),
        "overview_plot": result.get("overview_plot"),
        "group_curves_plot": result.get("group_curves_plot"),
        "quality_force_in_band_guided_minus_baseline": force_delta,
        "quality_force_smooth_guided_minus_baseline": smooth_delta,
        "quality_force_abs_error_baseline_minus_guided": error_delta,
        "real_comparison_ready": bool(coverage["has_baseline_and_guided"]),
    }


def summarize_insertion(result: dict[str, Any] | None, ok: bool, output: str) -> dict[str, Any]:
    if not ok or result is None:
        missing = "No force_trace.csv" in output
        return {
            "evaluator_ok": False,
            "missing_force_trace": bool(missing),
            "detail": "No insertion force_trace.csv found yet; run baseline/guided robot tests first."
            if missing else "insertion evaluator failed",
            "error_output": output[-4000:],
        }
    coverage = coverage_pair(result)
    meta = result.get("metadata_coverage", {})
    success_delta = delta(result, "success")
    bounce_delta = delta(result, "bounce_count", guided_minus_baseline=False)
    retry_delta = delta(result, "retry_count", guided_minus_baseline=False)
    return {
        "evaluator_ok": True,
        **coverage,
        "summary_json": str(Path(result.get("summary_csv", "")).with_name("insertion_rollout_summary.json")),
        "summary_md": str(Path(result.get("summary_csv", "")).with_name("insertion_rollout_summary.md")),
        "overview_plot": result.get("overview_plot"),
        "metadata_template_csv": result.get("metadata_template_csv"),
        "metadata_success_and_stopped_early_complete": bool(meta.get("success_and_stopped_early_complete")),
        "success_guided_minus_baseline": success_delta,
        "bounce_baseline_minus_guided": bounce_delta,
        "retry_baseline_minus_guided": retry_delta,
        "real_comparison_ready": bool(
            coverage["has_baseline_and_guided"] and meta.get("success_and_stopped_early_complete")
        ),
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
        "",
        "## Board",
        "",
        f"- evaluator_ok: `{board.get('evaluator_ok')}`",
        f"- detail: `{board.get('detail')}`",
        f"- baseline_trials: `{board.get('baseline_trials', 0)}`",
        f"- guided_trials: `{board.get('guided_trials', 0)}`",
        f"- summary_md: `{board.get('summary_md')}`",
        f"- overview_plot: `{board.get('overview_plot')}`",
        f"- force in-band guided-baseline: `{fmt(board.get('quality_force_in_band_guided_minus_baseline'))}`",
        f"- force smooth guided-baseline: `{fmt(board.get('quality_force_smooth_guided_minus_baseline'))}`",
        f"- force abs-error baseline-guided: `{fmt(board.get('quality_force_abs_error_baseline_minus_guided'))}`",
        "",
        "## Insertion",
        "",
        f"- evaluator_ok: `{insertion.get('evaluator_ok')}`",
        f"- detail: `{insertion.get('detail')}`",
        f"- baseline_trials: `{insertion.get('baseline_trials', 0)}`",
        f"- guided_trials: `{insertion.get('guided_trials', 0)}`",
        f"- metadata complete: `{insertion.get('metadata_success_and_stopped_early_complete')}`",
        f"- summary_md: `{insertion.get('summary_md')}`",
        f"- metadata_template_csv: `{insertion.get('metadata_template_csv')}`",
        f"- success guided-baseline: `{fmt(insertion.get('success_guided_minus_baseline'))}`",
        f"- bounce baseline-guided: `{fmt(insertion.get('bounce_baseline_minus_guided'))}`",
        f"- retry baseline-guided: `{fmt(insertion.get('retry_baseline_minus_guided'))}`",
        "",
        "## Evidence Boundary",
        "",
        "- Board force/marker/action curves are real evidence only after server-side rollout logs exist for both baseline and guided.",
        "- Insertion success/bounce/retry metrics are real evidence only after metadata is complete for every trial.",
        "- Offline scorer metrics, Foresight gradient audits, and dry-runs remain readiness evidence, not task improvement.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board_root", default=DEFAULT_BOARD_ROOT)
    parser.add_argument("--insertion_root", default=DEFAULT_INSERTION_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="current_tac_quality")
    parser.add_argument("--skip_board", action="store_true")
    parser.add_argument("--skip_insertion", action="store_true")
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
        ]
        insertion_ok, insertion_output = run_cmd(insertion_cmd)
        insertion_json = insertion_eval_dir / "insertion" / "insertion_rollout_summary.json"
        if insertion_ok and insertion_json.exists():
            insertion_result = load_json(insertion_json)

    result = {
        "board_root": args.board_root,
        "insertion_root": args.insertion_root,
        "output_dir": str(out_dir),
        "board": summarize_board(board_result, board_ok, board_output),
        "insertion": summarize_insertion(insertion_result, insertion_ok, insertion_output),
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
