#!/usr/bin/env python3
"""Synthetic smoke test for the unified TacQuality real-rollout gate.

This script creates tiny fake baseline/guided rollout logs for board and
insertion, then runs ``eval_tac_quality_real_rollouts.py`` against them.  It is
only a pipeline check: the generated files are not robot evidence and must not
be used as performance results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("/home/chenshuai/Project/output/tac_quality_real_rollout_gate_synthetic_smoke")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    preferred = [
        "step",
        "t",
        "ft_fz",
        "ft_f_mag",
        "left_fz",
        "left_f_mag",
        "left_marker_mag_mean",
        "left_marker_mag_max",
        "action_0",
        "action_1",
    ]
    fieldnames = preferred + [key for key in keys if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    trial_dir: Path,
    *,
    task: str,
    group: str,
    arm: str,
    pair_id: str,
    success: bool | None = None,
    stopped_early: bool | None = None,
    bounce_count: int | None = None,
    retry_count: int | None = None,
) -> None:
    meta: dict[str, Any] = {
        "synthetic_smoke": True,
        "log_side": "synthetic",
        "task": task,
        "group": group,
        "pair_id": pair_id,
        "steps": 64,
        "stop_reason": "synthetic_complete",
        "server_metadata": {
            "protocol": "synthetic_tac_quality_gate_smoke",
            "task": task,
            "arm": arm,
            "guidance": "baseline_no_tac_quality_guidance" if group == "baseline" else "synthetic_guided",
            "synthetic_smoke": True,
        },
        "evidence_boundary": "Synthetic smoke only; not real robot evidence.",
    }
    if success is not None:
        meta["success"] = bool(success)
        meta["task_success"] = bool(success)
    if stopped_early is not None:
        meta["stopped_early"] = bool(stopped_early)
    if bounce_count is not None:
        meta["bounce_count"] = int(bounce_count)
    if retry_count is not None:
        meta["retry_count"] = int(retry_count)
    (trial_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def force_rows(*, force_level: float, force_noise: float, marker_level: float, action_scale: float, n: int = 64) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for step in range(n):
        phase = 2.0 * math.pi * step / max(1, n - 1)
        force = force_level + force_noise * math.sin(phase)
        action0 = action_scale * math.sin(phase)
        action1 = action_scale * math.cos(phase)
        rows.append(
            {
                "step": float(step),
                "t": 0.05 * step,
                "ft_fz": force,
                "ft_f_mag": abs(force),
                "left_fz": 0.9 * force,
                "left_f_mag": 0.9 * abs(force),
                "left_marker_mag_mean": marker_level + 0.05 * math.sin(phase),
                "left_marker_mag_max": marker_level + 0.10 * math.sin(phase),
                "left_marker_contact_area": 0.85,
                "action_0": action0,
                "action_1": action1,
                "guidance_score_delta_mean": 0.0,
                "guidance_accept_rate": 1.0,
            }
        )
    return rows


def make_board_logs(root: Path, n_pairs: int) -> None:
    # Calibration defaults in eval_board_force_rollouts.py center near 8.48 N.
    # Guided traces are intentionally closer and smoother so the gate can pass.
    for idx in range(n_pairs):
        pair_id = f"board_{idx:03d}"
        base_dir = root / "baseline" / f"synthetic_pair_{idx:03d}"
        guided_dir = root / "guided" / f"synthetic_pair_{idx:03d}"
        write_csv(
            base_dir / "force_trace.csv",
            force_rows(force_level=13.0 + 0.2 * idx, force_noise=1.6, marker_level=3.0, action_scale=0.04),
        )
        write_metadata(base_dir, task="board", group="baseline", arm="baseline", pair_id=pair_id)
        write_csv(
            guided_dir / "force_trace.csv",
            force_rows(force_level=8.5 + 0.1 * idx, force_noise=0.35, marker_level=3.0, action_scale=0.03),
        )
        write_metadata(guided_dir, task="board", group="guided", arm="marker_joint_s12_guided", pair_id=pair_id)


def make_insertion_logs(root: Path, n_pairs: int) -> None:
    for idx in range(n_pairs):
        pair_id = f"insertion_{idx:03d}"
        base_dir = root / "baseline" / f"synthetic_pair_{idx:03d}"
        guided_dir = root / "guided" / f"synthetic_pair_{idx:03d}"
        write_csv(
            base_dir / "force_trace.csv",
            force_rows(force_level=11.0 + idx, force_noise=1.0, marker_level=2.5, action_scale=0.05),
        )
        write_metadata(
            base_dir,
            task="insertion",
            group="baseline",
            arm="baseline",
            pair_id=pair_id,
            success=False,
            stopped_early=True,
            bounce_count=1,
            retry_count=1,
        )
        write_csv(
            guided_dir / "force_trace.csv",
            force_rows(force_level=8.0 + 0.2 * idx, force_noise=0.4, marker_level=2.2, action_scale=0.04),
        )
        write_metadata(
            guided_dir,
            task="insertion",
            group="guided",
            arm="good_margin_guided",
            pair_id=pair_id,
            success=True,
            stopped_early=False,
            bounce_count=0,
            retry_count=0,
        )


def run_gate(output_root: Path, n_pairs: int) -> dict[str, Any]:
    board_root = output_root / "synthetic_board_rollouts"
    insertion_root = output_root / "synthetic_insertion_rollouts"
    make_board_logs(board_root, n_pairs)
    make_insertion_logs(insertion_root, n_pairs)

    eval_output = output_root / "gate_eval"
    cmd = [
        sys.executable,
        "for_show_xiaomi/eval_tac_quality_real_rollouts.py",
        "--board_root",
        str(board_root),
        "--insertion_root",
        str(insertion_root),
        "--output_dir",
        str(eval_output),
        "--tag",
        "synthetic_gate_smoke",
        "--board_expected_baseline_arm",
        "baseline",
        "--board_expected_guided_arm",
        "marker_joint_s12_guided",
        "--insertion_expected_baseline_arm",
        "baseline",
        "--insertion_expected_guided_arm",
        "good_margin_guided",
        "--min_board_pairs",
        str(n_pairs),
        "--min_insertion_pairs",
        str(n_pairs),
        "--allow_synthetic_smoke",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    summary_json = eval_output / "synthetic_gate_smoke" / "tac_quality_real_rollout_eval.json"
    result = {
        "synthetic_smoke": True,
        "not_real_robot_evidence": True,
        "n_pairs": int(n_pairs),
        "board_root": str(board_root),
        "insertion_root": str(insertion_root),
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "summary_json": str(summary_json),
        "summary_md": str(summary_json.with_suffix(".md")),
        "pass": False,
    }
    if summary_json.exists():
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        result["gate_summary"] = summary
        result["pass"] = bool(
            proc.returncode == 0
            and summary.get("real_rollout_evidence_complete") is False
            and summary.get("board", {}).get("acceptance", {}).get("pass") is True
            and summary.get("insertion", {}).get("acceptance", {}).get("pass") is True
            and summary.get("board", {}).get("real_comparison_ready") is False
            and summary.get("insertion", {}).get("real_comparison_ready") is False
        )
    return result


def write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result.get("gate_summary", {})
    board = summary.get("board", {}) if isinstance(summary, dict) else {}
    insertion = summary.get("insertion", {}) if isinstance(summary, dict) else {}
    lines = [
        "# TacQuality Real-Rollout Gate Synthetic Smoke",
        "",
        "This is a pipeline smoke test using generated fake traces.",
        "It is not real robot evidence and must not be used as a performance claim.",
        "",
        f"- pass: `{result.get('pass')}`",
        f"- n_pairs: `{result.get('n_pairs')}`",
        f"- summary_json: `{result.get('summary_json')}`",
        f"- summary_md: `{result.get('summary_md')}`",
        f"- real_rollout_evidence_complete: `{summary.get('real_rollout_evidence_complete')}`",
        f"- board_real_comparison_ready: `{board.get('real_comparison_ready')}`",
        f"- insertion_real_comparison_ready: `{insertion.get('real_comparison_ready')}`",
        f"- board_acceptance: `{(board.get('acceptance') or {}).get('pass')}`",
        f"- insertion_acceptance: `{(insertion.get('acceptance') or {}).get('pass')}`",
        "",
        "## Boundary",
        "",
        "- Generated traces live under a `synthetic_*` output root.",
        "- Use this only to check evaluator wiring, pairing, expected-arm checks, and acceptance logic.",
        "- Real claims still require server-side logs from robot rollouts.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n_pairs", type=int, default=3)
    parser.add_argument("--clean", action="store_true", help="Remove output_root before writing synthetic logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser()
    if args.clean and output_root.exists():
        import shutil

        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    result = run_gate(output_root, int(args.n_pairs))
    result_path = output_root / "synthetic_real_rollout_gate_smoke.json"
    md_path = output_root / "synthetic_real_rollout_gate_smoke.md"
    result["result_json"] = str(result_path)
    result["result_md"] = str(md_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "pass": result["pass"],
        "result_json": str(result_path),
        "result_md": str(md_path),
        "summary_json": result["summary_json"],
    }, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
