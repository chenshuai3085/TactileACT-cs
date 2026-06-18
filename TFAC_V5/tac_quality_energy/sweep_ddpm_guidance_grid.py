#!/usr/bin/env python3
"""Grid search DDPM-step TacQuality guidance settings.

This script wraps the board and insertion DDPM-step sweep scripts and produces
a compact cross-setting comparison table.  The result is offline sampler
evidence only: it measures whether the scorer gives stable local gradients
under recorded observations, not real robot improvement.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from TFAC_V5.tac_quality_energy.sweep_board_ddpm_step_guidance import (  # noqa: E402
    build_parser as build_board_parser,
    run_sweep as run_board_sweep,
)
from TFAC_V5.tac_quality_energy.sweep_insertion_ddpm_step_guidance import (  # noqa: E402
    build_parser as build_insertion_parser,
    run_sweep as run_insertion_sweep,
)


DEFAULT_OUTPUT_DIR = Path(
    "/home/chenshuai/Project/output/tac_quality_ddpm_step_guidance_audit/"
    "guidance_grid_20260619"
)


def parse_csv_floats(value: str) -> list[float]:
    return [float(x) for x in str(value).split(",") if x.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in str(value).split(",") if x.strip()]


def make_args(task: str, overrides: dict[str, Any]) -> argparse.Namespace:
    parser = build_board_parser() if task == "board" else build_insertion_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def metric(summary: dict[str, Any], key: str, subkey: str | None = None, default: Any = None) -> Any:
    value = summary.get(key, default)
    if subkey is None:
        return value
    if isinstance(value, dict):
        return value.get(subkey, default)
    return default


def aggregate_row(task: str, setting_id: str, result: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    summary = result["summary"]
    guidance = result["guidance"]
    return {
        "task": task,
        "setting_id": setting_id,
        "scheduler": guidance.get("scheduler"),
        "num_inference_steps": guidance.get("num_inference_steps"),
        "guidance_steps": guidance.get("guidance_steps"),
        "guidance_scale": guidance.get("guidance_scale"),
        "max_delta_norm": guidance.get("max_delta_norm"),
        "sample_clip": guidance.get("sample_clip"),
        "accept_only_improved": guidance.get("accept_only_improved"),
        "final_accept_only": guidance.get("final_accept_only"),
        "n_points": summary.get("n_points"),
        "n_rows": summary.get("n_rows"),
        "n_skipped_episodes": summary.get("n_skipped_episodes"),
        "final_improve_rate": summary.get("final_score_improve_rate"),
        "final_delta_mean": metric(summary, "final_score_delta", "mean"),
        "final_delta_std": metric(summary, "final_score_delta", "std"),
        "final_delta_min": metric(summary, "final_score_delta", "min"),
        "final_delta_max": metric(summary, "final_score_delta", "max"),
        "raw_final_delta_mean": metric(summary, "raw_final_score_delta", "mean"),
        "raw_final_delta_min": metric(summary, "raw_final_score_delta", "min"),
        "finite_grad_mean": metric(summary, "finite_grad_rate", "mean"),
        "positive_grad_mean": metric(summary, "positive_grad_rate", "mean"),
        "accept_rate_mean": metric(summary, "accept_rate", "mean"),
        "final_accept_rate_mean": metric(summary, "final_accept_rate", "mean"),
        "action_delta_norm_mean": metric(summary, "guided_action_delta_norm", "mean"),
        "action_delta_norm_max": metric(summary, "guided_action_delta_norm", "max"),
        "raw_action_delta_norm_mean": metric(summary, "raw_guided_action_delta_norm", "mean"),
        "contact_gate_mean": metric(summary, "contact_gate_value", "mean"),
        "output_dir": str(output_dir),
    }


def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    improve = float(row.get("final_improve_rate") or 0.0)
    min_delta = float(row.get("final_delta_min") or 0.0)
    mean_delta = float(row.get("final_delta_mean") or 0.0)
    action_norm = float(row.get("action_delta_norm_mean") or 0.0)
    return (-improve, -min_delta, -mean_delta, action_norm)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]], *, boundary: str) -> str:
    lines = [
        "# DDPM-Step TacQuality Guidance Parameter Grid",
        "",
        f"- evidence boundary: {boundary}",
        "- selection rule: prefer high improve rate, non-negative worst-case delta, and small action update norm.",
        "",
        "## Summary Table",
        "",
        "| task | setting | inf | guide | scale | rows | improve | mean delta | min delta | step accept | final accept | action norm mean |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {task} | `{setting_id}` | {num_inference_steps} | {guidance_steps} | "
            "{guidance_scale:.6g} | {n_rows} | "
            "{final_improve_rate:.4f} | {final_delta_mean:.6f} | "
            "{final_delta_min:.6f} | {accept_rate_mean:.4f} | "
            "{final_accept_rate_mean:.4f} | "
            "{action_delta_norm_mean:.6f} |".format(**row)
        )

    lines.extend(["", "## Recommended Settings", ""])
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        task_rows.sort(key=sort_key)
        best = task_rows[0]
        lines.extend(
            [
                f"### {task}",
                "",
                f"- recommended offline setting: `{best['setting_id']}`",
                f"- improve rate: `{best['final_improve_rate']:.4f}`",
                f"- final delta mean/min: `{best['final_delta_mean']:.6f}` / `{best['final_delta_min']:.6f}`",
                f"- accept rate mean: `{best['accept_rate_mean']:.4f}`",
                f"- final accept rate mean: `{best['final_accept_rate_mean']:.4f}`",
                f"- action delta norm mean/max: `{best['action_delta_norm_mean']:.6f}` / `{best['action_delta_norm_max']:.6f}`",
                f"- full output: `{best['output_dir']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- A setting is not a robot-performance claim unless it is later validated by paired real rollouts.",
            "- Positive offline deltas mean the current differentiable scorer can provide a local gradient in the sampler.",
            "- If a task has any negative worst-case delta, keep that setting experimental and prefer smaller or later-only guidance.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_task_grid(
    *,
    task: str,
    output_root: Path,
    num_inference_steps: list[int],
    guidance_steps: list[int],
    guidance_scales: list[float],
    max_delta_norms: list[float],
    shared_overrides: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for inf in num_inference_steps:
        for gsteps in guidance_steps:
            if gsteps > inf:
                continue
            for scale in guidance_scales:
                for max_delta in max_delta_norms:
                    setting_id = f"{task}_inf{inf}_g{gsteps}_s{scale:g}_d{max_delta:g}"
                    setting_dir = output_root / setting_id
                    overrides = copy.deepcopy(shared_overrides)
                    overrides.update(
                        {
                            "num_inference_steps": int(inf),
                            "guidance_steps": int(gsteps),
                            "guidance_scale": float(scale),
                            "max_delta_norm": float(max_delta),
                            "output_dir": str(setting_dir),
                            "disable_accept_only": bool(shared_overrides.get("disable_accept_only", False)),
                            "disable_final_accept_only": bool(shared_overrides.get("disable_final_accept_only", False)),
                        }
                    )
                    print(f"\n=== {setting_id} ===", flush=True)
                    args = make_args(task, overrides)
                    result = run_board_sweep(args) if task == "board" else run_insertion_sweep(args)
                    setting_dir.mkdir(parents=True, exist_ok=True)
                    json_name = (
                        "board_ddpm_step_guidance_sweep.json"
                        if task == "board"
                        else "insertion_ddpm_step_guidance_sweep.json"
                    )
                    json_path = setting_dir / json_name
                    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                    row = aggregate_row(task, setting_id, result, setting_dir)
                    rows.append(row)
                    results.append({"setting_id": setting_id, "result": result, "summary": row})
    return rows, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="insertion,board", help="Comma-separated tasks: insertion,board")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_inference_steps", default="4")
    parser.add_argument("--guidance_steps", default="1,2")
    parser.add_argument("--guidance_scales", default="0.00025,0.0005,0.001")
    parser.add_argument("--max_delta_norms", default="0.005,0.01,0.02")
    parser.add_argument("--max_episodes", type=int, default=4)
    parser.add_argument("--starts_per_episode", type=int, default=2)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--disable_accept_only", action="store_true")
    parser.add_argument("--disable_final_accept_only", action="store_true")
    parser.add_argument("--board_contact_min", type=float, default=1.8)
    parser.add_argument("--insertion_contact_min", type=float, default=1.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    rows: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    common = {
        "max_episodes": int(args.max_episodes),
        "starts_per_episode": int(args.starts_per_episode),
        "seeds": args.seeds,
        "gpu": int(args.gpu),
        "disable_accept_only": bool(args.disable_accept_only),
        "disable_final_accept_only": bool(args.disable_final_accept_only),
    }
    for task in tasks:
        if task not in {"board", "insertion"}:
            raise ValueError(f"Unknown task: {task}")
        overrides = dict(common)
        overrides["contact_min"] = float(args.board_contact_min if task == "board" else args.insertion_contact_min)
        task_rows, task_results = run_task_grid(
            task=task,
            output_root=output_root,
            num_inference_steps=parse_csv_ints(args.num_inference_steps),
            guidance_steps=parse_csv_ints(args.guidance_steps),
            guidance_scales=parse_csv_floats(args.guidance_scales),
            max_delta_norms=parse_csv_floats(args.max_delta_norms),
            shared_overrides=overrides,
        )
        rows.extend(task_rows)
        all_results.extend(task_results)

    rows.sort(key=lambda row: (row["task"], sort_key(row)))
    csv_path = output_root / "guidance_grid_summary.csv"
    json_path = output_root / "guidance_grid_summary.json"
    md_path = output_root / "guidance_grid_summary.md"
    boundary = (
        "Offline sampler parameter grid only. It evaluates scorer-gradient "
        "stability under recorded observations and does not prove real robot improvement."
    )
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"boundary": boundary, "rows": rows, "results": all_results}, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(rows, boundary=boundary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_root),
                "csv": str(csv_path),
                "json": str(json_path),
                "markdown": str(md_path),
                "n_settings": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
