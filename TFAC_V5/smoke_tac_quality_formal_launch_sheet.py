"""Smoke-test every server command in the formal TacQuality launch sheet.

The launch sheet is what will be used during real rollout collection.  This
smoke parses each listed server command, converts it into a dry-run invocation,
and executes it.  It catches command drift such as missing checkpoints, wrong
arm names, unsupported DP variants, or forgotten --disable_guidance on
baselines before robot collection starts.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_LAUNCH_SHEET = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
    "formal_paired12/tac_quality_formal_launch_sheet.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_formal_launch_sheet_smoke")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def set_arg(parts: List[str], name: str, value: str) -> List[str]:
    parts = list(parts)
    if name in parts:
        idx = parts.index(name)
        parts[idx + 1] = value
    else:
        parts.extend([name, value])
    return parts


def add_flag(parts: List[str], name: str) -> List[str]:
    parts = list(parts)
    if name not in parts:
        parts.append(name)
    return parts


def dry_run_command(command: str, smoke_output: Path, gpu: int) -> List[str]:
    parts = shlex.split(command)
    if parts[:3] != ["python", "-m", "for_show_xiaomi.serve_dp_tac_quality_guided"]:
        raise ValueError(f"Unexpected launch command prefix: {parts[:3]}")
    parts[0] = sys.executable
    parts = set_arg(parts, "--gpu", str(gpu))
    parts = set_arg(parts, "--smoke_output", str(smoke_output))
    parts = add_flag(parts, "--dry_run_guidance_smoke")
    return parts


def run_command(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], text=True, capture_output=True)
    return {
        "command": " ".join(shlex.quote(x) for x in cmd),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "passed_process": proc.returncode == 0,
    }


def evaluate_smoke_output(path: Path, task: str, arm: str) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "passed_output": False}
    data = load_json(path)
    report = data.get("report", {})
    if arm == "baseline":
        passed = bool(
            data.get("dry_run_guidance_smoke_pass") is True
            and data.get("guidance_disabled") is True
            and report.get("guidance_disabled") is True
            and report.get("reranking") is False
            and report.get("every_step_ddpm_guidance") is False
        )
    elif arm == "action_aware_guided":
        passed = bool(
            data.get("dry_run_guidance_smoke_pass") is True
            and data.get("guidance_disabled") is False
            and report.get("finite_grad_rate", 0.0) >= 0.999
            and report.get("positive_grad_rate", 0.0) >= 0.999
            and report.get("max_delta_within_trust_region") is True
            and report.get("called_from_inference_mode") is True
            and report.get("returned_requires_grad") is False
            and report.get("integration_contract", {}).get("reranking") is False
            and report.get("integration_contract", {}).get("every_step_ddpm_guidance") is False
            and report.get("integration_contract", {}).get("line_search_required") is True
            and report.get("adapter_policy") == "final_clean_action_line_search_accept_only_refinement"
        )
    else:
        passed = bool(
            data.get("dry_run_guidance_smoke_pass") is True
            and data.get("guidance_disabled") is False
            and report.get("finite_grad_rate", 0.0) >= 0.999
            and report.get("positive_grad_rate", 0.0) >= 0.999
            and report.get("called_from_inference_mode") is True
            and report.get("returned_requires_grad") is False
            and report.get("integration_contract", {}).get("reranking") is False
            and report.get("integration_contract", {}).get("every_step_ddpm_guidance") is False
        )
    return {
        "path": str(path),
        "exists": True,
        "passed_output": passed,
        "dry_run_guidance_smoke_pass": data.get("dry_run_guidance_smoke_pass"),
        "task": data.get("task", task),
        "arm": data.get("arm", arm),
        "guidance_disabled": data.get("guidance_disabled"),
        "improved_rate": report.get("improved_rate"),
        "accept_rate": report.get("accept_rate"),
        "finite_grad_rate": report.get("finite_grad_rate"),
        "positive_grad_rate": report.get("positive_grad_rate"),
    }


def metric_at_least(value: Any, threshold: float) -> bool:
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def build(args: argparse.Namespace) -> Dict[str, Any]:
    launch = load_json(Path(args.launch_sheet))
    out_dir = Path(args.output_dir) / args.tag
    smoke_dir = out_dir / "per_command"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for task, task_info in launch["tasks"].items():
        for arm, command in task_info["launch_commands"].items():
            smoke_output = smoke_dir / f"{task}_{arm}_command_smoke.json"
            cmd = dry_run_command(command, smoke_output, args.gpu)
            run = run_command(cmd)
            output = evaluate_smoke_output(smoke_output, task, arm)
            rows.append(
                {
                    "task": task,
                    "arm": arm,
                    "source_launch_command": command,
                    "dry_run_command": run["command"],
                    "process": run,
                    "output": output,
                    "passes_launch_command_smoke": bool(run["passed_process"] and output["passed_output"]),
                }
            )
    result = {
        "purpose": "Dry-run every server command listed in the formal TacQuality launch sheet.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "launch_sheet": str(args.launch_sheet),
        "tag": args.tag,
        "n_commands": len(rows),
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "commands": rows,
        "checks": {
            "n_commands_expected": 8,
            "all_commands_present": len(rows) == 8,
            "formal_six_commands_present": sum(
                1
                for row in rows
                if row["arm"] in {"baseline", "default_guided", "distilled_guided"}
            )
            == 6,
            "optional_action_aware_commands_present": sum(
                1 for row in rows if row["arm"] == "action_aware_guided"
            )
            == 2,
            "all_commands_pass_process": all(row["process"]["passed_process"] for row in rows),
            "all_commands_pass_output_contract": all(row["output"]["passed_output"] for row in rows),
            "baseline_commands_disable_guidance": all(
                row["output"].get("guidance_disabled") is True
                for row in rows
                if row["arm"] == "baseline"
            ),
            "guided_commands_have_gradients": all(
                (
                    metric_at_least(row["output"].get("finite_grad_rate"), 0.999)
                    and metric_at_least(row["output"].get("positive_grad_rate"), 0.999)
                )
                for row in rows
                if row["arm"] != "baseline"
            ),
            "optional_action_aware_line_search_contract": all(
                row["output"].get("passed_output") is True
                for row in rows
                if row["arm"] == "action_aware_guided"
            ),
        },
    }
    result["overall_pass"] = all(result["checks"].values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Launch Sheet Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- n_commands: `{result['n_commands']}`",
        f"- git_commit: `{result['git_commit']}`",
        "",
        "| task | arm | pass | guidance_disabled | finite_grad | positive_grad | improved_rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["commands"]:
        out = row["output"]
        lines.append(
            "| {task} | {arm} | {passed} | {disabled} | {finite} | {positive} | {improved} |".format(
                task=row["task"],
                arm=row["arm"],
                passed=row["passes_launch_command_smoke"],
                disabled=out.get("guidance_disabled"),
                finite=out.get("finite_grad_rate"),
                positive=out.get("positive_grad_rate"),
                improved=out.get("improved_rate"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This checks launch command integrity only. It does not replace real rollout gates.",
            "Baseline commands must disable guidance; guided commands must expose a valid gradient path.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--gpu", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_formal_launch_sheet_smoke.json"
    md_path = out_dir / "tac_quality_formal_launch_sheet_smoke.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "n_commands": result["n_commands"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
