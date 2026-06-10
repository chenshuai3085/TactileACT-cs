"""Execute and audit the current TacQuality next-step dry-run.

This is the one-command pre-robot check for the current formal schedule row:

  1. read build_tac_quality_next_collection_step.py output;
  2. execute its pre_collection_dry_run_command with this Python interpreter;
  3. audit the emitted smoke JSON with smoke_tac_quality_next_collection_step.

It does not run the robot and does not create scientific quality evidence.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from TFAC_V5.smoke_tac_quality_next_collection_step import (
    DEFAULT_NEXT_STEP,
    OUT_DIR,
    build as build_smoke_audit,
    load_json,
    write_markdown,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def command_for_current_python(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("pre_collection_dry_run_command is empty")
    if parts[0] != "python":
        raise ValueError(f"Expected command to start with python, got: {parts[0]}")
    parts[0] = sys.executable
    return parts


def run_command(cmd: list[str], cwd: Path) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return {
        "command": " ".join(shlex.quote(x) for x in cmd),
        "returncode": proc.returncode,
        "passed_process": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    next_step_path = Path(args.next_step)
    next_step = load_json(next_step_path) or {}
    command = str(next_step.get("pre_collection_dry_run_command") or "")
    cmd = command_for_current_python(command)
    run = run_command(cmd, Path(__file__).resolve().parents[1])

    audit_args = argparse.Namespace(next_step=str(next_step_path), output_dir=args.output_dir, tag=args.tag)
    audit = build_smoke_audit(audit_args)
    result = {
        "purpose": "Execute the current next-step dry-run and audit its smoke output.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "next_step": str(next_step_path),
        "next_row": next_step.get("next_row"),
        "pre_collection_dry_run_command": command,
        "pre_collection_dry_run_output": next_step.get("pre_collection_dry_run_output"),
        "process": run,
        "smoke_audit": audit,
    }
    result["overall_pass"] = bool(run["passed_process"] and audit.get("overall_pass") is True)
    return result


def write_runner_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Next-Step Dry-Run Runner",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- next_step: `{result['next_step']}`",
        f"- pre_collection_dry_run_output: `{result['pre_collection_dry_run_output']}`",
        "",
        "## Process",
        "",
        f"- returncode: `{result['process']['returncode']}`",
        f"- passed_process: `{result['process']['passed_process']}`",
        "",
        "```bash",
        result["process"]["command"],
        "```",
        "",
        "## Smoke Audit",
        "",
        f"- overall_pass: `{result['smoke_audit']['overall_pass']}`",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in result["smoke_audit"]["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Output Tail", "", "```text", result["process"]["stdout_tail"], "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next_step", default=str(DEFAULT_NEXT_STEP))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)

    audit_json = out_dir / "tac_quality_next_collection_step_smoke.json"
    audit_md = out_dir / "tac_quality_next_collection_step_smoke.md"
    audit_json.write_text(json.dumps(result["smoke_audit"], ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result["smoke_audit"], audit_md)

    json_path = out_dir / "tac_quality_next_collection_step_smoke_runner.json"
    md_path = out_dir / "tac_quality_next_collection_step_smoke_runner.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_runner_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "scientific_evidence": result["scientific_evidence"],
                "process_returncode": result["process"]["returncode"],
                "smoke_audit_pass": result["smoke_audit"]["overall_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
