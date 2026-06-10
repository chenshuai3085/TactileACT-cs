"""Smoke-check the current TacQuality next collection step.

This script audits the pre-collection dry-run JSON produced from
build_tac_quality_next_collection_step.py.  It does not run the robot and does
not count as quality evidence; it only verifies that the current scheduled arm
has already passed the local server/scorer/foresight dry-run contract.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_NEXT_STEP = Path(
    "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
    "formal_paired12/tac_quality_next_collection_step.json"
)
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_next_collection_step_smoke")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get(d: Optional[Dict[str, Any]], dotted: str, default=None):
    cur: Any = d
    if cur is None:
        return default
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def smoke_contract(next_step: Dict[str, Any], smoke: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    row = next_step.get("next_row") or {}
    report = (smoke or {}).get("report", {})
    arm = row.get("arm")
    baseline = arm == "baseline"
    checks = {
        "next_step_pass": next_step.get("next_step_pass") is True,
        "scientific_evidence_false": next_step.get("scientific_evidence") is False,
        "has_next_step": next_step.get("has_next_step") is True,
        "dry_run_required": next_step.get("pre_collection_dry_run_required") is True,
        "dry_run_command_present": "--dry_run_guidance_smoke"
        in str(next_step.get("pre_collection_dry_run_command", "")),
        "dry_run_output_declared": str(next_step.get("pre_collection_dry_run_output", "")).endswith("_smoke.json"),
        "dry_run_output_exists": smoke is not None,
        "smoke_pass": get(smoke, "dry_run_guidance_smoke_pass") is True,
        "task_matches": get(smoke, "task") == row.get("task"),
        "arm_matches": get(smoke, "arm") == arm,
        "not_reranking": get(smoke, "not_reranking") is True,
        "final_clean_action_guidance": get(smoke, "guidance_location") == "after DP clean action chunk",
        "baseline_guidance_disabled": (
            get(smoke, "guidance_disabled") is True
            and report.get("guidance_disabled") is True
            and report.get("reranking") is False
            and report.get("every_step_ddpm_guidance") is False
        )
        if baseline
        else True,
        "guided_grad_contract": (
            (report.get("finite_grad_rate", 0.0) or 0.0) >= 0.999
            and (report.get("positive_grad_rate", 0.0) or 0.0) >= 0.999
            and report.get("called_from_inference_mode") is True
            and report.get("returned_requires_grad") is False
        )
        if not baseline
        else True,
    }
    return checks


def build(args: argparse.Namespace) -> Dict[str, Any]:
    next_step_path = Path(args.next_step)
    next_step = load_json(next_step_path) or {}
    smoke_output = Path(str(next_step.get("pre_collection_dry_run_output", "")))
    smoke = load_json(smoke_output) if str(smoke_output) else None
    checks = smoke_contract(next_step, smoke)
    result = {
        "purpose": "Audit the current next-step pre-collection dry-run smoke output.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "next_step": str(next_step_path),
        "pre_collection_dry_run_output": str(smoke_output),
        "next_row": next_step.get("next_row"),
        "smoke_summary": {
            "dry_run_guidance_smoke_pass": get(smoke, "dry_run_guidance_smoke_pass"),
            "task": get(smoke, "task"),
            "arm": get(smoke, "arm"),
            "guidance_disabled": get(smoke, "guidance_disabled"),
            "device": get(smoke, "device"),
            "variant": get(smoke, "variant"),
            "report": get(smoke, "report"),
        },
        "checks": checks,
    }
    result["overall_pass"] = all(checks.values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Next Collection Step Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- next_step: `{result['next_step']}`",
        f"- pre_collection_dry_run_output: `{result['pre_collection_dry_run_output']}`",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Smoke Summary", "", "```json"])
    lines.append(json.dumps(result["smoke_summary"], ensure_ascii=False, indent=2))
    lines.extend(["```", ""])
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
    json_path = out_dir / "tac_quality_next_collection_step_smoke.json"
    md_path = out_dir / "tac_quality_next_collection_step_smoke.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "scientific_evidence": result["scientific_evidence"],
                "pre_collection_dry_run_output": result["pre_collection_dry_run_output"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
