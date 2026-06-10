"""Build explicit TacQuality outcome-label cards for real rollout collection.

The formal gates require ``success`` and ``stopped_early`` metadata.  This
artifact defines how an operator should fill those fields for socket insertion
and board wiping.  It deliberately does not infer labels from force, tactile,
scorer values, or filenames.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_outcome_label_card")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def task_card(task: str) -> Dict[str, Any]:
    if task == "insertion":
        return {
            "task": "insertion",
            "success_true": [
                "Plug/socket insertion attempt reaches the intended inserted/engaged state.",
                "No outside-wall impact or bounce event requires aborting the trial.",
                "Robot/operator does not stop early because of excessive contact, unsafe motion, or task failure.",
            ],
            "success_false": [
                "Pre-bounce risk becomes a real bounce/impact on the socket outer wall.",
                "Insertion is not completed within the rollout attempt.",
                "The trial is manually or automatically aborted because contact is unsafe or clearly wrong.",
            ],
            "stopped_early_true": [
                "Operator or safety logic stops before the planned rollout horizon.",
                "The policy is interrupted after bounce/impact, excessive force, loss of setup, or unsafe motion.",
            ],
            "stopped_early_false": [
                "The rollout reaches the normal planned termination without manual/safety interruption.",
            ],
            "quality_proxy_notes": [
                "Risk score, pre-bounce labels, and tactile/force traces can support review notes.",
                "They must not replace explicit success/stopped_early outcome entry.",
            ],
            "review_examples": [
                {
                    "case": "Smooth insertion without bounce",
                    "success": True,
                    "stopped_early": False,
                },
                {
                    "case": "Hits socket outside wall and operator stops",
                    "success": False,
                    "stopped_early": True,
                },
            ],
        }
    if task == "board":
        return {
            "task": "board",
            "success_true": [
                "The wiping stroke completes the intended contact segment on the board.",
                "Contact force stays operationally acceptable: not obviously too weak to wipe and not unsafe/excessive.",
                "Force evolution is smooth enough that the rollout would be accepted as a usable wiping action.",
            ],
            "success_false": [
                "Force is too small for meaningful wiping contact for a substantial part of the stroke.",
                "Force is too large or unsafe.",
                "Force changes are jerky/rough enough that the action is not acceptable.",
                "The rollout loses board contact, leaves the intended wiping area, or is aborted.",
            ],
            "stopped_early_true": [
                "Operator or safety logic stops before the planned rollout horizon.",
                "The trial is interrupted because force is too high, contact is lost, the arm leaves the work area, or the setup is invalid.",
            ],
            "stopped_early_false": [
                "The rollout reaches the normal planned termination without manual/safety interruption.",
            ],
            "quality_proxy_notes": [
                "Force-band and smoothness metrics are the planned quantitative proxies for board quality.",
                "They can support review notes, but formal completion still requires explicit success/stopped_early metadata.",
            ],
            "review_examples": [
                {
                    "case": "Completes stroke with moderate smooth force",
                    "success": True,
                    "stopped_early": False,
                },
                {
                    "case": "Force is too high and safety/operator stops",
                    "success": False,
                    "stopped_early": True,
                },
                {
                    "case": "Completes horizon but barely contacts board",
                    "success": False,
                    "stopped_early": False,
                },
            ],
        }
    raise ValueError(f"Unknown task: {task}")


def build(args: argparse.Namespace) -> Dict[str, Any]:
    tasks = {task: task_card(task) for task in ["insertion", "board"]}
    checks = {
        "both_tasks_present": set(tasks) == {"insertion", "board"},
        "success_true_defined": all(bool(row["success_true"]) for row in tasks.values()),
        "success_false_defined": all(bool(row["success_false"]) for row in tasks.values()),
        "stopped_early_defined": all(bool(row["stopped_early_true"]) for row in tasks.values()),
        "no_auto_inference_policy": True,
        "review_examples_present": all(bool(row["review_examples"]) for row in tasks.values()),
    }
    result = {
        "purpose": "Operator-facing outcome-label standard for TacQuality formal real rollouts.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "tasks": tasks,
        "metadata_fields": {
            "success": "bool; task-level outcome, not scorer/proxy prediction",
            "stopped_early": "bool; whether rollout was interrupted before normal planned termination",
        },
        "guardrails": [
            "Do not infer success/stopped_early from scorer outputs.",
            "Do not infer success/stopped_early from force or tactile traces without operator/reviewer judgment.",
            "Do not infer success/stopped_early from filename, arm name, or task name.",
            "When uncertain, set conservative outcome values and write review_note in metadata_review_needed.csv.",
            "Formal gates still require --require_outcome_metadata.",
        ],
        "usage": {
            "finalize_with_attrs": (
                "python TFAC_V5/finalize_and_refresh_tac_quality_collection.py "
                "--source <collected_episode.hdf5> --success <true_or_false> "
                "--stopped_early <true_or_false>"
            ),
            "manual_review_sheet": (
                "python TFAC_V5/build_tac_quality_metadata_review_sheet.py "
                "--pairing_dir /home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12"
            ),
        },
        "checks": checks,
        "outcome_label_card_pass": bool(all(checks.values())),
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Outcome Label Card",
        "",
        f"- outcome_label_card_pass: `{result['outcome_label_card_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        "",
        "## Metadata Fields",
        "",
    ]
    for key, value in result["metadata_fields"].items():
        lines.append(f"- `{key}`: {value}")
    for task, row in result["tasks"].items():
        lines.extend(["", f"## {task}", "", "### success=true", ""])
        for item in row["success_true"]:
            lines.append(f"- {item}")
        lines.extend(["", "### success=false", ""])
        for item in row["success_false"]:
            lines.append(f"- {item}")
        lines.extend(["", "### stopped_early=true", ""])
        for item in row["stopped_early_true"]:
            lines.append(f"- {item}")
        lines.extend(["", "### Review Examples", ""])
        for item in row["review_examples"]:
            lines.append(
                f"- {item['case']}: success={item['success']}, stopped_early={item['stopped_early']}"
            )
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_outcome_label_card.json"
    md_path = out_dir / "tac_quality_outcome_label_card.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "outcome_label_card_pass": result["outcome_label_card_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
