"""Build a single-row operator handoff for the current TacQuality collection.

The current collection gate says whether the next formal row is safe to collect.
This script turns that GO row into one compact packet containing the exact
preflight, launch, save path, finalize, and post-run refresh commands.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_current_collection_handoff")
DEFAULT_NEXT_STEP = Path(
    "/home/chenshuai/Project/output/tac_quality_next_collection_step/"
    "formal_paired12/tac_quality_next_collection_step.json"
)
DEFAULT_GATE = Path(
    "/home/chenshuai/Project/output/tac_quality_current_collection_gate/"
    "formal_paired12/tac_quality_current_collection_gate.json"
)
DEFAULT_OUTCOME_LABEL_CARD = Path(
    "/home/chenshuai/Project/output/tac_quality_outcome_label_card/"
    "tac_quality_outcome_label_card.json"
)
DEFAULT_SCORER_FREEZE_MANIFEST = Path(
    "/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/"
    "tac_quality_scorer_freeze_manifest.json"
)


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


def with_save_hint(command: str, path: str) -> str:
    return f"TACQUALITY_RECOMMENDED_HDF5={shlex.quote(path)} {command}"


def build(args: argparse.Namespace) -> Dict[str, Any]:
    next_step = load_json(Path(args.next_step)) or {}
    gate = load_json(Path(args.gate)) or {}
    outcome_card = load_json(Path(args.outcome_label_card)) or {}
    freeze_manifest = load_json(Path(args.scorer_freeze_manifest)) or {}
    row = next_step.get("next_row") or {}
    task = row.get("task")
    task_outcome_card = get(outcome_card, f"tasks.{task}", {}) or {}
    recommended_path = str(next_step.get("recommended_path") or row.get("recommended_path") or "")
    launch_command = str(next_step.get("launch_command") or row.get("launch_command") or "")
    launch_with_hint = str(next_step.get("launch_command_with_save_path_hint") or "")
    if not launch_with_hint and recommended_path and launch_command:
        launch_with_hint = with_save_hint(launch_command, recommended_path)
    preflight_commands = [
        "conda run -n TactileACT python TFAC_V5/build_tac_quality_next_collection_step.py --tag formal_paired12",
        "conda run -n TactileACT python TFAC_V5/run_tac_quality_next_collection_step_smoke.py --tag formal_paired12",
        "conda run -n TactileACT python TFAC_V5/build_tac_quality_current_collection_gate.py --tag formal_paired12",
    ]
    finalize_commands = [
        "python TFAC_V5/finalize_and_refresh_tac_quality_collection.py --source <collected_episode.hdf5>",
        "python TFAC_V5/finalize_and_refresh_tac_quality_collection.py --source <collected_episode.hdf5> --success <true_or_false> --stopped_early <true_or_false>",
        "python TFAC_V5/finalize_and_refresh_tac_quality_collection.py --source <collected_episode.hdf5> --success <true_or_false> --stopped_early <true_or_false> --scorer_freeze_manifest /home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/tac_quality_scorer_freeze_manifest.json",
        "python TFAC_V5/finalize_and_refresh_tac_quality_collection.py --source_dir <collection_output_dir>",
        "python TFAC_V5/finalize_tac_quality_collected_hdf5.py --source <collected_episode.hdf5>",
        "python TFAC_V5/finalize_tac_quality_collected_hdf5.py --source <collected_episode.hdf5> --success <true_or_false> --stopped_early <true_or_false>",
        "python TFAC_V5/finalize_tac_quality_collected_hdf5.py --source <collected_episode.hdf5> --success <true_or_false> --stopped_early <true_or_false> --scorer_freeze_manifest /home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/tac_quality_scorer_freeze_manifest.json",
        "python TFAC_V5/finalize_tac_quality_collected_hdf5.py --source_dir <collection_output_dir>",
    ]
    post_finalize_commands = [
        "conda run -n TactileACT python TFAC_V5/build_tac_quality_collection_progress.py --tag formal_paired12",
        "conda run -n TactileACT python TFAC_V5/build_tac_quality_next_collection_step.py --tag formal_paired12",
        "conda run -n TactileACT python TFAC_V5/run_tac_quality_next_collection_step_smoke.py --tag formal_paired12",
        "conda run -n TactileACT python TFAC_V5/build_tac_quality_current_collection_gate.py --tag formal_paired12",
    ]
    checks = {
        "gate_pass": gate.get("current_collection_gate_pass") is True,
        "operator_go": gate.get("operator_go_no_go") == "go",
        "next_step_pass": next_step.get("next_step_pass") is True,
        "same_recommended_path": recommended_path == gate.get("recommended_path"),
        "launch_command_present": "serve_dp_tac_quality_guided" in launch_command,
        "launch_has_save_hint": launch_with_hint.startswith("TACQUALITY_RECOMMENDED_HDF5="),
        "recommended_path_hdf5": recommended_path.endswith(".hdf5"),
        "recommended_path_not_exists": get(gate, "checks.recommended_path_not_exists") is True,
        "preflight_commands_present": all("TactileACT" in cmd for cmd in preflight_commands),
        "finalize_commands_present": any("finalize_and_refresh_tac_quality_collection.py" in cmd for cmd in finalize_commands)
        and any("finalize_tac_quality_collected_hdf5.py" in cmd for cmd in finalize_commands),
        "post_finalize_refresh_present": all("TactileACT" in cmd for cmd in post_finalize_commands),
        "outcome_label_card_present": bool(task_outcome_card)
        and outcome_card.get("outcome_label_card_pass") is True,
        "scorer_freeze_manifest_present": freeze_manifest.get("scorer_freeze_manifest_pass") is True,
    }
    result = {
        "purpose": "Single-row operator handoff for the current formal TacQuality collection.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "next_step": str(args.next_step),
        "gate": str(args.gate),
        "current_row": row,
        "operator_go_no_go": gate.get("operator_go_no_go"),
        "recommended_path": recommended_path,
        "recommended_filename": row.get("recommended_filename"),
        "rollout_dir": row.get("rollout_dir"),
        "preflight_commands": preflight_commands,
        "launch_command_with_save_path_hint": launch_with_hint,
        "raw_hdf5_source_placeholder": "<collected_episode.hdf5>",
        "finalize_commands": finalize_commands,
        "post_finalize_commands": post_finalize_commands,
        "outcome_label_card": {
            "artifact": str(args.outcome_label_card),
            "outcome_label_card_pass": outcome_card.get("outcome_label_card_pass"),
            "task": task,
            "success_true": task_outcome_card.get("success_true"),
            "success_false": task_outcome_card.get("success_false"),
            "stopped_early_true": task_outcome_card.get("stopped_early_true"),
            "stopped_early_false": task_outcome_card.get("stopped_early_false"),
        },
        "scorer_freeze_manifest": {
            "artifact": str(args.scorer_freeze_manifest),
            "scorer_freeze_manifest_pass": freeze_manifest.get("scorer_freeze_manifest_pass"),
            "n_arms": len(freeze_manifest.get("arms", {}) or {}),
            "n_runtime_modules": len(freeze_manifest.get("runtime_modules", {}) or {}),
        },
        "checks": checks,
        "handoff_pass": all(checks.values()),
        "guardrails": [
            "Run this handoff only when operator_go_no_go is go.",
            "Save or finalize the collected HDF5 exactly to recommended_path.",
            "If the rollout outcome is known, pass --success and --stopped_early during finalize so generated metadata is not blank.",
            "Use the outcome_label_card section below to decide success/stopped_early; do not infer from scorer outputs.",
            "Use the scorer_freeze_manifest section below to keep scorer/runtime artifacts fixed during collection.",
            "After finalizing, regenerate progress/next-step/gate before collecting the next row.",
            "This handoff is not policy-quality evidence.",
        ],
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    row = result["current_row"] or {}
    lines = [
        "# TacQuality Current Collection Handoff",
        "",
        f"- handoff_pass: `{result['handoff_pass']}`",
        f"- operator_go_no_go: `{result['operator_go_no_go']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- task: `{row.get('task')}`",
        f"- pair_id: `{row.get('pair_id')}`",
        f"- arm: `{row.get('arm')}`",
        f"- recommended_filename: `{result['recommended_filename']}`",
        f"- recommended_path: `{result['recommended_path']}`",
        f"- rollout_dir: `{result['rollout_dir']}`",
        "",
        "## Preflight",
        "",
    ]
    for command in result["preflight_commands"]:
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["## Launch", "", "```bash", result["launch_command_with_save_path_hint"], "```", ""])
    lines.extend(["## Finalize", ""])
    for command in result["finalize_commands"]:
        lines.extend(["```bash", command, "```", ""])
    label_card = result.get("outcome_label_card", {})
    lines.extend(["## Outcome Label Card", ""])
    lines.append(f"- artifact: `{label_card.get('artifact')}`")
    lines.append(f"- outcome_label_card_pass: `{label_card.get('outcome_label_card_pass')}`")
    lines.extend(["", "### success=true", ""])
    for item in label_card.get("success_true") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### success=false", ""])
    for item in label_card.get("success_false") or []:
        lines.append(f"- {item}")
    lines.extend(["", "### stopped_early=true", ""])
    for item in label_card.get("stopped_early_true") or []:
        lines.append(f"- {item}")
    freeze = result.get("scorer_freeze_manifest", {})
    lines.extend(["", "## Scorer Freeze Manifest", ""])
    lines.append(f"- artifact: `{freeze.get('artifact')}`")
    lines.append(f"- scorer_freeze_manifest_pass: `{freeze.get('scorer_freeze_manifest_pass')}`")
    lines.append(f"- n_arms: `{freeze.get('n_arms')}`")
    lines.append(f"- n_runtime_modules: `{freeze.get('n_runtime_modules')}`")
    lines.extend(["## Post-Finalize Refresh", ""])
    for command in result["post_finalize_commands"]:
        lines.extend(["```bash", command, "```", ""])
    lines.extend(["## Checks", "", "| check | pass |", "|---|---:|"])
    for name, passed in result["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--next_step", default=str(DEFAULT_NEXT_STEP))
    parser.add_argument("--gate", default=str(DEFAULT_GATE))
    parser.add_argument("--outcome_label_card", default=str(DEFAULT_OUTCOME_LABEL_CARD))
    parser.add_argument("--scorer_freeze_manifest", default=str(DEFAULT_SCORER_FREEZE_MANIFEST))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_current_collection_handoff.json"
    md_path = out_dir / "tac_quality_current_collection_handoff.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "handoff_pass": result["handoff_pass"],
                "operator_go_no_go": result["operator_go_no_go"],
                "scientific_evidence": result["scientific_evidence"],
                "recommended_path": result["recommended_path"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
