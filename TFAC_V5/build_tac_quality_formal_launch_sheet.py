"""Build a one-page launch sheet for formal TacQuality rollout collection.

The real-rollout experiment packet already contains CSV templates and gate
commands.  This script combines it with the guided-server launch packet so the
collection step has exact server commands, intended rollout directories, and
post-collection gate commands for both tasks.

It is an execution handoff only; it does not claim robot validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_formal_launch_sheet")
DEFAULT_EXPERIMENT_PACKET = Path(
    "/home/chenshuai/Project/output/real_rollout_experiment_packet/"
    "formal_paired12/real_rollout_experiment_packet.json"
)
DEFAULT_GUIDED_PACKET = Path(
    "/home/chenshuai/Project/output/tac_quality_guided_server_packet/"
    "auto_discovered/tac_quality_guided_server_packet.json"
)
DEFAULT_GATE_RUNNER = Path(
    "/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner/"
    "formal_paired12_preflight/formal_tac_quality_rollout_gate_runner.json"
)
DEFAULT_ROLLOUT_ROOT = Path("/home/chenshuai/Project/output/tac_quality_formal_rollouts")
DEFAULT_GENERATED_PAIRING_DIR = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12"
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def replace_port(cmd: str, port: int) -> str:
    parts = cmd.split()
    if "--port" in parts:
        idx = parts.index("--port")
        parts[idx + 1] = str(port)
        return " ".join(parts)
    return f"{cmd} --port {port}"


def task_dirs(root: Path, task: str) -> Dict[str, str]:
    return {
        "baseline": str(root / task / "baseline"),
        "default_guided": str(root / task / "default_guided"),
        "distilled_guided": str(root / task / "distilled_guided"),
    }


def gate_runner_command(task: str, dirs: Dict[str, str], args: argparse.Namespace) -> str:
    base = [
        "python",
        "TFAC_V5/run_formal_tac_quality_rollout_gates.py",
        "--packet",
        str(args.experiment_packet),
        "--output_dir",
        "/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner",
        "--tag",
        args.gate_tag,
        "--min_episodes",
        str(args.min_episodes),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--use_generated_pairing",
        "--generated_pairing_dir",
        str(args.generated_pairing_dir),
        f"--{task}_baseline_dir",
        dirs["baseline"],
        f"--{task}_default_guided_dir",
        dirs["default_guided"],
        f"--{task}_distilled_guided_dir",
        dirs["distilled_guided"],
    ]
    return " ".join(base)


def all_tasks_gate_runner_command(all_dirs: Dict[str, Dict[str, str]], args: argparse.Namespace) -> str:
    parts = [
        "python",
        "TFAC_V5/run_formal_tac_quality_rollout_gates.py",
        "--packet",
        str(args.experiment_packet),
        "--output_dir",
        "/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner",
        "--tag",
        args.gate_tag,
        "--min_episodes",
        str(args.min_episodes),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--use_generated_pairing",
        "--generated_pairing_dir",
        str(args.generated_pairing_dir),
    ]
    for task, dirs in all_dirs.items():
        parts.extend(
            [
                f"--{task}_baseline_dir",
                dirs["baseline"],
                f"--{task}_default_guided_dir",
                dirs["default_guided"],
                f"--{task}_distilled_guided_dir",
                dirs["distilled_guided"],
            ]
        )
    return " ".join(parts)


def build(args: argparse.Namespace) -> Dict[str, Any]:
    exp = load_json(Path(args.experiment_packet))
    guided = load_json(Path(args.guided_server_packet))
    gate_runner = load_json(Path(args.gate_runner_packet))
    rollout_root = Path(args.rollout_root)
    tasks: Dict[str, Any] = {}
    all_dirs: Dict[str, Dict[str, str]] = {}
    for idx, task in enumerate(["insertion", "board"]):
        dirs = task_dirs(rollout_root, task)
        all_dirs[task] = dirs
        ports = {
            "baseline": args.base_port + idx * 10,
            "default_guided": args.base_port + idx * 10 + 1,
            "distilled_guided": args.base_port + idx * 10 + 2,
        }
        launch = {
            "baseline": replace_port(guided["tasks"][task]["baseline_command"], ports["baseline"]),
            "default_guided": replace_port(guided["tasks"][task]["default_guided_command_template"], ports["default_guided"]),
            "distilled_guided": replace_port(guided["tasks"][task]["distilled_guided_command_template"], ports["distilled_guided"]),
        }
        tasks[task] = {
            "rollout_dirs": dirs,
            "ports": ports,
            "launch_commands": launch,
            "paired_n_pairs": exp["tasks"][task]["paired_n_pairs"],
            "pairing_template": exp["tasks"][task]["pairing_template"],
            "three_arm_pairing_template": exp["tasks"][task]["three_arm_pairing_template"],
            "metadata_template": exp["tasks"][task]["metadata_template"],
            "task_gate_runner_command": gate_runner_command(task, dirs, args),
            "collection_order": [
                "baseline",
                "default_guided",
                "distilled_guided",
            ],
        }
    result = {
        "purpose": "Formal launch sheet for collecting TacQuality baseline/default/distilled rollout HDF5 files.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "experiment_packet": str(args.experiment_packet),
        "guided_server_packet": str(args.guided_server_packet),
        "gate_runner_packet": str(args.gate_runner_packet),
        "generated_pairing_dir": str(args.generated_pairing_dir),
        "post_collection_pairing_command": (
            "python TFAC_V5/build_tac_quality_rollout_pairing.py "
            f"--tag {args.tag}"
        ),
        "guided_server_ready": bool(guided.get("guided_server_ready")),
        "launch_packet_ready": bool(guided.get("launch_packet_ready")),
        "gate_runner_preflight_ready_before_collection": bool(gate_runner.get("preflight_ready")),
        "rollout_root": str(rollout_root),
        "tasks": tasks,
        "all_tasks_gate_runner_command": all_tasks_gate_runner_command(all_dirs, args),
        "next_steps": [
            "Create the rollout directories listed in this sheet.",
            "For each task and arm, launch the corresponding server command.",
            "Collect HDF5 rollouts into the matching rollout directory.",
            "Run post_collection_pairing_command to generate concrete pairing/metadata CSVs from collected HDF5s.",
            "Review generated metadata CSVs and fill blank success/stopped_early cells if HDF5 attrs are absent.",
            "Run all_tasks_gate_runner_command without --run_gates for preflight.",
            "Run the same command with --run_gates after preflight passes.",
            "Re-run TFAC_V5/audit_tac_quality_goal_completion.py.",
        ],
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
    }
    result["launch_sheet_ready"] = bool(
        result["guided_server_ready"]
        and result["launch_packet_ready"]
        and all("serve_dp_tac_quality_guided" in row["launch_commands"]["default_guided"] for row in tasks.values())
        and all("serve_dp_tac_quality_guided" in row["launch_commands"]["distilled_guided"] for row in tasks.values())
    )
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Formal Launch Sheet",
        "",
        f"- launch_sheet_ready: `{result['launch_sheet_ready']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
            f"- rollout_root: `{result['rollout_root']}`",
            f"- generated_pairing_dir: `{result['generated_pairing_dir']}`",
            f"- git_commit: `{result['git_commit']}`",
        "",
        "## Collection Commands",
        "",
    ]
    for task, row in result["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                f"- paired_n_pairs: `{row['paired_n_pairs']}`",
                "",
                "| arm | port | rollout_dir |",
                "|---|---:|---|",
            ]
        )
        for arm in row["collection_order"]:
            lines.append(f"| {arm} | {row['ports'][arm]} | `{row['rollout_dirs'][arm]}` |")
        lines.extend(["", "Launch baseline:", "", "```bash", row["launch_commands"]["baseline"], "```", ""])
        lines.extend(["Launch default guided:", "", "```bash", row["launch_commands"]["default_guided"], "```", ""])
        lines.extend(["Launch distilled guided:", "", "```bash", row["launch_commands"]["distilled_guided"], "```", ""])
        lines.extend(
            [
                "Pairing / metadata:",
                "",
                "```text",
                f"pairing: {row['pairing_template']}",
                f"three_arm_pairing: {row['three_arm_pairing_template']}",
                f"metadata: {row['metadata_template']}",
                f"generated_pairing_dir: {result['generated_pairing_dir']}/{task}",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Gate Runner",
            "",
            "Generate concrete pairing/metadata from collected HDF5s:",
            "",
            "```bash",
            result["post_collection_pairing_command"],
            "```",
            "",
            "Preflight all tasks after collection:",
            "",
            "```bash",
            result["all_tasks_gate_runner_command"],
            "```",
            "",
            "Run gates after preflight passes:",
            "",
            "```bash",
            result["all_tasks_gate_runner_command"] + " --run_gates",
            "```",
            "",
            "## Notes",
            "",
            "This sheet is not scientific evidence. It is the collection handoff for obtaining the missing real rollout evidence.",
            "The guided server commands use final-clean-action TacQuality guidance, not reranking.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_packet", default=str(DEFAULT_EXPERIMENT_PACKET))
    parser.add_argument("--guided_server_packet", default=str(DEFAULT_GUIDED_PACKET))
    parser.add_argument("--gate_runner_packet", default=str(DEFAULT_GATE_RUNNER))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--rollout_root", default=str(DEFAULT_ROLLOUT_ROOT))
    parser.add_argument("--generated_pairing_dir", default=str(DEFAULT_GENERATED_PAIRING_DIR))
    parser.add_argument("--base_port", type=int, default=8766)
    parser.add_argument("--gate_tag", default="formal_paired12_collected")
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_formal_launch_sheet.json"
    md_path = out_dir / "tac_quality_formal_launch_sheet.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "launch_sheet_ready": result["launch_sheet_ready"],
                "scientific_evidence": result["scientific_evidence"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
