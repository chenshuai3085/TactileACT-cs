"""Run the post-collection TacQuality validation pipeline.

This is the single entry point after formal HDF5 rollouts are collected.  It
orchestrates, in order:

  1. concrete pairing/metadata generation from collected HDF5s;
  2. pairing/metadata completeness audit;
  3. formal gate runner preflight, optionally with --run_gates;
  4. real rollout source audit.

By default it never runs the final evaluators; pass --run_gates explicitly
after pairing/metadata audit and preflight are ready.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_LAUNCH_SHEET = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
    "formal_paired12/tac_quality_formal_launch_sheet.json"
)
DEFAULT_SCHEDULE = Path(
    "/home/chenshuai/Project/output/tac_quality_collection_schedule/"
    "formal_paired12/tac_quality_collection_schedule.json"
)
DEFAULT_PACKET = Path(
    "/home/chenshuai/Project/output/real_rollout_experiment_packet/"
    "formal_paired12/real_rollout_experiment_packet.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_post_collection_pipeline")
DEFAULT_PAIRING_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_pairing")
DEFAULT_PAIRING_TAG = "formal_paired12"
DEFAULT_SCHEMA_AUDIT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_hdf5_schema_audit")
DEFAULT_GATE_OUT_DIR = Path("/home/chenshuai/Project/output/formal_tac_quality_rollout_gate_runner")
DEFAULT_QUALITY_GATE_OUT_DIR = Path("/home/chenshuai/Project/output/real_rollout_quality_gate")
DEFAULT_ABLATION_GATE_OUT_DIR = Path("/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate")
DEFAULT_OPTIONAL_ACTION_AWARE_OUT_DIR = Path(
    "/home/chenshuai/Project/output/optional_action_aware_rollout_gate_runner"
)
DEFAULT_OPTIONAL_ACTION_AWARE_QUALITY_OUT_DIR = Path(
    "/home/chenshuai/Project/output/optional_action_aware_rollout_quality_gate"
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def gate_commands_require_outcome_metadata(gate_report: Dict[str, Any]) -> Dict[str, Any]:
    commands = []
    for task in ["insertion", "board"]:
        task_cmds = gate_report.get("tasks", {}).get(task, {}).get("commands", {})
        for name in ["two_arm", "three_arm"]:
            value = task_cmds.get(name)
            if value:
                commands.append(value)
    return {
        "n_commands": len(commands),
        "all_require_outcome_metadata": bool(commands and all("--require_outcome_metadata" in cmd for cmd in commands)),
        "commands": commands,
    }


def _load_gate_output_json(gate_report: Dict[str, Any], name: str) -> Dict[str, Any]:
    if "two_arm" in name:
        root = Path(gate_report.get("quality_gate_output_dir", ""))
        path = root / name.replace("_two_arm", "_baseline_vs_guided") / "real_rollout_quality_gate.json"
    else:
        root = Path(gate_report.get("ablation_gate_output_dir", ""))
        task = name.replace("_three_arm", "")
        path = root / f"{task}_baseline_vs_default_vs_distilled" / "real_rollout_scorer_ablation_gate.json"
    if not path.exists():
        return {}
    return load_json(path)


def gate_outputs_have_outcome_metadata(gate_report: Dict[str, Any]) -> Dict[str, Any]:
    rows = {}
    all_ok = True
    for name, result in gate_report.get("gate_results", {}).items():
        text = f"{result.get('stdout_tail', '')}\n{result.get('stderr_tail', '')}"
        report = _load_gate_output_json(gate_report, name)
        if "two_arm" in name:
            has_flag = report.get("decision", {}).get("require_outcome_metadata") is True
            has_ok = report.get("decision", {}).get("outcome_metadata_ok") is True
        else:
            has_flag = report.get("decision_config", {}).get("require_outcome_metadata") is True
            has_ok = report.get("outcome_metadata_coverage", {}).get("complete") is True
        has_flag = has_flag or '"require_outcome_metadata": true' in text
        has_ok = has_ok or '"outcome_metadata_ok": true' in text
        rows[name] = {
            "require_outcome_metadata_seen": has_flag,
            "outcome_metadata_ok_seen": has_ok,
        }
        all_ok = all_ok and has_flag and has_ok
    return {
        "all_gate_outputs_require_and_pass_outcome_metadata": bool(rows and all_ok),
        "gate_outputs": rows,
    }


def rollout_dirs_from_launch(launch: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for task in ["insertion", "board"]:
        dirs = launch["tasks"][task]["rollout_dirs"]
        args.extend(
            [
                f"--{task}_baseline_dir",
                dirs["baseline"],
                f"--{task}_default_guided_dir",
                dirs["default_guided"],
                f"--{task}_distilled_guided_dir",
                dirs["distilled_guided"],
            ]
        )
    return args


def action_aware_dirs_from_launch(launch: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for task in ["insertion", "board"]:
        dirs = launch["tasks"][task]["rollout_dirs"]
        task_args = [
            f"--{task}_baseline_dir",
            dirs["baseline"],
        ]
        if "action_aware_guided" in dirs:
            task_args.extend(
                [
                    f"--{task}_action_aware_guided_dir",
                    dirs["action_aware_guided"],
                ]
            )
        args.extend(task_args)
    return args


def pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    launch = load_json(Path(args.launch_sheet))
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pairing_dir = Path(args.pairing_output_dir) / args.pairing_tag

    pairing_cmd = [
        sys.executable,
        "TFAC_V5/build_tac_quality_rollout_pairing.py",
        "--launch_sheet",
        str(args.launch_sheet),
        "--schedule",
        str(args.schedule),
        "--output_dir",
        str(args.pairing_output_dir),
        "--tag",
        args.pairing_tag,
    ]
    pairing_result = run_cmd(pairing_cmd)
    pairing_json = pairing_dir / "tac_quality_rollout_pairing.json"
    pairing_report = load_json(pairing_json) if pairing_json.exists() else {}

    metadata_cmd = [
        sys.executable,
        "TFAC_V5/audit_tac_quality_pairing_metadata.py",
        "--launch_sheet",
        str(args.launch_sheet),
        "--pairing_dir",
        str(pairing_dir),
        "--output_dir",
        str(args.metadata_audit_output_dir),
        "--min_pairs",
        str(args.min_episodes),
    ]
    metadata_result = run_cmd(metadata_cmd)
    metadata_json = Path(args.metadata_audit_output_dir) / "tac_quality_pairing_metadata_audit.json"
    metadata_report = load_json(metadata_json) if metadata_json.exists() else {}

    schema_cmd = [
        sys.executable,
        "TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py",
        "--launch_sheet",
        str(args.launch_sheet),
        "--output_dir",
        str(args.schema_audit_output_dir),
        "--min_episodes",
        str(args.min_episodes),
        "--min_steps",
        str(args.min_steps),
    ]
    schema_result = run_cmd(schema_cmd)
    schema_json = Path(args.schema_audit_output_dir) / "tac_quality_rollout_hdf5_schema_audit.json"
    schema_report = load_json(schema_json) if schema_json.exists() else {}

    gate_cmd = [
        sys.executable,
        "TFAC_V5/run_formal_tac_quality_rollout_gates.py",
        "--packet",
        str(args.packet),
        "--output_dir",
        str(args.gate_output_dir),
        "--tag",
        args.gate_tag,
        "--min_episodes",
        str(args.min_episodes),
        "--min_steps",
        str(args.min_steps),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--use_generated_pairing",
        "--generated_pairing_dir",
        str(pairing_dir),
        "--quality_gate_output_dir",
        str(args.quality_gate_output_dir),
        "--ablation_gate_output_dir",
        str(args.ablation_gate_output_dir),
    ]
    gate_cmd.extend(rollout_dirs_from_launch(launch))
    if args.run_gates:
        gate_cmd.append("--run_gates")
    gate_result = run_cmd(gate_cmd)
    gate_json = Path(args.gate_output_dir) / args.gate_tag / "formal_tac_quality_rollout_gate_runner.json"
    gate_report = load_json(gate_json) if gate_json.exists() else {}

    action_aware_cmd = [
        sys.executable,
        "TFAC_V5/run_optional_action_aware_rollout_gate.py",
        "--packet",
        str(args.packet),
        "--output_dir",
        str(args.optional_action_aware_output_dir),
        "--tag",
        args.optional_action_aware_tag,
        "--rollout_root",
        str(launch.get("rollout_root", "/home/chenshuai/Project/output/tac_quality_formal_rollouts")),
        "--action_aware_pairing_dir",
        str(pairing_dir),
        "--quality_gate_output_dir",
        str(args.optional_action_aware_quality_gate_output_dir),
        "--min_episodes",
        str(args.min_episodes),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
    ]
    action_aware_cmd.extend(action_aware_dirs_from_launch(launch))
    if args.run_optional_action_aware_gate:
        action_aware_cmd.append("--run_gates")
    action_aware_result = run_cmd(action_aware_cmd)
    action_aware_json = (
        Path(args.optional_action_aware_output_dir)
        / args.optional_action_aware_tag
        / "optional_action_aware_rollout_gate_runner.json"
    )
    action_aware_report = load_json(action_aware_json) if action_aware_json.exists() else {}

    source_cmd = [
        sys.executable,
        "TFAC_V5/audit_tac_quality_real_rollout_sources.py",
        "--output_dir",
        str(args.source_audit_output_dir),
    ]
    source_result = run_cmd(source_cmd)
    source_json = Path(args.source_audit_output_dir) / "tac_quality_real_rollout_source_audit.json"
    source_report = load_json(source_json) if source_json.exists() else {}

    metadata_ready = metadata_report.get("all_tasks_ready") is True
    schema_ready = schema_report.get("all_tasks_ready") is True
    preflight_ready = gate_report.get("preflight_ready") is True
    gates_passed = gate_report.get("all_requested_gates_passed") is True
    strict_outcome_commands = gate_commands_require_outcome_metadata(gate_report)
    strict_outcome_outputs = gate_outputs_have_outcome_metadata(gate_report)
    action_aware_gate_passed = action_aware_report.get("all_requested_gates_passed") is True
    can_run_gates = bool(
        pairing_report.get("overall_ready") is True
        and metadata_ready
        and schema_ready
        and preflight_ready
        and strict_outcome_commands["all_require_outcome_metadata"]
    )
    pipeline_pass = bool(
        pairing_result["passed"]
        and metadata_result["passed"]
        and schema_result["passed"]
        and gate_result["passed"]
        and action_aware_result["passed"]
        and source_result["passed"]
        and (not args.require_ready or can_run_gates)
        and (
            not args.run_gates
            or (
                gates_passed
                and strict_outcome_outputs["all_gate_outputs_require_and_pass_outcome_metadata"]
            )
        )
        and (not args.run_optional_action_aware_gate or action_aware_gate_passed)
    )
    result = {
        "purpose": "Post-collection orchestration for formal TacQuality real-rollout validation.",
        "scientific_evidence": bool(args.run_gates and gates_passed and source_report.get("all_four_real_evidence_present") is True),
        "git_commit": git_commit(),
        "run_gates_requested": bool(args.run_gates),
        "run_optional_action_aware_gate_requested": bool(args.run_optional_action_aware_gate),
        "require_ready": bool(args.require_ready),
        "pipeline_pass": pipeline_pass,
        "can_run_gates": can_run_gates,
        "pairing": {
            "command": pairing_result,
            "json": str(pairing_json),
            "overall_ready": pairing_report.get("overall_ready"),
        },
        "metadata_audit": {
            "command": metadata_result,
            "json": str(metadata_json),
            "all_tasks_ready": metadata_report.get("all_tasks_ready"),
            "tasks": metadata_report.get("tasks"),
        },
        "hdf5_schema_audit": {
            "command": schema_result,
            "json": str(schema_json),
            "all_tasks_ready": schema_report.get("all_tasks_ready"),
            "tasks": schema_report.get("tasks"),
        },
        "gate_runner": {
            "command": gate_result,
            "json": str(gate_json),
            "preflight_ready": gate_report.get("preflight_ready"),
            "use_generated_pairing": gate_report.get("use_generated_pairing"),
            "run_gates_requested": gate_report.get("run_gates_requested"),
            "all_requested_gates_passed": gate_report.get("all_requested_gates_passed"),
            "strict_outcome_metadata_commands": strict_outcome_commands,
            "strict_outcome_metadata_outputs": strict_outcome_outputs,
        },
        "optional_action_aware": {
            "command": action_aware_result,
            "json": str(action_aware_json),
            "preflight_ready": action_aware_report.get("preflight_ready"),
            "run_gates_requested": action_aware_report.get("run_gates_requested"),
            "all_requested_gates_passed": action_aware_report.get("all_requested_gates_passed"),
            "scientific_evidence": action_aware_report.get("scientific_evidence"),
            "formal_gate_dependency": action_aware_report.get("formal_gate_dependency"),
            "tasks": action_aware_report.get("tasks"),
        },
        "source_audit": {
            "command": source_result,
            "json": str(source_json),
            "all_four_real_evidence_present": source_report.get("all_four_real_evidence_present"),
            "n_real_evidence": source_report.get("n_real_evidence"),
            "n_blockers": source_report.get("n_blockers"),
            "synthetic_guardrail_pass": source_report.get("synthetic_guardrail_pass"),
        },
        "next_required_step": (
            "Collect missing HDF5 rollouts or fill metadata blanks, then rerun this pipeline."
            if not can_run_gates
            else (
                "Rerun with --run_gates to execute formal evaluators."
                if not args.run_gates
                else (
                    "Optionally rerun with --run_optional_action_aware_gate to evaluate ActionAware."
                    if not args.run_optional_action_aware_gate
                    else "Run goal audit and review real rollout source audit."
                )
            )
        ),
    }
    json_path = out_dir / "tac_quality_post_collection_pipeline.json"
    md_path = out_dir / "tac_quality_post_collection_pipeline.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "pipeline_pass": result["pipeline_pass"],
                "can_run_gates": result["can_run_gates"],
                "run_gates_requested": result["run_gates_requested"],
                "pairing_ready": result["pairing"]["overall_ready"],
                "metadata_ready": result["metadata_audit"]["all_tasks_ready"],
                "schema_ready": result["hdf5_schema_audit"]["all_tasks_ready"],
                "preflight_ready": result["gate_runner"]["preflight_ready"],
                "action_aware_preflight_ready": result["optional_action_aware"]["preflight_ready"],
                "action_aware_run_gates_requested": result["optional_action_aware"]["run_gates_requested"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Post-Collection Pipeline",
        "",
        f"- pipeline_pass: `{result['pipeline_pass']}`",
        f"- can_run_gates: `{result['can_run_gates']}`",
        f"- run_gates_requested: `{result['run_gates_requested']}`",
        f"- run_optional_action_aware_gate_requested: `{result['run_optional_action_aware_gate_requested']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- pairing_ready: `{result['pairing']['overall_ready']}`",
        f"- metadata_ready: `{result['metadata_audit']['all_tasks_ready']}`",
        f"- schema_ready: `{result['hdf5_schema_audit']['all_tasks_ready']}`",
        f"- preflight_ready: `{result['gate_runner']['preflight_ready']}`",
        f"- optional_action_aware_preflight_ready: `{result['optional_action_aware']['preflight_ready']}`",
        f"- optional_action_aware_gate_passed: `{result['optional_action_aware']['all_requested_gates_passed']}`",
        f"- source_real_evidence: `{result['source_audit']['n_real_evidence']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--pairing_output_dir", default=str(DEFAULT_PAIRING_OUTPUT_DIR))
    parser.add_argument("--pairing_tag", default=DEFAULT_PAIRING_TAG)
    parser.add_argument("--schema_audit_output_dir", default=str(DEFAULT_SCHEMA_AUDIT_OUT_DIR))
    parser.add_argument("--metadata_audit_output_dir", default="/home/chenshuai/Project/output/tac_quality_pairing_metadata_audit")
    parser.add_argument("--gate_output_dir", default=str(DEFAULT_GATE_OUT_DIR))
    parser.add_argument("--gate_tag", default="formal_paired12_preflight")
    parser.add_argument("--quality_gate_output_dir", default=str(DEFAULT_QUALITY_GATE_OUT_DIR))
    parser.add_argument("--ablation_gate_output_dir", default=str(DEFAULT_ABLATION_GATE_OUT_DIR))
    parser.add_argument("--optional_action_aware_output_dir", default=str(DEFAULT_OPTIONAL_ACTION_AWARE_OUT_DIR))
    parser.add_argument("--optional_action_aware_tag", default="formal_paired12_preflight")
    parser.add_argument(
        "--optional_action_aware_quality_gate_output_dir",
        default=str(DEFAULT_OPTIONAL_ACTION_AWARE_QUALITY_OUT_DIR),
    )
    parser.add_argument("--source_audit_output_dir", default="/home/chenshuai/Project/output/tac_quality_real_rollout_source_audit")
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--require_ready", action="store_true")
    parser.add_argument("--run_gates", action="store_true")
    parser.add_argument("--run_optional_action_aware_gate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    pipeline(parse_args())
