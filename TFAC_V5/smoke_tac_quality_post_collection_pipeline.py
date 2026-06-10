"""Synthetic smoke for the TacQuality post-collection pipeline.

This validates the one-command post-collection entry point with synthetic
HDF5 rollouts and explicit ``--run_gates``.  All outputs are isolated under
the smoke directory so formal real-rollout evidence directories are not
polluted.

This is not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.smoke_tac_quality_generated_pairing_gate_runner import (  # noqa: E402
    build_launch_sheet,
    make_rollouts,
)
from TFAC_V5.run_formal_tac_quality_rollout_gates import DEFAULT_PACKET, load_json  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_post_collection_pipeline_smoke")
DEFAULT_LAUNCH_SHEET = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
    "formal_paired12/tac_quality_formal_launch_sheet.json"
)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mkdir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    mkdir_clean(out_root)
    rollout_root = out_root / "synthetic_rollouts"
    rollout_dirs = make_rollouts(rollout_root, args.n_pairs, args.seed)
    base_launch_sheet = load_json(Path(args.launch_sheet))
    smoke_launch_sheet = out_root / "synthetic_launch_sheet.json"
    build_launch_sheet(base_launch_sheet, rollout_dirs, smoke_launch_sheet)

    pipeline_out = out_root / "pipeline"
    pairing_out = out_root / "pairing"
    metadata_out = out_root / "metadata_audit"
    gate_out = out_root / "gate_runner"
    quality_out = out_root / "quality_gate_results"
    ablation_out = out_root / "ablation_gate_results"
    source_out = out_root / "source_audit"
    action_aware_out = out_root / "optional_action_aware_gate_runner"
    action_aware_quality_out = out_root / "optional_action_aware_quality_gate_results"

    cmd = [
        sys.executable,
        "TFAC_V5/run_tac_quality_post_collection_pipeline.py",
        "--launch_sheet",
        str(smoke_launch_sheet),
        "--packet",
        str(args.packet),
        "--output_dir",
        str(pipeline_out),
        "--tag",
        "synthetic",
        "--pairing_output_dir",
        str(pairing_out),
        "--pairing_tag",
        "synthetic",
        "--metadata_audit_output_dir",
        str(metadata_out),
        "--gate_output_dir",
        str(gate_out),
        "--gate_tag",
        "synthetic",
        "--quality_gate_output_dir",
        str(quality_out),
        "--ablation_gate_output_dir",
        str(ablation_out),
        "--optional_action_aware_output_dir",
        str(action_aware_out),
        "--optional_action_aware_tag",
        "synthetic",
        "--optional_action_aware_quality_gate_output_dir",
        str(action_aware_quality_out),
        "--source_audit_output_dir",
        str(source_out),
        "--min_episodes",
        str(args.n_pairs),
        "--bootstrap_samples",
        str(args.bootstrap_samples),
        "--require_ready",
        "--run_gates",
        "--run_optional_action_aware_gate",
    ]
    run = run_cmd(cmd)
    pipeline_json = pipeline_out / "synthetic" / "tac_quality_post_collection_pipeline.json"
    report = load_json(pipeline_json) if pipeline_json.exists() else {}

    # The source audit reads only the standard formal real-rollout output dirs,
    # so it should not count synthetic smoke outputs as real evidence.
    source = report.get("source_audit", {})
    expected = {
        "pipeline_pass": report.get("pipeline_pass") is True,
        "can_run_gates": report.get("can_run_gates") is True,
        "run_gates_requested": report.get("run_gates_requested") is True,
        "pairing_ready": report.get("pairing", {}).get("overall_ready") is True,
        "metadata_ready": report.get("metadata_audit", {}).get("all_tasks_ready") is True,
        "preflight_ready": report.get("gate_runner", {}).get("preflight_ready") is True,
        "gates_passed": report.get("gate_runner", {}).get("all_requested_gates_passed") is True,
        "optional_action_aware_preflight_ready": report.get("optional_action_aware", {}).get("preflight_ready") is True,
        "optional_action_aware_gates_passed": report.get("optional_action_aware", {}).get("all_requested_gates_passed") is True,
        "optional_action_aware_not_formal_dependency": (
            report.get("optional_action_aware", {}).get("formal_gate_dependency") is False
        ),
        "source_guardrail_keeps_formal_gap": source.get("n_real_evidence") == 0 and source.get("n_blockers") == 4,
        "not_scientific_evidence": report.get("scientific_evidence") is False,
    }
    summary = {
        "purpose": "Synthetic smoke for post-collection TacQuality pipeline with --run_gates.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "n_pairs": args.n_pairs,
        "overall_pass": bool(run["passed"] and all(expected.values())),
        "pipeline_command": run,
        "pipeline_json": str(pipeline_json),
        "checks": expected,
        "pipeline_summary": {
            "pipeline_pass": report.get("pipeline_pass"),
            "can_run_gates": report.get("can_run_gates"),
            "run_gates_requested": report.get("run_gates_requested"),
            "pairing_ready": report.get("pairing", {}).get("overall_ready"),
            "metadata_ready": report.get("metadata_audit", {}).get("all_tasks_ready"),
            "preflight_ready": report.get("gate_runner", {}).get("preflight_ready"),
            "gates_passed": report.get("gate_runner", {}).get("all_requested_gates_passed"),
            "optional_action_aware": report.get("optional_action_aware"),
            "source_audit": source,
        },
        "note": "Synthetic smoke only; formal real evidence dirs are not used for evaluator outputs.",
    }
    json_path = out_root / "tac_quality_post_collection_pipeline_smoke.json"
    md_path = out_root / "tac_quality_post_collection_pipeline_smoke.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(
        json.dumps(
            {
                "overall_pass": summary["overall_pass"],
                "pipeline_pass": summary["pipeline_summary"]["pipeline_pass"],
                "can_run_gates": summary["pipeline_summary"]["can_run_gates"],
                "gates_passed": summary["pipeline_summary"]["gates_passed"],
                "optional_action_aware_gates_passed": summary["pipeline_summary"]["optional_action_aware"].get(
                    "all_requested_gates_passed"
                )
                if isinstance(summary["pipeline_summary"]["optional_action_aware"], dict)
                else None,
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Post-Collection Pipeline Smoke",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- scientific_evidence: `{summary['scientific_evidence']}`",
        f"- n_pairs: `{summary['n_pairs']}`",
        f"- note: {summary['note']}",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--packet", default=str(DEFAULT_PACKET))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic_n10")
    parser.add_argument("--n_pairs", type=int, default=10)
    parser.add_argument("--bootstrap_samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
