"""Synthetic smoke for finalizing collected TacQuality HDF5 files.

The guided DP server does not write HDF5 rollouts.  A robot/client-side
collector writes a raw file first, then ``finalize_tac_quality_collected_hdf5``
copies or moves it to the current schedule row's recommended path.

This smoke validates the file-placement utility in an isolated output
directory.  It is not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_finalize_collected_hdf5_smoke")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mkdir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_hdf5(path: Path, seed: int, steps: int = 8) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    force = rng.normal(0.4, 0.03, size=(steps, 6)).astype(np.float32)
    left_marker = rng.normal(0.0, 0.01, size=(steps, 9, 9, 2)).astype(np.float32)
    right_marker = rng.normal(0.0, 0.01, size=(steps, 9, 9, 2)).astype(np.float32)
    joint = rng.normal(0.0, 0.02, size=(steps, 7)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("ft", data=force)
        f.create_dataset("observations/tac/left/marker_offset", data=left_marker)
        f.create_dataset("observations/tac/right/marker_offset", data=right_marker)
        f.create_dataset("actions/joint_abs", data=joint)
        f.attrs["success"] = 1.0
        f.attrs["stopped_early"] = 0.0


def write_next_step(path: Path, target: Path, task: str = "insertion") -> None:
    row = {
        "task": task,
        "pair_id": "trial_001",
        "arm": "baseline",
        "recommended_filename": target.name,
    }
    payload = {
        "has_next_step": True,
        "recommended_path": str(target),
        "next_row": row,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    json_payload = None
    if proc.stdout.strip():
        try:
            json_payload = json.loads(proc.stdout)
        except Exception:
            json_payload = None
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "json_stdout": json_payload,
    }


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    mkdir_clean(out_root)
    source = out_root / "raw_collection" / "episode_000.hdf5"
    newer_source = out_root / "raw_collection" / "episode_001.hdf5"
    target = out_root / "scheduled" / "trial_001__insertion__baseline.hdf5"
    next_step = out_root / "synthetic_next_step.json"
    write_hdf5(source, args.seed)
    write_hdf5(newer_source, args.seed + 1)
    write_next_step(next_step, target)

    copy_cmd = [
        sys.executable,
        "TFAC_V5/finalize_tac_quality_collected_hdf5.py",
        "--next_step",
        str(next_step),
        "--source",
        str(source),
        "--output_dir",
        str(out_root / "reports"),
        "--tag",
        "copy",
    ]
    copy_run = run_cmd(copy_cmd)
    copy_report_path = out_root / "reports" / "copy" / "tac_quality_finalize_collected_hdf5.json"
    copy_report = json.loads(copy_report_path.read_text(encoding="utf-8")) if copy_report_path.exists() else {}

    refuse_cmd = [
        sys.executable,
        "TFAC_V5/finalize_tac_quality_collected_hdf5.py",
        "--next_step",
        str(next_step),
        "--source",
        str(newer_source),
        "--output_dir",
        str(out_root / "reports"),
        "--tag",
        "refuse_existing",
    ]
    refuse_run = run_cmd(refuse_cmd)
    refuse_report_path = out_root / "reports" / "refuse_existing" / "tac_quality_finalize_collected_hdf5.json"
    refuse_report = json.loads(refuse_report_path.read_text(encoding="utf-8")) if refuse_report_path.exists() else {}

    source_dir_target = out_root / "scheduled_source_dir" / "trial_001__insertion__baseline.hdf5"
    source_dir_next_step = out_root / "synthetic_next_step_source_dir.json"
    write_next_step(source_dir_next_step, source_dir_target)
    source_dir_cmd = [
        sys.executable,
        "TFAC_V5/finalize_tac_quality_collected_hdf5.py",
        "--next_step",
        str(source_dir_next_step),
        "--source_dir",
        str(out_root / "raw_collection"),
        "--output_dir",
        str(out_root / "reports"),
        "--tag",
        "source_dir_newest",
    ]
    source_dir_run = run_cmd(source_dir_cmd)
    source_dir_report_path = out_root / "reports" / "source_dir_newest" / "tac_quality_finalize_collected_hdf5.json"
    source_dir_report = (
        json.loads(source_dir_report_path.read_text(encoding="utf-8")) if source_dir_report_path.exists() else {}
    )

    checks = {
        "copy_command_passed": copy_run["passed"],
        "copy_finalize_pass": copy_report.get("finalize_pass") is True,
        "copy_operation": copy_report.get("operation") == "copy",
        "copy_keeps_source": source.exists(),
        "copy_target_schema_ok": copy_report.get("target_audit", {}).get("schema_ok") is True,
        "refuse_command_passed": refuse_run["passed"],
        "refuse_existing_target": refuse_report.get("refusal_reason") == "target exists; pass --overwrite to replace it",
        "refuse_finalize_not_pass": refuse_report.get("finalize_pass") is False,
        "source_dir_command_passed": source_dir_run["passed"],
        "source_dir_finalize_pass": source_dir_report.get("finalize_pass") is True,
        "source_dir_picks_newest": Path(source_dir_report.get("source", "")).name == "episode_001.hdf5",
        "source_dir_target_schema_ok": source_dir_report.get("target_audit", {}).get("schema_ok") is True,
    }
    summary = {
        "purpose": "Synthetic smoke for TacQuality collected-HDF5 finalize utility.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "reports": {
            "copy": str(copy_report_path),
            "refuse_existing": str(refuse_report_path),
            "source_dir_newest": str(source_dir_report_path),
        },
        "commands": {
            "copy": copy_run,
            "refuse_existing": refuse_run,
            "source_dir_newest": source_dir_run,
        },
        "note": "Synthetic smoke only; does not use formal rollout directories or count as robot evidence.",
    }
    json_path = out_root / "tac_quality_finalize_collected_hdf5_smoke.json"
    md_path = out_root / "tac_quality_finalize_collected_hdf5_smoke.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(
        json.dumps(
            {
                "overall_pass": summary["overall_pass"],
                "scientific_evidence": summary["scientific_evidence"],
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
        "# TacQuality Finalize Collected HDF5 Smoke",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- scientific_evidence: `{summary['scientific_evidence']}`",
        f"- note: {summary['note']}",
        "",
        "## Checks",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in summary["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Reports", ""])
    for name, report in summary["reports"].items():
        lines.append(f"- {name}: `{report}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
