"""Smoke-test explicit finalize attrs flowing into generated metadata.

This synthetic smoke verifies the operator path used after real collection:

1. finalize a raw HDF5 into the scheduled recommended_path;
2. write explicit success/stopped_early attrs only because CLI args provide them;
3. build generated pairing/metadata CSVs;
4. audit that metadata is complete.

It is plumbing evidence only, not policy-quality evidence.
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

from TFAC_V5.audit_tac_quality_pairing_metadata import build_audit as build_metadata_audit  # noqa: E402
from TFAC_V5.build_tac_quality_rollout_pairing import build_pairings  # noqa: E402
from TFAC_V5.finalize_tac_quality_collected_hdf5 import build as build_finalize  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_finalize_attrs_to_pairing_smoke")
ARMS = ("baseline", "default_guided", "distilled_guided")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mkdir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_raw_hdf5(path: Path, steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("ft", data=rng.normal(size=(steps, 6)).astype("float32"))
        f.create_dataset("observations/tac/left/marker_offset", data=rng.normal(size=(steps, 9, 9, 2)).astype("float32"))
        f.create_dataset("observations/tac/right/marker_offset", data=rng.normal(size=(steps, 9, 9, 2)).astype("float32"))
        f.create_dataset("actions/joint_abs", data=rng.normal(size=(steps, 7)).astype("float32"))


def build_launch_sheet(path: Path, rollout_root: Path) -> Dict[str, Any]:
    tasks: Dict[str, Any] = {}
    for task in ("insertion", "board"):
        tasks[task] = {
            "rollout_dirs": {
                arm: str(rollout_root / task / arm)
                for arm in ARMS
            }
        }
    payload = {
        "purpose": "Synthetic launch sheet for finalize attrs metadata smoke.",
        "scientific_evidence": False,
        "tasks": tasks,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def build_schedule(path: Path, rollout_root: Path, n_pairs: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    global_step = 1
    for task in ("insertion", "board"):
        for pair_idx in range(1, n_pairs + 1):
            pair_id = f"trial_{pair_idx:03d}"
            for arm in ARMS:
                filename = f"{pair_id}__{task}__{arm}.hdf5"
                rows.append(
                    {
                        "global_step": global_step,
                        "task": task,
                        "pair_id": pair_id,
                        "arm": arm,
                        "rollout_dir": str(rollout_root / task / arm),
                        "recommended_filename": filename,
                        "recommended_path": str(rollout_root / task / arm / filename),
                    }
                )
                global_step += 1
    payload = {
        "purpose": "Synthetic schedule for finalize attrs metadata smoke.",
        "scientific_evidence": False,
        "long_schedule_rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_next_step(path: Path, row: Dict[str, Any]) -> None:
    payload = {
        "purpose": "Synthetic single next-step for finalize attrs metadata smoke.",
        "scientific_evidence": False,
        "has_next_step": True,
        "next_step_pass": True,
        "next_row": row,
        "recommended_path": row["recommended_path"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    mkdir_clean(out_root)
    raw_root = out_root / "raw"
    rollout_root = out_root / "rollouts"
    launch_sheet = out_root / "synthetic_launch_sheet.json"
    schedule_path = out_root / "synthetic_schedule.json"
    next_step_path = out_root / "next_step.json"
    pairing_out = out_root / "pairing"
    audit_out = out_root / "metadata_audit"

    build_launch_sheet(launch_sheet, rollout_root)
    schedule = build_schedule(schedule_path, rollout_root, args.n_pairs)

    finalize_reports: List[Dict[str, Any]] = []
    for idx, row in enumerate(schedule["long_schedule_rows"]):
        raw = raw_root / f"raw_{idx + 1:03d}.hdf5"
        write_raw_hdf5(raw, args.steps, args.seed + idx)
        write_next_step(next_step_path, row)
        run_args = argparse.Namespace(
            next_step=str(next_step_path),
            source=str(raw),
            source_dir=None,
            output_dir=str(out_root / "finalize_reports"),
            tag=f"row_{idx + 1:03d}",
            move=False,
            overwrite=False,
            allow_bad_schema=False,
            dry_run=False,
            min_steps=3,
            success="true",
            stopped_early="false",
        )
        finalize_reports.append(build_finalize(run_args))

    pairing_args = argparse.Namespace(
        launch_sheet=str(launch_sheet),
        schedule=str(schedule_path),
        output_dir=str(pairing_out),
        tag="synthetic",
        max_pairs=0,
    )
    pairing = build_pairings(pairing_args)
    pairing_json = pairing_out / "synthetic" / "tac_quality_rollout_pairing.json"
    pairing_json.parent.mkdir(parents=True, exist_ok=True)
    pairing_json.write_text(json.dumps(pairing, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_args = argparse.Namespace(
        launch_sheet=str(launch_sheet),
        pairing_dir=str(pairing_out / "synthetic"),
        output_dir=str(audit_out),
        min_pairs=args.n_pairs,
    )
    metadata_audit = build_metadata_audit(audit_args)
    audit_out.mkdir(parents=True, exist_ok=True)
    audit_json = audit_out / "tac_quality_pairing_metadata_audit.json"
    audit_json.write_text(json.dumps(metadata_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "all_finalize_passed": all(row.get("finalize_pass") is True for row in finalize_reports),
        "all_explicit_attrs_requested": all(
            row.get("explicit_outcome_attrs", {}).get("requested") is True for row in finalize_reports
        ),
        "pairing_overall_ready": pairing.get("overall_ready") is True,
        "metadata_audit_ready": metadata_audit.get("all_tasks_ready") is True,
        "metadata_has_no_blank_rows": all(
            task.get("counts", {}).get("blank_metadata_rows") == 0
            for task in metadata_audit.get("tasks", {}).values()
        ),
    }
    summary = {
        "purpose": "Synthetic smoke for explicit finalize attrs flowing into generated metadata.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "n_pairs": int(args.n_pairs),
        "finalized_rows": len(finalize_reports),
        "outputs": {
            "launch_sheet": str(launch_sheet),
            "schedule": str(schedule_path),
            "pairing_json": str(pairing_json),
            "metadata_audit_json": str(audit_json),
        },
        "note": "Synthetic smoke verifies metadata plumbing only; it does not validate scorer quality.",
    }
    out_json = out_root / "tac_quality_finalize_attrs_to_pairing_smoke.json"
    out_md = out_root / "tac_quality_finalize_attrs_to_pairing_smoke.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)
    print(
        json.dumps(
            {
                "overall_pass": summary["overall_pass"],
                "checks": checks,
                "json": str(out_json),
                "markdown": str(out_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return summary


def write_markdown(summary: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Finalize Attrs To Pairing Smoke",
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
    lines.extend(["", "## Outputs", ""])
    for name, output in summary["outputs"].items():
        lines.append(f"- {name}: `{output}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic")
    parser.add_argument("--n_pairs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=59)
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
