"""Audit HDF5 schema for formal TacQuality rollout directories.

Formal quality gates depend on force, tactile marker, and action sequences.
This script checks the collected HDF5 files before pairing/gate evaluation so
missing fields are caught early.

It does not evaluate policy quality and does not claim scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.eval_real_rollout_quality_gate import discover_hdf5  # noqa: E402


DEFAULT_LAUNCH_SHEET = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
    "formal_paired12/tac_quality_formal_launch_sheet.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_hdf5_schema_audit")
TASKS = ("insertion", "board")
ARMS = ("baseline", "default_guided", "distilled_guided")
FORCE_KEYS = ("ft", "observations/tac/left/force6d", "observations/tac/right/force6d")
MARKER_KEYS = ("observations/tac/left/marker_offset", "observations/tac/right/marker_offset")
ACTION_KEYS = ("actions/joint_abs", "actions/eef_abs")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dataset_info(f: h5py.File, key: str) -> Optional[Dict[str, Any]]:
    if key not in f:
        return None
    d = f[key]
    return {
        "key": key,
        "shape": [int(x) for x in d.shape],
        "dtype": str(d.dtype),
    }


def first_existing(f: h5py.File, keys: tuple[str, ...]) -> Optional[Dict[str, Any]]:
    for key in keys:
        info = dataset_info(f, key)
        if info is not None:
            return info
    return None


def valid_len(info: Optional[Dict[str, Any]], min_steps: int) -> bool:
    if info is None:
        return False
    shape = info.get("shape") or []
    return bool(shape and shape[0] >= min_steps)


def audit_file(path: Path, min_steps: int) -> Dict[str, Any]:
    try:
        with h5py.File(path, "r") as f:
            force = first_existing(f, FORCE_KEYS)
            left_marker = dataset_info(f, "observations/tac/left/marker_offset")
            right_marker = dataset_info(f, "observations/tac/right/marker_offset")
            action = first_existing(f, ACTION_KEYS)
            success_attr = "success" in f.attrs
            stopped_attr = "stopped_early" in f.attrs
            freeze_sha_attr = "tac_quality_scorer_freeze_manifest_sha256" in f.attrs
            freeze_commit_attr = "tac_quality_scorer_freeze_git_commit" in f.attrs
    except Exception as exc:
        return {
            "path": str(path),
            "schema_ok": False,
            "open_ok": False,
            "reason": f"failed to open/read HDF5: {exc}",
        }

    missing: List[str] = []
    if not valid_len(force, min_steps):
        missing.append("force_source")
    if not valid_len(left_marker, min_steps):
        missing.append("observations/tac/left/marker_offset")
    if not valid_len(right_marker, min_steps):
        missing.append("observations/tac/right/marker_offset")
    if not valid_len(action, min_steps):
        missing.append("action_source")
    schema_ok = not missing
    return {
        "path": str(path),
        "schema_ok": bool(schema_ok),
        "open_ok": True,
        "missing_or_short_required_fields": missing,
        "force_source": force,
        "left_marker": left_marker,
        "right_marker": right_marker,
        "action_source": action,
        "attrs": {
            "success": success_attr,
            "stopped_early": stopped_attr,
            "tac_quality_scorer_freeze_manifest_sha256": freeze_sha_attr,
            "tac_quality_scorer_freeze_git_commit": freeze_commit_attr,
        },
        "missing_optional_attrs": [
            name
            for name, exists in {
                "success": success_attr,
                "stopped_early": stopped_attr,
                "tac_quality_scorer_freeze_manifest_sha256": freeze_sha_attr,
                "tac_quality_scorer_freeze_git_commit": freeze_commit_attr,
            }.items()
            if not exists
        ],
    }


def audit_arm(path: Path, min_episodes: int, min_steps: int) -> Dict[str, Any]:
    files = discover_hdf5(path) if path.exists() else []
    file_reports = [audit_file(p, min_steps) for p in files]
    bad = [r for r in file_reports if not r.get("schema_ok", False)]
    missing_attrs = [r for r in file_reports if r.get("missing_optional_attrs")]
    missing_freeze_attrs = [
        r
        for r in file_reports
        if any(
            name in (r.get("missing_optional_attrs") or [])
            for name in ["tac_quality_scorer_freeze_manifest_sha256", "tac_quality_scorer_freeze_git_commit"]
        )
    ]
    return {
        "path": str(path),
        "exists": path.exists(),
        "n_hdf5": len(files),
        "min_episodes": int(min_episodes),
        "min_steps": int(min_steps),
        "ready": bool(path.exists() and len(files) >= min_episodes and not bad),
        "schema_ok_files": len(files) - len(bad),
        "schema_bad_files": len(bad),
        "missing_optional_attr_files": len(missing_attrs),
        "missing_freeze_attr_files": len(missing_freeze_attrs),
        "bad_examples": bad[:5],
        "missing_attr_examples": [
            {
                "path": r["path"],
                "missing_optional_attrs": r.get("missing_optional_attrs", []),
            }
            for r in missing_attrs[:5]
        ],
        "missing_freeze_attr_examples": [
            {
                "path": r["path"],
                "missing_optional_attrs": r.get("missing_optional_attrs", []),
            }
            for r in missing_freeze_attrs[:5]
        ],
    }


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    launch = load_json(Path(args.launch_sheet))
    tasks: Dict[str, Any] = {}
    for task in TASKS:
        task_dirs = launch["tasks"][task]["rollout_dirs"]
        arms = {
            arm: audit_arm(Path(task_dirs[arm]), args.min_episodes, args.min_steps)
            for arm in ARMS
        }
        tasks[task] = {
            "ready": all(row["ready"] for row in arms.values()),
            "arms": arms,
        }
    all_ready = all(row["ready"] for row in tasks.values())
    return {
        "purpose": "Audit formal TacQuality rollout HDF5 schema before quality gates.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "launch_sheet": str(args.launch_sheet),
        "min_episodes": int(args.min_episodes),
        "min_steps": int(args.min_steps),
        "tasks": tasks,
        "all_tasks_ready": bool(all_ready),
        "next_required_step": (
            "Collect schema-complete HDF5 rollouts in every formal arm directory."
            if not all_ready
            else "Generate pairing/metadata CSVs and run post-collection pipeline."
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Rollout HDF5 Schema Audit",
        "",
        f"- all_tasks_ready: `{result['all_tasks_ready']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "| task | arm | ready | n_hdf5 | schema_bad | missing optional attrs | missing freeze attrs |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task, task_row in result["tasks"].items():
        for arm, row in task_row["arms"].items():
            lines.append(
                f"| {task} | {arm} | {row['ready']} | {row['n_hdf5']} | "
                f"{row['schema_bad_files']} | {row['missing_optional_attr_files']} | "
                f"{row['missing_freeze_attr_files']} |"
            )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min_episodes", type=int, default=10)
    parser.add_argument("--min_steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_rollout_hdf5_schema_audit.json"
    md_path = out_dir / "tac_quality_rollout_hdf5_schema_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "all_tasks_ready": result["all_tasks_ready"],
                "tasks": {
                    task: {
                        "ready": row["ready"],
                        "n_hdf5": {arm: arm_row["n_hdf5"] for arm, arm_row in row["arms"].items()},
                    }
                    for task, row in result["tasks"].items()
                },
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
