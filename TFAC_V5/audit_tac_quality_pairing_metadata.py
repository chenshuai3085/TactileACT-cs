"""Audit generated TacQuality pairing and metadata CSV completeness.

After real rollouts are collected, ``build_tac_quality_rollout_pairing.py``
creates concrete pairing and metadata CSVs.  This audit checks whether those
CSVs are complete enough for formal gates:

  - required CSV files exist;
  - two-arm and three-arm pairing rows are present;
  - paired HDF5 paths resolve under the formal rollout directories;
  - metadata rows cover every paired rollout file;
  - success/stopped_early metadata cells are filled.

It does not evaluate policy quality and does not claim scientific evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Set


DEFAULT_LAUNCH_SHEET = Path(
    "/home/chenshuai/Project/output/tac_quality_formal_launch_sheet/"
    "formal_paired12/tac_quality_formal_launch_sheet.json"
)
DEFAULT_PAIRING_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_pairing_metadata_audit")
FORMAL_ARMS = ("baseline", "default_guided", "distilled_guided")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_rollout(value: str, root: Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def metadata_keys(rows: List[Dict[str, str]]) -> Set[str]:
    keys: Set[str] = set()
    for row in rows:
        for col in ("path", "file"):
            value = (row.get(col) or "").strip()
            if value:
                p = Path(value)
                keys.update({str(p), p.name, p.stem})
        stem = (row.get("stem") or "").strip()
        if stem:
            keys.add(stem)
    return keys


def covered_by_metadata(path: Path, keys: Set[str]) -> bool:
    return str(path) in keys or path.name in keys or path.stem in keys


def blank_metadata_rows(rows: List[Dict[str, str]]) -> List[str]:
    out: List[str] = []
    for row in rows:
        success = (row.get("success") or row.get("task_success") or "").strip()
        stopped = (row.get("stopped_early") or row.get("early_stop") or "").strip()
        if success == "" or stopped == "":
            label = row.get("path") or row.get("file") or row.get("stem") or "<unknown>"
            out.append(str(label))
    return out


def audit_task(task: str, launch_task: Dict[str, Any], pairing_dir: Path, min_pairs: int) -> Dict[str, Any]:
    task_dir = pairing_dir / task
    two_csv = task_dir / "pairing_generated.csv"
    three_csv = task_dir / "three_arm_pairing_generated.csv"
    action_aware_csv = task_dir / "action_aware_pairing.csv"
    meta_csv = task_dir / "metadata_generated.csv"
    two_rows = read_csv(two_csv)
    three_rows = read_csv(three_csv)
    action_aware_rows = read_csv(action_aware_csv)
    meta_rows = read_csv(meta_csv)
    meta_keys = metadata_keys(meta_rows)
    rollout_dirs = {arm: Path(launch_task["rollout_dirs"][arm]) for arm in FORMAL_ARMS}
    if "action_aware_guided" in launch_task["rollout_dirs"]:
        rollout_dirs["action_aware_guided"] = Path(launch_task["rollout_dirs"]["action_aware_guided"])

    missing_files: List[str] = []
    missing_metadata: List[str] = []
    paired_paths: List[Path] = []
    for row in two_rows:
        for col, arm in (("baseline", "baseline"), ("guided", "default_guided")):
            path = resolve_rollout(row.get(col, ""), rollout_dirs[arm])
            paired_paths.append(path)
    for row in three_rows:
        for arm in FORMAL_ARMS:
            path = resolve_rollout(row.get(arm, ""), rollout_dirs[arm])
            paired_paths.append(path)
    action_aware_paths: List[Path] = []
    for row in action_aware_rows:
        for col, arm in (("baseline", "baseline"), ("guided", "action_aware_guided")):
            if arm not in rollout_dirs:
                continue
            path = resolve_rollout(row.get(col, ""), rollout_dirs[arm])
            action_aware_paths.append(path)
            paired_paths.append(path)
    for path in paired_paths:
        if not path.exists():
            missing_files.append(str(path))
        if not covered_by_metadata(path, meta_keys):
            missing_metadata.append(str(path))

    blank_rows = blank_metadata_rows(meta_rows)
    two_ready = len(two_rows) >= min_pairs and not missing_files and not missing_metadata
    three_ready = len(three_rows) >= min_pairs and not missing_files and not missing_metadata
    metadata_ready = len(meta_rows) >= len(set(str(p) for p in paired_paths)) and not blank_rows and not missing_metadata
    action_aware_ready = (
        bool(action_aware_rows)
        and len(action_aware_rows) >= min_pairs
        and all(path.exists() for path in action_aware_paths)
        and all(covered_by_metadata(path, meta_keys) for path in action_aware_paths)
    )
    ready = bool(two_ready and three_ready and metadata_ready)
    return {
        "task": task,
        "ready": ready,
        "two_arm_ready": bool(two_ready),
        "three_arm_ready": bool(three_ready),
        "metadata_ready": bool(metadata_ready),
        "action_aware_ready": bool(action_aware_ready),
        "min_pairs": int(min_pairs),
        "paths": {
            "pairing_csv": str(two_csv),
            "three_arm_pairing_csv": str(three_csv),
            "action_aware_pairing_csv": str(action_aware_csv),
            "metadata_csv": str(meta_csv),
        },
        "exists": {
            "pairing_csv": two_csv.exists(),
            "three_arm_pairing_csv": three_csv.exists(),
            "action_aware_pairing_csv": action_aware_csv.exists(),
            "metadata_csv": meta_csv.exists(),
        },
        "counts": {
            "two_arm_rows": len(two_rows),
            "three_arm_rows": len(three_rows),
            "action_aware_rows": len(action_aware_rows),
            "metadata_rows": len(meta_rows),
            "unique_paired_files": len(set(str(p) for p in paired_paths)),
            "missing_files": len(set(missing_files)),
            "missing_metadata": len(set(missing_metadata)),
            "blank_metadata_rows": len(blank_rows),
        },
        "examples": {
            "missing_files": sorted(set(missing_files))[:10],
            "missing_metadata": sorted(set(missing_metadata))[:10],
            "blank_metadata_rows": blank_rows[:10],
        },
    }


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    launch = load_json(Path(args.launch_sheet))
    tasks = {
        task: audit_task(task, launch["tasks"][task], Path(args.pairing_dir), args.min_pairs)
        for task in ["insertion", "board"]
    }
    ready = all(row["ready"] for row in tasks.values())
    return {
        "purpose": "Audit generated pairing/metadata CSV completeness before formal TacQuality gates.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "launch_sheet": str(args.launch_sheet),
        "pairing_dir": str(args.pairing_dir),
        "min_pairs": int(args.min_pairs),
        "tasks": tasks,
        "all_tasks_ready": bool(ready),
        "next_required_step": (
            "Collect HDF5 rollouts, regenerate pairing CSVs, and fill blank success/stopped_early metadata cells."
            if not ready
            else "Run formal gate runner with --use_generated_pairing."
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Pairing Metadata Audit",
        "",
        f"- all_tasks_ready: `{result['all_tasks_ready']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "| task | ready | two rows | three rows | action-aware rows | metadata rows | missing files | missing metadata | blank metadata |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, row in result["tasks"].items():
        c = row["counts"]
        lines.append(
            f"| {task} | {row['ready']} | {c['two_arm_rows']} | {c['three_arm_rows']} | "
            f"{c['action_aware_rows']} | {c['metadata_rows']} | {c['missing_files']} | "
            f"{c['missing_metadata']} | {c['blank_metadata_rows']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--pairing_dir", default=str(DEFAULT_PAIRING_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--min_pairs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_audit(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_pairing_metadata_audit.json"
    md_path = out_dir / "tac_quality_pairing_metadata_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "all_tasks_ready": result["all_tasks_ready"],
                "tasks": {
                    task: {
                        "ready": row["ready"],
                        "counts": row["counts"],
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
