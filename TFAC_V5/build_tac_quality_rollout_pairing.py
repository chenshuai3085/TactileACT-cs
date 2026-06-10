"""Build pairing and metadata CSVs from collected TacQuality rollout HDF5s.

The formal experiment packet ships template CSVs with placeholder episode
names.  After collecting real rollouts, this script inspects the actual
baseline/default/distilled directories and writes concrete two-arm and
three-arm pairing files for the final quality gates.

It does not infer task success when metadata is absent.  It only copies
``success`` and ``stopped_early`` attributes if they exist in the HDF5 files;
otherwise the metadata cells are left blank for manual review.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_pairing")
DEFAULT_SCHEDULE = Path(
    "/home/chenshuai/Project/output/tac_quality_collection_schedule/"
    "formal_paired12/tac_quality_collection_schedule.json"
)
FORMAL_ARMS = ("baseline", "default_guided", "distilled_guided")
OPTIONAL_ARMS = ("action_aware_guided",)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def rel_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def read_attr(path: Path, name: str) -> Optional[Any]:
    try:
        with h5py.File(path, "r") as f:
            if name in f.attrs:
                value = f.attrs[name]
                if hasattr(value, "item"):
                    value = value.item()
                return value
    except Exception:
        return None
    return None


def format_attr(value: Optional[Any]) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    return str(value)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def pair_count(paths_by_arm: Dict[str, List[Path]], arms: tuple[str, ...], requested: int) -> int:
    available = min(len(paths_by_arm[arm]) for arm in arms)
    if requested <= 0:
        return available
    return min(available, requested)


def scheduled_paths_for_task(schedule: Optional[Dict[str, Any]], task: str) -> Optional[Dict[str, Dict[str, Path]]]:
    if not schedule:
        return None
    rows = schedule.get("long_schedule_rows", [])
    by_pair: Dict[str, Dict[str, Path]] = {}
    for row in rows:
        if row.get("task") != task:
            continue
        arm = row.get("arm")
        pair_id = row.get("pair_id")
        recommended_path = row.get("recommended_path")
        if arm not in FORMAL_ARMS or not pair_id or not recommended_path:
            continue
        by_pair.setdefault(pair_id, {})[arm] = Path(recommended_path)
    return by_pair or None


def build_for_task(
    task: str,
    rollout_dirs: Dict[str, str],
    out_dir: Path,
    *,
    max_pairs: int,
    schedule: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    roots = {arm: Path(rollout_dirs[arm]) for arm in FORMAL_ARMS if arm in rollout_dirs}
    for arm in OPTIONAL_ARMS:
        if arm in rollout_dirs:
            roots[arm] = Path(rollout_dirs[arm])
    paths_by_arm = {arm: discover_hdf5(root) if root.exists() else [] for arm, root in roots.items()}
    scheduled_by_pair = scheduled_paths_for_task(schedule, task)
    schedule_mode = scheduled_by_pair is not None
    ordered_pair_ids: List[str] = []
    if schedule_mode:
        ordered_pair_ids = sorted(scheduled_by_pair)
        if max_pairs > 0:
            ordered_pair_ids = ordered_pair_ids[:max_pairs]
        complete_pair_ids = [
            pair_id
            for pair_id in ordered_pair_ids
            if all(scheduled_by_pair[pair_id].get(arm, Path("__missing__")).exists() for arm in FORMAL_ARMS)
        ]
        n_pairs = len(complete_pair_ids)
    else:
        complete_pair_ids = []
        n_pairs = pair_count(paths_by_arm, FORMAL_ARMS, max_pairs)
    action_aware_available = "action_aware_guided" in paths_by_arm
    action_aware_pairs = (
        pair_count(paths_by_arm, ("baseline", "action_aware_guided"), max_pairs)
        if action_aware_available
        else 0
    )

    two_arm_rows: List[Dict[str, Any]] = []
    three_arm_rows: List[Dict[str, Any]] = []
    action_aware_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []
    metadata_seen: set[str] = set()

    def add_metadata(arm: str, path: Path) -> None:
        key = f"{arm}:{path}"
        if key in metadata_seen:
            return
        metadata_seen.add(key)
        metadata_rows.append(
            {
                "file": path.name,
                "path": str(path),
                "stem": path.stem,
                "task": task,
                "arm": arm,
                "success": format_attr(read_attr(path, "success")),
                "stopped_early": format_attr(read_attr(path, "stopped_early")),
            }
        )

    for idx in range(n_pairs):
        if schedule_mode:
            pair_id = complete_pair_ids[idx]
            baseline = scheduled_by_pair[pair_id]["baseline"]
            default_guided = scheduled_by_pair[pair_id]["default_guided"]
            distilled_guided = scheduled_by_pair[pair_id]["distilled_guided"]
        else:
            pair_id = f"trial_{idx + 1:03d}"
            baseline = paths_by_arm["baseline"][idx]
            default_guided = paths_by_arm["default_guided"][idx]
            distilled_guided = paths_by_arm["distilled_guided"][idx]
        two_arm_rows.append(
            {
                "pair_id": pair_id,
                "baseline": rel_or_name(baseline, roots["baseline"]),
                "guided": rel_or_name(default_guided, roots["default_guided"]),
            }
        )
        three_arm_rows.append(
            {
                "pair_id": pair_id,
                "baseline": rel_or_name(baseline, roots["baseline"]),
                "default_guided": rel_or_name(default_guided, roots["default_guided"]),
                "distilled_guided": rel_or_name(distilled_guided, roots["distilled_guided"]),
            }
        )
        for arm, path in (
            ("baseline", baseline),
            ("default_guided", default_guided),
            ("distilled_guided", distilled_guided),
        ):
            add_metadata(arm, path)

    for idx in range(action_aware_pairs):
        pair_id = f"trial_{idx + 1:03d}"
        baseline = paths_by_arm["baseline"][idx]
        action_aware = paths_by_arm["action_aware_guided"][idx]
        action_aware_rows.append(
            {
                "pair_id": pair_id,
                "baseline": rel_or_name(baseline, roots["baseline"]),
                "guided": rel_or_name(action_aware, roots["action_aware_guided"]),
            }
        )
        add_metadata("baseline", baseline)
        add_metadata("action_aware_guided", action_aware)

    task_dir = out_dir / task
    two_arm_csv = task_dir / "pairing_generated.csv"
    three_arm_csv = task_dir / "three_arm_pairing_generated.csv"
    action_aware_csv = task_dir / "action_aware_pairing.csv"
    metadata_csv = task_dir / "metadata_generated.csv"
    write_csv(two_arm_csv, ["pair_id", "baseline", "guided"], two_arm_rows)
    write_csv(
        three_arm_csv,
        ["pair_id", "baseline", "default_guided", "distilled_guided"],
        three_arm_rows,
    )
    write_csv(action_aware_csv, ["pair_id", "baseline", "guided"], action_aware_rows)
    write_csv(
        metadata_csv,
        ["file", "path", "stem", "task", "arm", "success", "stopped_early"],
        metadata_rows,
    )

    missing_attrs = [
        f"{row['task']}/{row['arm']}/{row['file']}"
        for row in metadata_rows
        if row["success"] == "" or row["stopped_early"] == ""
    ]
    if schedule_mode:
        ready = n_pairs > 0 and len(complete_pair_ids) == len(ordered_pair_ids)
    else:
        ready = n_pairs > 0 and all(len(paths_by_arm[arm]) >= n_pairs for arm in FORMAL_ARMS)
    action_aware_ready = action_aware_pairs > 0 and action_aware_available
    return {
        "task": task,
        "ready": bool(ready),
        "action_aware_ready": bool(action_aware_ready),
        "n_pairs": int(n_pairs),
        "action_aware_pairs": int(action_aware_pairs),
        "max_pairs": int(max_pairs),
        "schedule_mode": bool(schedule_mode),
        "scheduled_pairs": len(ordered_pair_ids) if schedule_mode else None,
        "complete_scheduled_pairs": len(complete_pair_ids) if schedule_mode else None,
        "roots": {arm: str(root) for arm, root in roots.items()},
        "n_hdf5": {arm: len(paths_by_arm.get(arm, [])) for arm in (*FORMAL_ARMS, *OPTIONAL_ARMS)},
        "outputs": {
            "pairing_csv": str(two_arm_csv),
            "three_arm_pairing_csv": str(three_arm_csv),
            "action_aware_pairing_csv": str(action_aware_csv),
            "metadata_csv": str(metadata_csv),
        },
        "metadata_missing_attr_rows": len(missing_attrs),
        "metadata_missing_attr_examples": missing_attrs[:10],
        "next_manual_review": (
            "Fill blank success/stopped_early cells in metadata_generated.csv if HDF5 attrs are absent."
            if missing_attrs
            else "Metadata attributes were copied from HDF5 attrs."
        ),
    }


def build_pairings(args: argparse.Namespace) -> Dict[str, Any]:
    launch = load_json(Path(args.launch_sheet))
    schedule = load_json(Path(args.schedule)) if Path(args.schedule).exists() else None
    out_dir = Path(args.output_dir) / args.tag
    tasks = {
        task: build_for_task(
            task,
            launch["tasks"][task]["rollout_dirs"],
            out_dir,
            max_pairs=args.max_pairs,
            schedule=schedule,
        )
        for task in ["insertion", "board"]
    }
    return {
        "purpose": "Generate concrete pairing/metadata CSVs from collected TacQuality rollout HDF5s.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "launch_sheet": str(args.launch_sheet),
        "schedule": str(args.schedule),
        "schedule_used": schedule is not None,
        "tag": args.tag,
        "tasks": tasks,
        "overall_ready": all(row["ready"] for row in tasks.values()),
        "next_required_step": (
            "Use generated pairing/metadata CSVs with TFAC_V5/run_formal_tac_quality_rollout_gates.py --run_gates "
            "after manually filling blank metadata cells if any."
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Rollout Pairing",
        "",
        f"- overall_ready: `{result['overall_ready']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "| task | ready | pairs | baseline n | default n | distilled n | action-aware n | action-aware pairs | missing metadata attrs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for task, row in result["tasks"].items():
        lines.append(
            f"| {task} | {row['ready']} | {row['n_pairs']} | "
            f"{row['n_hdf5']['baseline']} | {row['n_hdf5']['default_guided']} | "
            f"{row['n_hdf5']['distilled_guided']} | {row['n_hdf5']['action_aware_guided']} | "
            f"{row['action_aware_pairs']} | {row['metadata_missing_attr_rows']} |"
        )
    lines.extend(["", "## Outputs", ""])
    for task, row in result["tasks"].items():
        lines.append(f"### {task}")
        for name, output in row["outputs"].items():
            lines.append(f"- {name}: `{output}`")
        lines.append(f"- manual_review: {row['next_manual_review']}")
        lines.append(f"- schedule_mode: `{row['schedule_mode']}`")
        if row["schedule_mode"]:
            lines.append(
                f"- complete_scheduled_pairs: `{row['complete_scheduled_pairs']}` / `{row['scheduled_pairs']}`"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
    parser.add_argument("--schedule", default=str(DEFAULT_SCHEDULE))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--max_pairs", type=int, default=0, help="0 means use all complete triplets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_pairings(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_rollout_pairing.json"
    md_path = out_dir / "tac_quality_rollout_pairing.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_ready": result["overall_ready"],
                "tasks": {
                    task: {
                        "ready": row["ready"],
                        "n_pairs": row["n_pairs"],
                        "schedule_mode": row["schedule_mode"],
                        "scheduled_pairs": row["scheduled_pairs"],
                        "complete_scheduled_pairs": row["complete_scheduled_pairs"],
                        "action_aware_pairs": row["action_aware_pairs"],
                        "n_hdf5": row["n_hdf5"],
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
