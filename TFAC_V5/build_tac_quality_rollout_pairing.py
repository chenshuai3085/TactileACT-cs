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
ARMS = ("baseline", "default_guided", "distilled_guided")


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


def pair_count(paths_by_arm: Dict[str, List[Path]], requested: int) -> int:
    available = min(len(paths_by_arm[arm]) for arm in ARMS)
    if requested <= 0:
        return available
    return min(available, requested)


def build_for_task(
    task: str,
    rollout_dirs: Dict[str, str],
    out_dir: Path,
    *,
    max_pairs: int,
) -> Dict[str, Any]:
    roots = {arm: Path(rollout_dirs[arm]) for arm in ARMS}
    paths_by_arm = {arm: discover_hdf5(root) if root.exists() else [] for arm, root in roots.items()}
    n_pairs = pair_count(paths_by_arm, max_pairs)

    two_arm_rows: List[Dict[str, Any]] = []
    three_arm_rows: List[Dict[str, Any]] = []
    metadata_rows: List[Dict[str, Any]] = []

    for idx in range(n_pairs):
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

    task_dir = out_dir / task
    two_arm_csv = task_dir / "pairing_generated.csv"
    three_arm_csv = task_dir / "three_arm_pairing_generated.csv"
    metadata_csv = task_dir / "metadata_generated.csv"
    write_csv(two_arm_csv, ["pair_id", "baseline", "guided"], two_arm_rows)
    write_csv(
        three_arm_csv,
        ["pair_id", "baseline", "default_guided", "distilled_guided"],
        three_arm_rows,
    )
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
    ready = n_pairs > 0 and all(len(paths_by_arm[arm]) >= n_pairs for arm in ARMS)
    return {
        "task": task,
        "ready": bool(ready),
        "n_pairs": int(n_pairs),
        "max_pairs": int(max_pairs),
        "roots": {arm: str(root) for arm, root in roots.items()},
        "n_hdf5": {arm: len(paths_by_arm[arm]) for arm in ARMS},
        "outputs": {
            "pairing_csv": str(two_arm_csv),
            "three_arm_pairing_csv": str(three_arm_csv),
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
    out_dir = Path(args.output_dir) / args.tag
    tasks = {
        task: build_for_task(
            task,
            launch["tasks"][task]["rollout_dirs"],
            out_dir,
            max_pairs=args.max_pairs,
        )
        for task in ["insertion", "board"]
    }
    return {
        "purpose": "Generate concrete pairing/metadata CSVs from collected TacQuality rollout HDF5s.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "launch_sheet": str(args.launch_sheet),
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
        "| task | ready | pairs | baseline n | default n | distilled n | missing metadata attrs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, row in result["tasks"].items():
        lines.append(
            f"| {task} | {row['ready']} | {row['n_pairs']} | "
            f"{row['n_hdf5']['baseline']} | {row['n_hdf5']['default_guided']} | "
            f"{row['n_hdf5']['distilled_guided']} | {row['metadata_missing_attr_rows']} |"
        )
    lines.extend(["", "## Outputs", ""])
    for task, row in result["tasks"].items():
        lines.append(f"### {task}")
        for name, output in row["outputs"].items():
            lines.append(f"- {name}: `{output}`")
        lines.append(f"- manual_review: {row['next_manual_review']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch_sheet", default=str(DEFAULT_LAUNCH_SHEET))
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
