"""Build a formal real-rollout experiment packet for TacQuality validation.

The packet is a collection checklist, CSV templates, and exact commands for
the two remaining completion blockers:

  1. socket insertion baseline-vs-guided rollout validation;
  2. board wiping baseline-vs-guided rollout validation.

It does not run a robot and does not claim validation.  It makes the formal
collection protocol explicit and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


OUT_DIR = Path("/home/chenshuai/Project/output/real_rollout_experiment_packet")
TASKS = ["insertion", "board"]
DEFAULT_PLAN_ROOT = Path("/home/chenshuai/Project/output/real_rollout_sample_size_plan")


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def task_plan_path(task: str, plan_root: Path) -> Path:
    return plan_root / f"{task}_default_plan" / "real_rollout_sample_size_plan.json"


def write_pairing_template(path: Path, n_pairs: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_id", "baseline", "guided"])
        for i in range(1, n_pairs + 1):
            writer.writerow([f"trial_{i:03d}", f"episode_{i:03d}.hdf5", f"episode_{i:03d}.hdf5"])


def write_metadata_template(path: Path, n_pairs: int) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stem", "success", "stopped_early"])
        for i in range(1, n_pairs + 1):
            writer.writerow([f"episode_{i:03d}", "", ""])


def task_packet(task: str, out_dir: Path, plan_root: Path) -> Dict:
    plan = load_json(task_plan_path(task, plan_root))
    n_pairs = int(plan["recommendation"]["paired_n_pairs"])
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    pairing_csv = task_dir / "pairing_template.csv"
    baseline_metadata = task_dir / "baseline_metadata_template.csv"
    guided_metadata = task_dir / "guided_metadata_template.csv"
    merged_metadata = task_dir / "metadata_template.csv"
    write_pairing_template(pairing_csv, n_pairs)
    write_metadata_template(baseline_metadata, n_pairs)
    write_metadata_template(guided_metadata, n_pairs)
    write_metadata_template(merged_metadata, n_pairs)

    baseline_dir = f"<{task}_baseline_rollout_dir>"
    guided_dir = f"<{task}_guided_rollout_dir>"
    prepare_cmd = (
        "python TFAC_V5/prepare_real_rollout_validation.py "
        f"--task {task} "
        f"--baseline_dir {baseline_dir} "
        f"--guided_dir {guided_dir} "
        f"--pairing_csv {pairing_csv} "
        f"--metadata_csv {merged_metadata} "
        "--output_dir /home/chenshuai/Project/output/real_rollout_validation_ready "
        f"--tag {task}_formal_ready"
    )
    gate_cmd = (
        "python TFAC_V5/eval_real_rollout_quality_gate.py "
        f"--task {task} "
        f"--baseline_dir {baseline_dir} "
        f"--guided_dir {guided_dir} "
        f"--pairing_csv {pairing_csv} "
        f"--metadata_csv {merged_metadata} "
        "--output_dir /home/chenshuai/Project/output/real_rollout_quality_gate "
        f"--tag {task}_baseline_vs_guided "
        "--min_episodes 10 "
        "--bootstrap_samples 2000"
    )
    checklist = [
        f"Collect {n_pairs} paired baseline DP rollouts for {task}.",
        f"Collect {n_pairs} paired TacQuality-guided DP rollouts for {task}.",
        "Use identical initial conditions / task setup within each pair when possible.",
        "Store HDF5 files with matching stems, or edit pairing_template.csv.",
        "Fill metadata_template.csv with success and stopped_early for every rollout stem.",
        "Run the prepare command and resolve all blocking issues.",
        "Run the gate command and inspect production_validation_pass.",
    ]
    task_readme = task_dir / "README.md"
    task_readme.write_text(
        "\n".join(
            [
                f"# {task} Real Rollout Validation Packet",
                "",
                f"- paired_n_pairs: `{n_pairs}`",
                f"- unpaired_n_per_group: `{plan['recommendation']['unpaired_n_per_group']}`",
                "",
                "## Checklist",
                "",
                *[f"- {x}" for x in checklist],
                "",
                "## Prepare",
                "",
                "```bash",
                prepare_cmd,
                "```",
                "",
                "## Gate",
                "",
                "```bash",
                gate_cmd,
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "task": task,
        "task_dir": str(task_dir),
        "paired_n_pairs": n_pairs,
        "unpaired_n_per_group": int(plan["recommendation"]["unpaired_n_per_group"]),
        "pairing_template": str(pairing_csv),
        "metadata_template": str(merged_metadata),
        "baseline_metadata_template": str(baseline_metadata),
        "guided_metadata_template": str(guided_metadata),
        "prepare_command": prepare_cmd,
        "gate_command": gate_cmd,
        "checklist": checklist,
        "readme": str(task_readme),
    }


def build(args: argparse.Namespace) -> Dict:
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task_packet(task, out_dir, Path(args.plan_root)) for task in TASKS]
    result = {
        "scope": "Formal collection packet only; no robot validation is claimed.",
        "tag": args.tag,
        "out_dir": str(out_dir),
        "tasks": {row["task"]: row for row in tasks},
        "next_step_after_collection": "Run each task's prepare_command, then gate_command.",
    }
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# TacQuality Formal Real Rollout Experiment Packet",
                "",
                "This packet covers the remaining completion blockers.",
                "",
                "## Tasks",
                "",
                *[
                    f"- {row['task']}: collect {row['paired_n_pairs']} paired baseline/guided trials"
                    for row in tasks
                ],
                "",
                "## Order",
                "",
                "1. Collect HDF5 rollouts.",
                "2. Fill pairing and metadata CSVs.",
                "3. Run prepare commands.",
                "4. Run gate commands.",
                "5. Re-run `TFAC_V5/audit_tac_quality_goal_completion.py`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result["readme"] = str(readme)
    json_path = out_dir / "real_rollout_experiment_packet.json"
    result["json"] = str(json_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--plan_root", default=str(DEFAULT_PLAN_ROOT))
    return parser.parse_args()


def main() -> None:
    result = build(parse_args())
    print(
        json.dumps(
            {
                "out_dir": result["out_dir"],
                "tasks": {
                    task: {
                        "paired_n_pairs": row["paired_n_pairs"],
                        "readme": row["readme"],
                    }
                    for task, row in result["tasks"].items()
                },
                "json": result["json"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
