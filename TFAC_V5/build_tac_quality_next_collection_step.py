"""Build the next executable formal TacQuality collection step.

The collection progress report already tracks the next pending schedule row.
This script turns that row into a small operator-facing artifact with:

  - the exact server launch command;
  - the required HDF5 save path;
  - post-run verification commands;
  - a lightweight schema check if the recommended HDF5 already exists.

It does not evaluate quality and does not claim scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import h5py


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_next_collection_step")
DEFAULT_PROGRESS = Path(
    "/home/chenshuai/Project/output/tac_quality_collection_progress/"
    "formal_paired12/tac_quality_collection_progress.json"
)
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


def first_existing(f: h5py.File, keys: tuple[str, ...]) -> Optional[Dict[str, Any]]:
    for key in keys:
        if key in f:
            d = f[key]
            return {"key": key, "shape": [int(x) for x in d.shape], "dtype": str(d.dtype)}
    return None


def valid_len(info: Optional[Dict[str, Any]], min_steps: int) -> bool:
    return bool(info and info.get("shape") and info["shape"][0] >= min_steps)


def audit_recommended_hdf5(path: Path, min_steps: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "schema_ok": False,
            "reason": "recommended HDF5 does not exist yet",
        }
    try:
        with h5py.File(path, "r") as f:
            force = first_existing(f, FORCE_KEYS)
            left_marker = first_existing(f, ("observations/tac/left/marker_offset",))
            right_marker = first_existing(f, ("observations/tac/right/marker_offset",))
            action = first_existing(f, ACTION_KEYS)
            attrs = {
                "success": "success" in f.attrs,
                "stopped_early": "stopped_early" in f.attrs,
            }
    except Exception as exc:
        return {
            "path": str(path),
            "exists": True,
            "schema_ok": False,
            "reason": f"failed to open/read HDF5: {exc}",
        }
    missing = []
    if not valid_len(force, min_steps):
        missing.append("force_source")
    if not valid_len(left_marker, min_steps):
        missing.append("observations/tac/left/marker_offset")
    if not valid_len(right_marker, min_steps):
        missing.append("observations/tac/right/marker_offset")
    if not valid_len(action, min_steps):
        missing.append("action_source")
    return {
        "path": str(path),
        "exists": True,
        "schema_ok": not missing,
        "missing_or_short_required_fields": missing,
        "force_source": force,
        "left_marker": left_marker,
        "right_marker": right_marker,
        "action_source": action,
        "attrs": attrs,
        "missing_optional_attrs": [name for name, present in attrs.items() if not present],
    }


def with_save_path_hint(command: str, recommended_path: Path) -> str:
    # The server command may not expose a stable CLI flag for output file naming.
    # Keep the original launch command intact and add a shell-visible variable
    # for the operator/recorder wrapper to consume.
    return f"TACQUALITY_RECOMMENDED_HDF5={shlex.quote(str(recommended_path))} {command}"


def build(args: argparse.Namespace) -> Dict[str, Any]:
    progress = load_json(Path(args.progress))
    next_row = progress.get("next_row")
    recommended_path = Path(next_row["recommended_path"]) if next_row else None
    hdf5_audit = audit_recommended_hdf5(recommended_path, args.min_steps) if recommended_path else None
    launch_command = next_row.get("launch_command") if next_row else None
    launch_with_hint = with_save_path_hint(launch_command, recommended_path) if next_row else None
    result = {
        "purpose": "Next executable formal TacQuality rollout collection step.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "progress": str(args.progress),
        "progress_pass": progress.get("progress_pass"),
        "ready_for_post_collection": progress.get("ready_for_post_collection"),
        "n_completed_rows": progress.get("n_completed_rows"),
        "n_scheduled_rows": progress.get("n_scheduled_rows"),
        "has_next_step": next_row is not None,
        "next_row": next_row,
        "recommended_path": str(recommended_path) if recommended_path else None,
        "recommended_path_exists": recommended_path.exists() if recommended_path else None,
        "launch_command": launch_command,
        "launch_command_with_save_path_hint": launch_with_hint,
        "hdf5_audit": hdf5_audit,
        "post_run_commands": [
            "python TFAC_V5/build_tac_quality_collection_progress.py",
            "python TFAC_V5/audit_tac_quality_rollout_hdf5_schema.py",
            "python TFAC_V5/build_tac_quality_rollout_pairing.py --tag formal_paired12",
            "python TFAC_V5/run_tac_quality_post_collection_pipeline.py --tag formal_paired12",
        ],
        "next_required_step": (
            "Collect the next scheduled rollout and save the HDF5 exactly at recommended_path."
            if next_row
            else "All scheduled rows are covered; run the post-collection pipeline with --run_gates after metadata review."
        ),
        "guardrails": [
            "Do not rename the HDF5 after collection unless the schedule/progress artifacts are regenerated.",
            "Keep all three arms within a pair_id under matched initial setup before moving to the next pair.",
            "This next-step artifact is not quality evidence; only formal real-rollout gates can close the blocker.",
        ],
    }
    result["next_step_pass"] = bool(
        progress.get("progress_pass") is True
        and result["scientific_evidence"] is False
        and (
            (
                next_row is not None
                and str(next_row.get("recommended_filename", "")).endswith(".hdf5")
                and str(next_row.get("recommended_path", "")).endswith(".hdf5")
                and "serve_dp_tac_quality_guided" in str(launch_command)
                and recommended_path is not None
                and str(recommended_path).endswith(str(next_row.get("recommended_filename")))
            )
            or progress.get("ready_for_post_collection") is True
        )
    )
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Next Collection Step",
        "",
        f"- next_step_pass: `{result['next_step_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- completed: `{result['n_completed_rows']}` / `{result['n_scheduled_rows']}`",
        f"- ready_for_post_collection: `{result['ready_for_post_collection']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
    ]
    row = result.get("next_row")
    if row:
        lines.extend(
            [
                "## Collect This Row",
                "",
                f"- global_step: `{row['global_step']}`",
                f"- task: `{row['task']}`",
                f"- pair_id: `{row['pair_id']}`",
                f"- within_pair_order: `{row['within_pair_order']}`",
                f"- arm: `{row['arm']}`",
                f"- save_as: `{row['recommended_filename']}`",
                f"- recommended_path: `{row['recommended_path']}`",
                f"- path_exists_now: `{result['recommended_path_exists']}`",
                "",
                "Launch command with save-path hint:",
                "",
                "```bash",
                result["launch_command_with_save_path_hint"],
                "```",
                "",
                "Original launch command:",
                "",
                "```bash",
                result["launch_command"],
                "```",
                "",
            ]
        )
        audit = result.get("hdf5_audit") or {}
        lines.extend(
            [
                "## Current HDF5 Check",
                "",
                f"- exists: `{audit.get('exists')}`",
                f"- schema_ok: `{audit.get('schema_ok')}`",
                f"- missing_or_short_required_fields: `{audit.get('missing_or_short_required_fields')}`",
                f"- missing_optional_attrs: `{audit.get('missing_optional_attrs')}`",
                "",
            ]
        )
    else:
        lines.extend(["## Collect This Row", "", "No pending scheduled row remains.", ""])
    lines.extend(["## After Collection", ""])
    for command in result["post_run_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", default=str(DEFAULT_PROGRESS))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--min_steps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_next_collection_step.json"
    md_path = out_dir / "tac_quality_next_collection_step.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "next_step_pass": result["next_step_pass"],
                "scientific_evidence": result["scientific_evidence"],
                "has_next_step": result["has_next_step"],
                "recommended_path": result["recommended_path"],
                "recommended_path_exists": result["recommended_path_exists"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
