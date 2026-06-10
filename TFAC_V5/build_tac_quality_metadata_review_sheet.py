"""Build manual review sheets for TacQuality outcome metadata.

Generated pairing metadata copies ``success`` and ``stopped_early`` from HDF5
attrs when present.  Real collection may still leave blanks.  This helper makes
that manual step explicit:

  1. read generated metadata CSVs;
  2. write a compact review CSV containing only missing/invalid outcome rows;
  3. optionally merge a completed review CSV back into a metadata CSV.

It never infers outcomes from force, tactile, scorer outputs, or filenames.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_PAIRING_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_pairing/formal_paired12")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_metadata_review_sheet")
TASKS = ("insertion", "board")
BOOL_TRUE = {"1", "true", "yes", "y", "success", "succeeded", "pass"}
BOOL_FALSE = {"0", "false", "no", "n", "fail", "failed"}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def normalize_bool(value: Any) -> Optional[str]:
    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return "1"
    if text in BOOL_FALSE:
        return "0"
    try:
        numeric = float(text)
    except ValueError:
        return None
    if math.isclose(numeric, 1.0):
        return "1"
    if math.isclose(numeric, 0.0):
        return "0"
    return None


def valid_bool_cell(value: Any) -> bool:
    if value is None:
        return False
    return normalize_bool(value) is not None


def row_key(row: Dict[str, str]) -> str:
    for key in ("path", "file", "stem"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def review_needed(row: Dict[str, str]) -> bool:
    return not (valid_bool_cell(row.get("success")) and valid_bool_cell(row.get("stopped_early")))


def build_review_rows(task: str, rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not review_needed(row):
            continue
        out.append(
            {
                "task": task,
                "arm": row.get("arm", ""),
                "file": row.get("file", ""),
                "path": row.get("path", ""),
                "stem": row.get("stem", ""),
                "current_success": row.get("success", ""),
                "current_stopped_early": row.get("stopped_early", ""),
                "review_success": "",
                "review_stopped_early": "",
                "reviewer": "",
                "review_note": "",
            }
        )
    return out


def load_review(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    _, rows = read_csv(Path(path))
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        key = row_key(row)
        success = normalize_bool(row.get("review_success", ""))
        stopped = normalize_bool(row.get("review_stopped_early", ""))
        if key and success is not None and stopped is not None:
            out[key] = {
                "success": success,
                "stopped_early": stopped,
                "reviewer": row.get("reviewer", ""),
                "review_note": row.get("review_note", ""),
            }
    return out


def merge_metadata(rows: List[Dict[str, str]], review: Dict[str, Dict[str, str]]) -> Tuple[List[Dict[str, str]], int]:
    merged: List[Dict[str, str]] = []
    n_applied = 0
    for row in rows:
        item = dict(row)
        keys = [row_key(row), row.get("file", ""), row.get("stem", ""), row.get("path", "")]
        replacement = None
        for key in keys:
            if key and key in review:
                replacement = review[key]
                break
        if replacement and review_needed(item):
            item["success"] = replacement["success"]
            item["stopped_early"] = replacement["stopped_early"]
            n_applied += 1
        merged.append(item)
    return merged, n_applied


def process_task(task: str, pairing_dir: Path, out_dir: Path, review: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    metadata_csv = pairing_dir / task / "metadata_generated.csv"
    fieldnames, rows = read_csv(metadata_csv)
    review_rows = build_review_rows(task, rows)
    review_csv = out_dir / task / "metadata_review_needed.csv"
    review_fields = [
        "task",
        "arm",
        "file",
        "path",
        "stem",
        "current_success",
        "current_stopped_early",
        "review_success",
        "review_stopped_early",
        "reviewer",
        "review_note",
    ]
    write_csv(review_csv, review_fields, review_rows)

    merged_csv = out_dir / task / "metadata_merged.csv"
    merged_rows, n_applied = merge_metadata(rows, review)
    merged_complete = bool(merged_rows and all(not review_needed(row) for row in merged_rows))
    if rows:
        write_csv(merged_csv, fieldnames or ["file", "path", "stem", "task", "arm", "success", "stopped_early"], merged_rows)
    return {
        "task": task,
        "metadata_csv": str(metadata_csv),
        "metadata_exists": metadata_csv.exists(),
        "n_metadata_rows": len(rows),
        "n_review_needed": len(review_rows),
        "review_csv": str(review_csv),
        "merged_csv": str(merged_csv) if rows else None,
        "n_review_applied": int(n_applied),
        "merged_complete": merged_complete,
        "review_examples": review_rows[:10],
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    pairing_dir = Path(args.pairing_dir)
    out_dir = Path(args.output_dir) / args.tag
    review = load_review(args.completed_review_csv)
    tasks = {
        task: process_task(task, pairing_dir, out_dir, review)
        for task in TASKS
    }
    n_review_needed = sum(row["n_review_needed"] for row in tasks.values())
    n_review_applied = sum(row["n_review_applied"] for row in tasks.values())
    result = {
        "purpose": "Build manual review sheets for missing/invalid TacQuality success/stopped_early metadata.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "pairing_dir": str(pairing_dir),
        "completed_review_csv": args.completed_review_csv,
        "tag": args.tag,
        "tasks": tasks,
        "n_review_needed": int(n_review_needed),
        "n_review_applied": int(n_review_applied),
        "all_metadata_review_ready": bool(all(row["metadata_exists"] for row in tasks.values())),
        "all_metadata_complete_after_merge": bool(all(row["merged_complete"] for row in tasks.values())),
        "next_required_step": (
            "Fill review_success/review_stopped_early in metadata_review_needed.csv, then rerun with --completed_review_csv."
            if n_review_needed and not args.completed_review_csv
            else (
                "Use metadata_merged.csv or copy reviewed values back to metadata_generated.csv, then rerun metadata audit."
                if n_review_applied
                else "No metadata review rows needed."
            )
        ),
        "guardrails": [
            "This helper never infers success/stopped_early from force, tactile, scorer outputs, or filenames.",
            "Only explicit review_success/review_stopped_early values are merged.",
            "Formal gates still require audit_tac_quality_pairing_metadata.py and --require_outcome_metadata.",
        ],
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Metadata Review Sheet",
        "",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- n_review_needed: `{result['n_review_needed']}`",
        f"- n_review_applied: `{result['n_review_applied']}`",
        f"- all_metadata_complete_after_merge: `{result['all_metadata_complete_after_merge']}`",
        f"- next_required_step: {result['next_required_step']}",
        "",
        "| task | metadata rows | review needed | review applied | merged complete | review csv |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task, row in result["tasks"].items():
        lines.append(
            f"| {task} | {row['n_metadata_rows']} | {row['n_review_needed']} | "
            f"{row['n_review_applied']} | {row['merged_complete']} | `{row['review_csv']}` |"
        )
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairing_dir", default=str(DEFAULT_PAIRING_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="formal_paired12")
    parser.add_argument("--completed_review_csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args)
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_metadata_review_sheet.json"
    md_path = out_dir / "tac_quality_metadata_review_sheet.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "n_review_needed": result["n_review_needed"],
                "n_review_applied": result["n_review_applied"],
                "all_metadata_complete_after_merge": result["all_metadata_complete_after_merge"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
