"""Synthetic smoke for TacQuality metadata review-sheet helper."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.build_tac_quality_metadata_review_sheet import build as build_review_sheet  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_metadata_review_sheet_smoke")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mkdir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_metadata(pairing_dir: Path) -> None:
    fields = ["file", "path", "stem", "task", "arm", "success", "stopped_early"]
    for task in ["insertion", "board"]:
        rows = [
            {
                "file": f"{task}_ok.hdf5",
                "path": str(pairing_dir / task / f"{task}_ok.hdf5"),
                "stem": f"{task}_ok",
                "task": task,
                "arm": "baseline",
                "success": "1",
                "stopped_early": "0",
            },
            {
                "file": f"{task}_blank.hdf5",
                "path": str(pairing_dir / task / f"{task}_blank.hdf5"),
                "stem": f"{task}_blank",
                "task": task,
                "arm": "default_guided",
                "success": "",
                "stopped_early": "",
            },
            {
                "file": f"{task}_bad.hdf5",
                "path": str(pairing_dir / task / f"{task}_bad.hdf5"),
                "stem": f"{task}_bad",
                "task": task,
                "arm": "distilled_guided",
                "success": "maybe",
                "stopped_early": "0",
            },
        ]
        write_csv(pairing_dir / task / "metadata_generated.csv", fields, rows)


def make_completed_review(path: Path) -> None:
    fields = [
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
    rows: List[Dict[str, Any]] = []
    for task in ["insertion", "board"]:
        for suffix, success, stopped in [("blank", "true", "false"), ("bad", "false", "true")]:
            rows.append(
                {
                    "task": task,
                    "arm": "manual",
                    "file": f"{task}_{suffix}.hdf5",
                    "path": "",
                    "stem": f"{task}_{suffix}",
                    "current_success": "",
                    "current_stopped_early": "",
                    "review_success": success,
                    "review_stopped_early": stopped,
                    "reviewer": "smoke",
                    "review_note": "synthetic review value",
                }
            )
    write_csv(path, fields, rows)


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    mkdir_clean(out_root)
    pairing_dir = out_root / "pairing"
    make_metadata(pairing_dir)

    first_args = argparse.Namespace(
        pairing_dir=str(pairing_dir),
        output_dir=str(out_root / "review_out"),
        tag="first",
        completed_review_csv=None,
    )
    first = build_review_sheet(first_args)
    completed_review = out_root / "completed_review.csv"
    make_completed_review(completed_review)
    second_args = argparse.Namespace(
        pairing_dir=str(pairing_dir),
        output_dir=str(out_root / "review_out"),
        tag="merged",
        completed_review_csv=str(completed_review),
    )
    second = build_review_sheet(second_args)
    checks = {
        "first_detects_four_rows": first.get("n_review_needed") == 4,
        "first_not_scientific": first.get("scientific_evidence") is False,
        "second_applies_four_rows": second.get("n_review_applied") == 4,
        "second_merged_complete": second.get("all_metadata_complete_after_merge") is True,
        "does_not_auto_infer": "never infers" in " ".join(second.get("guardrails", [])).lower(),
    }
    summary = {
        "purpose": "Synthetic smoke for TacQuality metadata review-sheet helper.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "pairing_dir": str(pairing_dir),
        "completed_review_csv": str(completed_review),
        "first_summary": {
            "n_review_needed": first.get("n_review_needed"),
            "next_required_step": first.get("next_required_step"),
        },
        "second_summary": {
            "n_review_applied": second.get("n_review_applied"),
            "all_metadata_complete_after_merge": second.get("all_metadata_complete_after_merge"),
        },
    }
    json_path = out_root / "tac_quality_metadata_review_sheet_smoke.json"
    md_path = out_root / "tac_quality_metadata_review_sheet_smoke.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)
    print(
        json.dumps(
            {
                "overall_pass": summary["overall_pass"],
                "checks": checks,
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
        "# TacQuality Metadata Review Sheet Smoke",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- scientific_evidence: `{summary['scientific_evidence']}`",
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
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic")
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
