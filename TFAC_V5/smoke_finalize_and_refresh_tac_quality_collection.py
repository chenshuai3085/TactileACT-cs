"""Synthetic smoke for finalize-and-refresh TacQuality collection wrapper.

The smoke creates an isolated one-row next-step artifact and a synthetic HDF5,
then verifies that ``finalize_and_refresh_tac_quality_collection.py`` can
finalize the file without touching formal rollout directories.

This is not scientific evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.finalize_and_refresh_tac_quality_collection import build as build_finalize_and_refresh  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_finalize_and_refresh_smoke")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def mkdir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_hdf5(path: Path, steps: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        left = f.create_group("observations").create_group("tac").create_group("left")
        right = f["observations/tac"].create_group("right")
        left.create_dataset("marker_offset", data=rng.normal(size=(steps, 9, 9, 2)).astype("float32"))
        right.create_dataset("marker_offset", data=rng.normal(size=(steps, 9, 9, 2)).astype("float32"))
        left.create_dataset("force6d", data=rng.normal(size=(steps, 6)).astype("float32"))
        actions = f.create_group("actions")
        actions.create_dataset("joint_abs", data=rng.normal(size=(steps, 7)).astype("float32"))
        f.attrs["success"] = True
        f.attrs["stopped_early"] = False


def make_next_step(path: Path, target: Path) -> None:
    row = {
        "global_step": 1,
        "task": "insertion",
        "pair_id": "smoke_001",
        "within_pair_order": 1,
        "arm": "baseline",
        "rollout_dir": str(target.parent),
        "recommended_filename": target.name,
        "recommended_path": str(target),
        "launch_command": "python for_show_xiaomi/serve_dp_tac_quality_guided.py --disable_guidance",
    }
    payload = {
        "purpose": "Synthetic next step for finalize-and-refresh smoke.",
        "scientific_evidence": False,
        "has_next_step": True,
        "next_row": row,
        "recommended_path": str(target),
        "recommended_path_exists": target.exists(),
        "next_step_pass": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    mkdir_clean(out_root)
    raw = out_root / "raw" / "episode_000.hdf5"
    target = out_root / "formal_rollouts" / "insertion" / "baseline" / "smoke_001__insertion__baseline.hdf5"
    next_step = out_root / "next_step.json"
    make_hdf5(raw, args.steps, args.seed)
    make_next_step(next_step, target)

    run_args = argparse.Namespace(
        next_step=str(next_step),
        source=str(raw),
        source_dir=None,
        output_dir=str(out_root / "runner"),
        tag="synthetic",
        move=False,
        overwrite=False,
        allow_bad_schema=False,
        dry_run=False,
        skip_refresh=True,
        min_steps=3,
        success=None,
        stopped_early=None,
        scorer_freeze_manifest="/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest/tac_quality_scorer_freeze_manifest.json",
    )
    report = build_finalize_and_refresh(run_args)
    json_path = out_root / "runner" / "synthetic" / "tac_quality_finalize_and_refresh.json"
    md_path = out_root / "runner" / "synthetic" / "tac_quality_finalize_and_refresh.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    checks = {
        "wrapper_pass": report.get("finalize_and_refresh_pass") is True,
        "not_scientific_evidence": report.get("scientific_evidence") is False,
        "target_exists": target.exists(),
        "source_kept_by_copy": raw.exists(),
        "target_schema_ok": report.get("finalize", {}).get("finalize_pass") is True,
        "freeze_attrs_written": bool(
            (report.get("finalize", {}).get("scorer_freeze_attrs") or {}).get("written", {}).get(
                "tac_quality_scorer_freeze_manifest_sha256"
            )
        ),
        "skip_refresh_used": report.get("skip_refresh") is True,
    }
    summary = {
        "purpose": "Synthetic smoke for TacQuality finalize-and-refresh wrapper.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "raw": str(raw),
        "target": str(target),
        "next_step": str(next_step),
        "runner_json": str(json_path),
        "runner_markdown": str(md_path),
        "runner_summary": {
            "finalize_and_refresh_pass": report.get("finalize_and_refresh_pass"),
            "finalize": report.get("finalize"),
            "next_required_step": report.get("next_required_step"),
        },
        "note": "Smoke uses isolated synthetic HDF5 and --skip_refresh; it does not update formal real-rollout evidence.",
    }
    out_json = out_root / "tac_quality_finalize_and_refresh_smoke.json"
    out_md = out_root / "tac_quality_finalize_and_refresh_smoke.md"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_md)
    print(
        json.dumps(
            {
                "overall_pass": summary["overall_pass"],
                "wrapper_pass": checks["wrapper_pass"],
                "target_exists": checks["target_exists"],
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
        "# TacQuality Finalize And Refresh Smoke",
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
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=45)
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
