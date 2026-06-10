"""Smoke-test TacQuality scorer-freeze manifest generation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.build_tac_quality_scorer_freeze_manifest import build  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest_smoke")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_dir) / args.tag
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = build(
        argparse.Namespace(
            arm_configs=args.arm_configs,
            output_dir=str(out_root / "freeze"),
            exclude_optional=False,
        )
    )
    arms = manifest.get("arms", {})
    checks = {
        "freeze_manifest_pass": manifest.get("scorer_freeze_manifest_pass") is True,
        "not_scientific": manifest.get("scientific_evidence") is False,
        "has_six_formal_arms": len([row for row in arms.values() if row.get("formal_arm")]) == 6,
        "has_four_formal_guided_arms": len(
            [row for row in arms.values() if row.get("formal_arm") and row.get("guidance_enabled")]
        )
        == 4,
        "has_two_optional_action_aware_arms": len([row for row in arms.values() if row.get("optional_arm")]) == 2,
        "guided_checkpoints_have_sha256": all(
            (row.get("checkpoint") or {}).get("sha256")
            for row in arms.values()
            if row.get("guidance_enabled")
        ),
        "runtime_modules_have_sha256": all(
            row.get("sha256") for row in (manifest.get("runtime_modules") or {}).values()
        ),
        "baseline_no_scorer": manifest.get("checks", {}).get("baseline_has_no_scorer") is True,
        "default_scorers_match_selection": manifest.get("checks", {}).get("formal_default_scorers_match_selection")
        is True,
        "distilled_ablation_present": manifest.get("checks", {}).get("formal_distilled_ablation_present") is True,
    }
    summary = {
        "purpose": "Synthetic smoke for TacQuality scorer-freeze manifest.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "manifest_summary": {
            "scorer_freeze_manifest_pass": manifest.get("scorer_freeze_manifest_pass"),
            "n_arms": len(arms),
            "n_runtime_modules": len(manifest.get("runtime_modules") or {}),
            "guardrails": manifest.get("guardrails"),
        },
    }
    json_path = out_root / "tac_quality_scorer_freeze_manifest_smoke.json"
    md_path = out_root / "tac_quality_scorer_freeze_manifest_smoke.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TacQuality Scorer Freeze Manifest Smoke",
        "",
        f"- overall_pass: `{summary['overall_pass']}`",
        f"- scientific_evidence: `{summary['scientific_evidence']}`",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in checks.items():
        lines.append(f"| {name} | {passed} |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm_configs",
        default="/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json",
    )
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic")
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
