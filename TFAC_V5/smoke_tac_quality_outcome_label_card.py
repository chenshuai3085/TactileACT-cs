"""Synthetic smoke for TacQuality outcome-label card generation."""

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

from TFAC_V5.build_tac_quality_outcome_label_card import build  # noqa: E402


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_outcome_label_card_smoke")


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
    card = build(argparse.Namespace(output_dir=str(out_root)))
    text = json.dumps(card, ensure_ascii=False)
    checks = {
        "card_pass": card.get("outcome_label_card_pass") is True,
        "not_scientific": card.get("scientific_evidence") is False,
        "both_tasks_present": set(card.get("tasks", {})) == {"insertion", "board"},
        "insertion_bounce_bad_defined": "bounce" in text.lower(),
        "board_force_and_smoothness_defined": "force" in text.lower() and "smooth" in text.lower(),
        "manual_no_auto_inference_guardrail": "Do not infer success/stopped_early from scorer outputs."
        in card.get("guardrails", []),
        "finalize_command_records_outcome": "--success <true_or_false>" in card.get("usage", {}).get("finalize_with_attrs", ""),
    }
    summary = {
        "purpose": "Synthetic smoke for TacQuality outcome-label card.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "overall_pass": bool(all(checks.values())),
        "checks": checks,
        "card_summary": {
            "outcome_label_card_pass": card.get("outcome_label_card_pass"),
            "tasks": list(card.get("tasks", {}).keys()),
            "usage": card.get("usage"),
        },
    }
    json_path = out_root / "tac_quality_outcome_label_card_smoke.json"
    md_path = out_root / "tac_quality_outcome_label_card_smoke.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TacQuality Outcome Label Card Smoke",
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
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="synthetic")
    return parser.parse_args()


if __name__ == "__main__":
    smoke(parse_args())
