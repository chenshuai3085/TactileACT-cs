"""Audit whether formal TacQuality rollout gate artifacts are real evidence.

This checks the four artifacts that would close the remaining TacQuality goal:

  - insertion baseline-vs-guided quality gate
  - board baseline-vs-guided quality gate
  - insertion baseline/default/distilled scorer ablation
  - board baseline/default/distilled scorer ablation

It does not evaluate rollouts.  It only classifies the source status as
missing, real_candidate, synthetic_or_smoke, or incomplete_candidate so smoke
artifacts cannot accidentally close the real-rollout validation gap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_real_rollout_source_audit")

PATHS = {
    "insertion_two_arm": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "insertion_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
    "board_two_arm": Path(
        "/home/chenshuai/Project/output/real_rollout_quality_gate/"
        "board_baseline_vs_guided/real_rollout_quality_gate.json"
    ),
    "insertion_three_arm": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "insertion_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
    "board_three_arm": Path(
        "/home/chenshuai/Project/output/real_rollout_scorer_ablation_gate/"
        "board_baseline_vs_default_vs_distilled/real_rollout_scorer_ablation_gate.json"
    ),
}

SYNTHETIC_TOKENS = ("synthetic", "smoke")


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get(d: Optional[Dict[str, Any]], dotted: str, default=None):
    cur: Any = d
    if cur is None:
        return default
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def source_paths(data: Dict[str, Any], kind: str) -> List[str]:
    if kind == "two_arm":
        keys = ["baseline_dir", "guided_dir", "pairing_csv", "metadata_csv"]
    else:
        keys = [
            "baseline_dir",
            "default_guided_dir",
            "distilled_guided_dir",
            "pairing_csv",
            "metadata_csv",
        ]
    return [str(data.get(key, "")) for key in keys if data.get(key)]


def has_synthetic_token(paths: List[str]) -> bool:
    text = " ".join(paths).lower()
    return any(token in text for token in SYNTHETIC_TOKENS)


def classify_artifact(name: str, path: Path) -> Dict[str, Any]:
    kind = "two_arm" if name.endswith("two_arm") else "three_arm"
    data = load_json(path)
    if data is None:
        return {
            "name": name,
            "kind": kind,
            "path": str(path),
            "exists": False,
            "source_status": "missing",
            "can_count_as_real_evidence": False,
            "reason": "artifact missing",
            "source_paths": [],
        }
    paths = source_paths(data, kind)
    if has_synthetic_token(paths):
        return {
            "name": name,
            "kind": kind,
            "path": str(path),
            "exists": True,
            "source_status": "synthetic_or_smoke",
            "can_count_as_real_evidence": False,
            "reason": "source paths contain synthetic/smoke tokens",
            "source_paths": paths,
        }
    if kind == "two_arm":
        passed = get(data, "decision.production_validation_pass") is True
        debug = get(data, "debug_or_underpowered", True) is True
    else:
        passed = get(data, "production_ablation_pass") is True
        debug = get(data, "debug_or_underpowered", True) is True
    if passed and not debug:
        status = "real_candidate"
        reason = "artifact passes formal gate and source paths are not synthetic/smoke"
        can_count = True
    else:
        status = "incomplete_candidate"
        reason = "artifact exists but does not pass formal gate or is debug/underpowered"
        can_count = False
    return {
        "name": name,
        "kind": kind,
        "path": str(path),
        "exists": True,
        "source_status": status,
        "can_count_as_real_evidence": bool(can_count),
        "reason": reason,
        "source_paths": paths,
        "passed": bool(passed),
        "debug_or_underpowered": bool(debug),
    }


def build_audit() -> Dict[str, Any]:
    artifacts = {name: classify_artifact(name, path) for name, path in PATHS.items()}
    blockers = [
        row
        for row in artifacts.values()
        if not row["can_count_as_real_evidence"]
    ]
    return {
        "purpose": "Guardrail audit for real-vs-synthetic TacQuality rollout gate evidence.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "artifacts": artifacts,
        "all_four_real_evidence_present": len(blockers) == 0,
        "n_real_evidence": sum(1 for row in artifacts.values() if row["can_count_as_real_evidence"]),
        "n_blockers": len(blockers),
        "blockers": blockers,
        "synthetic_guardrail_pass": all(
            row["source_status"] != "synthetic_or_smoke" or not row["can_count_as_real_evidence"]
            for row in artifacts.values()
        ),
        "next_required_step": (
            "Collect real/production HDF5 rollouts and run formal gates into the standard real_rollout_* output dirs."
            if blockers
            else None
        ),
    }


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Real Rollout Source Audit",
        "",
        f"- all_four_real_evidence_present: `{result['all_four_real_evidence_present']}`",
        f"- n_real_evidence: `{result['n_real_evidence']}`",
        f"- n_blockers: `{result['n_blockers']}`",
        f"- synthetic_guardrail_pass: `{result['synthetic_guardrail_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        "",
        "| artifact | status | can count | reason |",
        "|---|---|---:|---|",
    ]
    for row in result["artifacts"].values():
        lines.append(
            f"| {row['name']} | {row['source_status']} | "
            f"{row['can_count_as_real_evidence']} | {row['reason']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_audit()
    json_path = out_dir / "tac_quality_real_rollout_source_audit.json"
    md_path = out_dir / "tac_quality_real_rollout_source_audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "all_four_real_evidence_present": result["all_four_real_evidence_present"],
                "n_real_evidence": result["n_real_evidence"],
                "n_blockers": result["n_blockers"],
                "synthetic_guardrail_pass": result["synthetic_guardrail_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
