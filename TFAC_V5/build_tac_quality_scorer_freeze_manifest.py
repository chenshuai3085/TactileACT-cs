"""Build a formal scorer-freeze manifest for TacQuality rollout collection.

Real rollout evidence is only interpretable if the scorer/runtime package does
not drift during collection.  This manifest records the exact scorer runtime,
checkpoint path, checkpoint sha256, and relevant module sha256 values used by
the formal rollout arm configs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_scorer_freeze_manifest")
DEFAULT_ARM_CONFIGS = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs.json"
)

MODULES = {
    "rollout_arm_configs_builder": ROOT / "TFAC_V5/build_tac_quality_rollout_arm_configs.py",
    "serving_guidance": ROOT / "TFAC_V5/tac_quality_serving_guidance.py",
    "guidance_runtime": ROOT / "TFAC_V5/tac_quality_guidance_runtime.py",
    "trust_region_refiner": ROOT / "TFAC_V5/tac_quality_trust_region_guidance.py",
    "dp_integration_adapter": ROOT / "TFAC_V5/tac_quality_dp_integration_adapter.py",
    "foresight_bridge": ROOT / "TFAC_V5/tac_quality_foresight_bridge.py",
    "insertion_runtime": ROOT / "TFAC_V5/insertion_risk_scorer_runtime.py",
    "board_runtime": ROOT / "TFAC_V5/ptg_proxy_scorer_v2_runtime.py",
    "distilled_runtime": ROOT / "TFAC_V5/distilled_tac_quality_energy_runtime.py",
    "action_aware_runtime": ROOT / "TFAC_V5/action_aware_scorer_runtime.py",
}

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


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
        "sha256": sha256_file(path),
    }


def iter_arm_rows(configs: Dict[str, Any], include_optional: bool) -> Iterable[tuple[str, str, Dict[str, Any]]]:
    arms = FORMAL_ARMS + (OPTIONAL_ARMS if include_optional else ())
    for task, task_row in (configs.get("tasks") or {}).items():
        for arm in arms:
            row = task_row.get(arm)
            if isinstance(row, dict):
                yield task, arm, row


def build(args: argparse.Namespace) -> Dict[str, Any]:
    configs = load_json(Path(args.arm_configs))
    arm_rows = {}
    include_optional = not args.exclude_optional
    for task, arm, row in iter_arm_rows(configs, include_optional):
        ckpt = row.get("checkpoint") or {}
        ckpt_path = Path(ckpt.get("path", ""))
        arm_rows[f"{task}:{arm}"] = {
            "task": task,
            "arm": arm,
            "formal_arm": arm in FORMAL_ARMS,
            "optional_arm": arm in OPTIONAL_ARMS,
            "guidance_enabled": row.get("guidance_enabled"),
            "scorer_runtime": row.get("scorer_runtime"),
            "policy": row.get("policy"),
            "score_mode": row.get("refiner", {}).get("score_mode"),
            "checkpoint": file_record(ckpt_path) if ckpt_path else None,
            "refiner": row.get("refiner"),
        }
    module_records = {name: file_record(path) for name, path in MODULES.items()}
    formal_guided = [
        row
        for row in arm_rows.values()
        if row["formal_arm"] and row["guidance_enabled"]
    ]
    optional_guided = [
        row
        for row in arm_rows.values()
        if row["optional_arm"] and row["guidance_enabled"]
    ]
    checks = {
        "arm_config_pass": configs.get("rollout_arm_config_pass") is True,
        "all_formal_guided_arms_present": len(formal_guided) == 4,
        "optional_action_aware_present": len(optional_guided) == 2,
        "all_guided_checkpoints_exist": all((row.get("checkpoint") or {}).get("exists") for row in formal_guided + optional_guided),
        "all_guided_checkpoints_hashed": all((row.get("checkpoint") or {}).get("sha256") for row in formal_guided + optional_guided),
        "all_runtime_modules_exist": all(row["exists"] for row in module_records.values()),
        "all_runtime_modules_hashed": all(row["sha256"] for row in module_records.values()),
        "baseline_has_no_scorer": all(
            row.get("scorer_runtime") is None and row.get("guidance_enabled") is False
            for key, row in arm_rows.items()
            if key.endswith(":baseline")
        ),
        "formal_default_scorers_match_selection": (
            arm_rows.get("insertion:default_guided", {}).get("scorer_runtime") == "InsertionRiskScorerRuntime"
            and arm_rows.get("board:default_guided", {}).get("scorer_runtime") == "PTGProxyScorerV2Runtime"
        ),
        "formal_distilled_ablation_present": all(
            arm_rows.get(f"{task}:distilled_guided", {}).get("scorer_runtime")
            == "DistilledTacQualityEnergyRuntime"
            for task in ["insertion", "board"]
        ),
    }
    result = {
        "purpose": "Freeze exact TacQuality scorer/runtime artifacts for formal real rollout collection.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "arm_configs": str(args.arm_configs),
        "include_optional": bool(include_optional),
        "arms": arm_rows,
        "runtime_modules": module_records,
        "checks": checks,
        "scorer_freeze_manifest_pass": bool(all(checks.values())),
        "guardrails": [
            "Do not overwrite scorer checkpoints after this manifest is generated for a formal rollout batch.",
            "If any checkpoint/runtime file changes, regenerate launch sheet, arm configs, this freeze manifest, and all readiness artifacts before collecting more rollouts.",
            "Real rollout gate reports should be interpreted against these sha256 hashes.",
            "This is a provenance/freeze artifact, not policy-quality evidence.",
        ],
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Scorer Freeze Manifest",
        "",
        f"- scorer_freeze_manifest_pass: `{result['scorer_freeze_manifest_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- arm_configs: `{result['arm_configs']}`",
        "",
        "## Arms",
        "",
        "| key | scorer | checkpoint sha256 | policy |",
        "|---|---|---|---|",
    ]
    for key, row in sorted(result["arms"].items()):
        ckpt = row.get("checkpoint") or {}
        lines.append(
            f"| {key} | {row.get('scorer_runtime')} | `{ckpt.get('sha256')}` | {row.get('policy')} |"
        )
    lines.extend(["", "## Runtime Modules", "", "| module | sha256 |", "|---|---|"])
    for name, row in sorted(result["runtime_modules"].items()):
        lines.append(f"| {name} | `{row.get('sha256')}` |")
    lines.extend(["", "## Guardrails", ""])
    for item in result["guardrails"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm_configs", default=str(DEFAULT_ARM_CONFIGS))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--exclude_optional", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_scorer_freeze_manifest.json"
    md_path = out_dir / "tac_quality_scorer_freeze_manifest.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "scorer_freeze_manifest_pass": result["scorer_freeze_manifest_pass"],
                "n_arms": len(result["arms"]),
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
