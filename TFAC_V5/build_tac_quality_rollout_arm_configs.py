"""Build machine-readable rollout-arm configs for TacQuality validation.

The formal real-rollout experiment has three arms per task:

  1. baseline DP: no TacQuality guidance;
  2. task-default guided: insertion risk scorer or board PTGProxyV2;
  3. distilled guided: DistilledTacQualityEnergyRuntime.

This script materializes those choices into JSON/Markdown so data collection
and deployment code do not rely on prose in a README.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_guidance_config import get_guidance_profile


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_arm_configs")

PATHS = {
    "insertion_scorer_ckpt": Path("/home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt"),
    "board_scorer_ckpt": Path("/home/chenshuai/Project/output/ptg_proxy_scorer_v2/ptg_proxy_scorer_v2_final.pt"),
    "distilled_ckpt": Path("/home/chenshuai/Project/output/distilled_tac_quality_energy/distilled_tac_quality_energy_final.pt"),
    "selection_gate": Path("/home/chenshuai/Project/output/tac_quality_scorer_selection_gate/tac_quality_scorer_selection_gate.json"),
    "board_calibration": Path("/home/chenshuai/Project/output/board_target_force_calibration/board_target_force_calibration.json"),
    "experiment_packet": Path("/home/chenshuai/Project/output/real_rollout_experiment_packet/formal_paired12/real_rollout_experiment_packet.json"),
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_info(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "bytes": int(path.stat().st_size) if exists and path.is_file() else None,
    }


def refiner_config(profile_name: str) -> Dict[str, Any]:
    profile = get_guidance_profile(profile_name)
    return {
        "profile": profile.name,
        "task_id": profile.task_id,
        "score_mode": "profile",
        "refinement": profile.refinement.__dict__,
        "energy": profile.energy.__dict__,
        "evidence": profile.evidence,
        "scope": profile.scope,
    }


def distilled_refiner_config(task: str) -> Dict[str, Any]:
    base = refiner_config(task)
    base["score_mode"] = "energy_clipped"
    base["energy"] = {
        "source": "DistilledTacQualityEnergyRuntime.energy_clipped",
        "teacher": "RF teacher distilled from ptg_proxy_scorer_v2 binary/quality evidence",
    }
    base["scope"] = "Cross-task ablation candidate; not current default replacement."
    return base


def build(args: argparse.Namespace) -> Dict[str, Any]:
    board_calibration = load_json(PATHS["board_calibration"])
    packet = load_json(PATHS["experiment_packet"])
    selection = load_json(PATHS["selection_gate"])
    board_force_args = {
        "board_target_force": board_calibration["recommended"]["board_target_force"],
        "board_force_sigma": board_calibration["recommended"]["board_force_sigma"],
        "calibration_json": str(PATHS["board_calibration"]),
    }
    tasks = {
        "insertion": {
            "baseline": {
                "arm": "baseline",
                "policy": "original_dp",
                "guidance_enabled": False,
                "scorer_runtime": None,
                "checkpoint": None,
                "expected_rollout_dir_placeholder": "<insertion_baseline_rollout_dir>",
            },
            "default_guided": {
                "arm": "default_guided",
                "policy": "dp_with_final_clean_action_trust_region_refinement",
                "guidance_enabled": True,
                "scorer_runtime": "InsertionRiskScorerRuntime",
                "checkpoint": file_info(PATHS["insertion_scorer_ckpt"]),
                "refiner": refiner_config("insertion"),
                "expected_rollout_dir_placeholder": "<insertion_default_guided_rollout_dir>",
            },
            "distilled_guided": {
                "arm": "distilled_guided",
                "policy": "dp_with_final_clean_action_trust_region_refinement",
                "guidance_enabled": True,
                "scorer_runtime": "DistilledTacQualityEnergyRuntime",
                "checkpoint": file_info(PATHS["distilled_ckpt"]),
                "refiner": distilled_refiner_config("insertion"),
                "expected_rollout_dir_placeholder": "<insertion_distilled_guided_rollout_dir>",
            },
        },
        "board": {
            "baseline": {
                "arm": "baseline",
                "policy": "original_dp",
                "guidance_enabled": False,
                "scorer_runtime": None,
                "checkpoint": None,
                "expected_rollout_dir_placeholder": "<board_baseline_rollout_dir>",
            },
            "default_guided": {
                "arm": "default_guided",
                "policy": "dp_with_final_clean_action_trust_region_refinement",
                "guidance_enabled": True,
                "scorer_runtime": "PTGProxyScorerV2Runtime",
                "checkpoint": file_info(PATHS["board_scorer_ckpt"]),
                "refiner": refiner_config("board"),
                "board_force_gate": board_force_args,
                "expected_rollout_dir_placeholder": "<board_default_guided_rollout_dir>",
            },
            "distilled_guided": {
                "arm": "distilled_guided",
                "policy": "dp_with_final_clean_action_trust_region_refinement",
                "guidance_enabled": True,
                "scorer_runtime": "DistilledTacQualityEnergyRuntime",
                "checkpoint": file_info(PATHS["distilled_ckpt"]),
                "refiner": distilled_refiner_config("board"),
                "board_force_gate": board_force_args,
                "expected_rollout_dir_placeholder": "<board_distilled_guided_rollout_dir>",
            },
        },
    }
    result = {
        "purpose": "Machine-readable rollout arm configs for TacQuality scorer validation.",
        "git_commit": git_commit(),
        "recommended_guidance_mode": "final_clean_action_trust_region_refinement",
        "tasks": tasks,
        "selection_summary": {
            "current_default_insertion_scorer": selection["selection"]["current_default_insertion_scorer"],
            "current_default_board_scorer": selection["selection"]["current_default_board_scorer"],
            "promoted_ablation_candidate": selection["selection"]["promoted_ablation_candidate"],
            "distilled_replacement_status": selection["selection"]["distilled_replacement_status"],
        },
        "formal_packet": {
            "json": str(PATHS["experiment_packet"]),
            "insertion_ablation_gate_command": packet["tasks"]["insertion"]["ablation_gate_command"],
            "board_ablation_gate_command": packet["tasks"]["board"]["ablation_gate_command"],
        },
        "checks": {
            "all_checkpoints_exist": all(
                PATHS[key].exists()
                for key in ["insertion_scorer_ckpt", "board_scorer_ckpt", "distilled_ckpt"]
            ),
            "board_calibration_exists": PATHS["board_calibration"].exists(),
            "experiment_packet_exists": PATHS["experiment_packet"].exists(),
        },
    }
    result["rollout_arm_config_pass"] = all(result["checks"].values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Rollout Arm Configs",
        "",
        f"- rollout_arm_config_pass: `{result['rollout_arm_config_pass']}`",
        f"- recommended_guidance_mode: `{result['recommended_guidance_mode']}`",
        "",
        "## Arms",
        "",
        "| task | arm | scorer | guidance | checkpoint |",
        "|---|---|---|---:|---|",
    ]
    for task, arms in result["tasks"].items():
        for arm, cfg in arms.items():
            ckpt = cfg.get("checkpoint")
            ckpt_path = ckpt["path"] if isinstance(ckpt, dict) else ""
            lines.append(
                f"| {task} | {arm} | {cfg.get('scorer_runtime')} | {cfg['guidance_enabled']} | {ckpt_path} |"
            )
    lines.extend(
        [
            "",
            "## Formal Commands",
            "",
            "```bash",
            result["formal_packet"]["insertion_ablation_gate_command"],
            result["formal_packet"]["board_ablation_gate_command"],
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_rollout_arm_configs.json"
    md_path = out_dir / "tac_quality_rollout_arm_configs.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "rollout_arm_config_pass": result["rollout_arm_config_pass"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
