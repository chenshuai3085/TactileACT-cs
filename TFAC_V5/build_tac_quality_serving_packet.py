"""Build a serving integration packet for TacQuality DP guidance.

This packet is the bridge between the offline scorer package and a real DP
server launch.  It preflights the files and normalization contracts needed to
instantiate:

  DP clean action -> TacQualityServingGuidance -> ForesightTacQualityBridge

The script can run without checkpoint arguments to produce a template packet.
When DP/Foresight paths are provided, it performs stricter checks.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_serving_packet")
DEFAULT_ROLLOUT_ARM_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json"
)
DEFAULT_DEPLOYMENT_SMOKE = Path(
    "/home/chenshuai/Project/output/tac_quality_deployment_bridge_smoke/"
    "tac_quality_deployment_bridge_smoke.json"
)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def file_info(path_value: Optional[str]) -> Dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False, "bytes": None}
    path = Path(path_value)
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
    }


def find_dp_config(dp_ckpt_dir: Optional[str]) -> Dict[str, Any]:
    if not dp_ckpt_dir:
        return {"provided": False, "ready": False, "reason": "dp_ckpt_dir not provided"}
    config_path = Path(dp_ckpt_dir) / "config.json"
    if not config_path.exists():
        return {"provided": True, "ready": False, "path": str(config_path), "reason": "config.json missing"}
    cfg = load_json(config_path)
    ns = cfg.get("norm_stats", {})
    required = ["action_min", "action_max", "qpos_min", "qpos_max"]
    missing = [key for key in required if key not in ns]
    action_dim = cfg.get("action_dim")
    pred_horizon = cfg.get("pred_horizon")
    checks = {
        "has_minmax_action_stats": "action_min" in ns and "action_max" in ns,
        "has_qpos_minmax_stats": "qpos_min" in ns and "qpos_max" in ns,
        "action_dim_matches_stats": bool(
            action_dim is not None
            and "action_min" in ns
            and len(ns["action_min"]) == int(action_dim)
            and len(ns["action_max"]) == int(action_dim)
        ),
        "pred_horizon_positive": bool(pred_horizon is not None and int(pred_horizon) > 0),
    }
    return {
        "provided": True,
        "ready": not missing and all(checks.values()),
        "path": str(config_path),
        "variant": cfg.get("variant"),
        "action_dim": action_dim,
        "pred_horizon": pred_horizon,
        "obs_horizon": cfg.get("obs_horizon"),
        "camera_names": cfg.get("camera_names"),
        "norm_stats_required": required,
        "missing": missing,
        "checks": checks,
    }


def find_foresight_stats(foresight_dir: Optional[str]) -> Dict[str, Any]:
    if not foresight_dir:
        return {"provided": False, "ready": False, "reason": "foresight_dir not provided"}
    root = Path(foresight_dir)
    stats_path = root / "dataset_stats.pkl"
    args_path = root / "args.json"
    if stats_path.exists():
        stats = load_pickle(stats_path)
        source = str(stats_path)
    elif args_path.exists():
        stats = load_json(args_path)
        stats = stats.get("norm_stats", stats)
        source = str(args_path)
    else:
        return {
            "provided": True,
            "ready": False,
            "path": str(root),
            "reason": "dataset_stats.pkl or args.json missing",
        }
    required = ["action_mean", "action_std", "qpos_mean", "qpos_std"]
    missing = [key for key in required if key not in stats]
    return {
        "provided": True,
        "ready": not missing,
        "path": str(root),
        "source": source,
        "required": required,
        "missing": missing,
        "action_dim": len(stats["action_mean"]) if "action_mean" in stats else None,
        "qpos_dim": len(stats["qpos_mean"]) if "qpos_mean" in stats else None,
    }


def arm_summary(rollout_config: Mapping[str, Any]) -> Dict[str, Any]:
    tasks: Dict[str, Any] = {}
    for task, arms in rollout_config["tasks"].items():
        tasks[task] = {}
        for arm_name, arm in arms.items():
            tasks[task][arm_name] = {
                "guidance_enabled": arm.get("guidance_enabled", False),
                "scorer_runtime": arm.get("scorer_runtime"),
                "checkpoint": arm.get("checkpoint"),
                "policy": arm.get("policy"),
            }
    return tasks


def serving_snippet() -> str:
    return "\n".join(
        [
            "from TFAC_V5.tac_quality_serving_guidance import build_serving_guidance_from_arm, load_rollout_arm_config",
            "from TFAC_V5.tac_quality_foresight_bridge import ForesightTacQualityBridge, ForesightBridgeConfig",
            "",
            "rollout_config = load_rollout_arm_config()",
            "helper = build_serving_guidance_from_arm(",
            "    task='<insertion_or_board>',",
            "    arm_name='<default_guided_or_distilled_guided>',",
            "    dp_norm_stats=config['norm_stats'],",
            "    rollout_config=rollout_config,",
            "    device=str(device),",
            ")",
            "bridge = ForesightTacQualityBridge(",
            "    foresight, fs_norm, qpos_raw=qpos_raw,",
            "    foresight_images=foresight_images,",
            "    marker_window_norm=marker_window_norm,",
            "    config=ForesightBridgeConfig(task='<insertion_or_board>', window=score_window, action_chunk=foresight_chunk),",
            ")",
            "with torch.inference_mode():",
            "    action_norm = ddpm_inference(...)",
            "    guided_action_norm, report = helper.guide_action_chunk(action_norm, bridge)",
            "    action_raw = (guided_action_norm + 1) / 2 * (action_max - action_min) + action_min",
        ]
    )


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rollout = load_json(Path(args.rollout_arm_config))
    deployment_smoke = load_json(Path(args.deployment_smoke)) if Path(args.deployment_smoke).exists() else None
    dp = find_dp_config(args.dp_ckpt_dir)
    foresight = find_foresight_stats(args.foresight_dir)
    ckpt = file_info(args.foresight_ckpt)
    dims_match = (
        dp.get("ready")
        and foresight.get("ready")
        and dp.get("action_dim") == foresight.get("action_dim")
    )
    strict_inputs_provided = bool(args.dp_ckpt_dir and args.foresight_dir and args.foresight_ckpt)
    checks = {
        "rollout_arm_config_exists": Path(args.rollout_arm_config).exists(),
        "deployment_bridge_smoke_pass": bool(deployment_smoke and deployment_smoke.get("overall_pass")),
        "dp_config_ready": bool(dp.get("ready")),
        "foresight_stats_ready": bool(foresight.get("ready")),
        "foresight_ckpt_exists": bool(ckpt["exists"]),
        "dp_action_dim_matches_foresight_action_dim": bool(dims_match),
        "strict_inputs_provided": strict_inputs_provided,
    }
    serving_ready = all(checks.values())
    result = {
        "purpose": "Serving integration packet for TacQuality final-action DP classifier guidance.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "serving_ready": bool(serving_ready),
        "template_only": not strict_inputs_provided,
        "recommended_mode": "final_clean_action_trust_region_refinement",
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "inputs": {
            "dp_ckpt_dir": args.dp_ckpt_dir,
            "foresight_dir": args.foresight_dir,
            "foresight_ckpt": args.foresight_ckpt,
            "rollout_arm_config": args.rollout_arm_config,
            "deployment_smoke": args.deployment_smoke,
        },
        "dp": dp,
        "foresight": foresight,
        "foresight_ckpt": ckpt,
        "rollout_arms": arm_summary(rollout),
        "checks": checks,
        "serving_code_snippet": serving_snippet(),
        "next_step": (
            "Launch a TacQuality-guided DP server with the checked DP/Foresight files."
            if serving_ready
            else "Provide --dp_ckpt_dir, --foresight_dir, and --foresight_ckpt to turn this template into a strict serving preflight."
        ),
    }
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Serving Integration Packet",
        "",
        f"- serving_ready: `{result['serving_ready']}`",
        f"- template_only: `{result['template_only']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- recommended_mode: `{result['recommended_mode']}`",
        f"- next_step: {result['next_step']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in result["checks"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Serving Snippet",
            "",
            "```python",
            result["serving_code_snippet"],
            "```",
            "",
            "## Interpretation",
            "",
            "This packet verifies serving wiring readiness. It is not a rollout result and does not replace the formal HDF5/robot gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp_ckpt_dir", default=None)
    parser.add_argument("--foresight_dir", default=None)
    parser.add_argument("--foresight_ckpt", default=None)
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_ARM_CONFIG))
    parser.add_argument("--deployment_smoke", default=str(DEFAULT_DEPLOYMENT_SMOKE))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="template_preflight")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_serving_packet.json"
    md_path = out_dir / "tac_quality_serving_packet.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "serving_ready": result["serving_ready"],
                "template_only": result["template_only"],
                "scientific_evidence": result["scientific_evidence"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
