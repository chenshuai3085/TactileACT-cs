"""Smoke-test deployment arms through Foresight bridge and DP adapter.

This is stronger than a scorer-only gradient smoke and weaker than a real
rollout gate.  It verifies that each guided rollout arm can be wired as:

  final clean action -> ForesightTacQualityBridge -> scorer -> d score/d action

The output is an engineering artifact for server integration.  It is not
scientific evidence that the policy improves on the robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.tac_quality_dp_integration_adapter import (  # noqa: E402
    ActionNormalizer,
    summarize_tensor,
)
from TFAC_V5.tac_quality_serving_guidance import (  # noqa: E402
    build_serving_guidance_from_arm,
    git_commit,
)
from TFAC_V5.tac_quality_foresight_bridge import (  # noqa: E402
    ForesightBridgeConfig,
    ForesightTacQualityBridge,
    SyntheticLatentForesight,
)
from TFAC_V5.tac_quality_guidance_runtime import TacQualityGuidanceRuntime  # noqa: E402


DEFAULT_CONFIG = Path("/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_deployment_bridge_smoke")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_norm_stats(action_dim: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "action_min": torch.linspace(-0.8, -0.2, action_dim, device=device),
        "action_max": torch.linspace(0.2, 0.8, action_dim, device=device),
        "action_mean": torch.zeros(action_dim, device=device),
        "action_std": torch.ones(action_dim, device=device),
        "qpos_mean": torch.zeros(action_dim, device=device),
        "qpos_std": torch.ones(action_dim, device=device),
    }


def make_bridge(task: str, action_dim: int, horizon: int, window: int, device: torch.device) -> ForesightTacQualityBridge:
    fs_norm = make_norm_stats(action_dim, device)
    foresight = SyntheticLatentForesight(action_dim=action_dim, pred_steps=4).to(device)
    qpos = torch.zeros(1, action_dim, device=device)
    images = [torch.zeros(1, 3, 64, 64, device=device), torch.zeros(1, 3, 64, 64, device=device)]
    marker_window = torch.zeros(1, window, 9, 9, 2, device=device)
    return ForesightTacQualityBridge(
        foresight,
        fs_norm,
        qpos_raw=qpos,
        foresight_images=images,
        marker_window_norm=marker_window,
        config=ForesightBridgeConfig(task=task, window=window, action_chunk=horizon),
    )


def bridge_grad_check(bridge: ForesightTacQualityBridge, action_raw: torch.Tensor) -> Dict[str, Any]:
    x = action_raw.detach().clone().requires_grad_(True)
    tactile = bridge(x)
    scalar = tactile["left_marker_seq"].square().mean() + tactile["eef_action_seq"].square().mean()
    if "right_marker_seq" in tactile:
        scalar = scalar + tactile["right_marker_seq"].square().mean()
    grad = torch.autograd.grad(scalar, x, retain_graph=False)[0]
    return {
        "tactile_shapes": {key: list(value.shape) for key, value in tactile.items()},
        "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
        "positive_grad_rate": float((grad.flatten(1).norm(dim=1) > 1e-8).float().mean().cpu()),
        "grad_norm": summarize_tensor(grad.flatten(1).norm(dim=1)),
    }


def check_arm(
    task: str,
    arm_name: str,
    cfg: Dict[str, Any],
    runtime: TacQualityGuidanceRuntime,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    device = runtime.device
    action_dim = args.action_dim
    norm_stats = make_norm_stats(action_dim, device)
    normalizer = ActionNormalizer.from_norm_stats(norm_stats, mode="minmax")
    action_norm = torch.empty(args.batch_size, args.horizon, action_dim, device=device).uniform_(-0.25, 0.25)
    action_raw = normalizer.denormalize(action_norm)
    bridge = make_bridge(task, action_dim, args.horizon, args.window, device)
    bridge_report = bridge_grad_check(bridge, action_raw)
    serving = build_serving_guidance_from_arm(
        task,
        arm_name,
        dp_norm_stats=norm_stats,
        rollout_config=cfg,
        device=str(device),
        norm_mode="minmax",
        runtime=runtime,
    )
    with torch.inference_mode():
        _, adapter_report = serving.guide_action_chunk(action_norm, bridge)
    arm = cfg["tasks"][task][arm_name]
    return {
        "task": task,
        "arm": arm_name,
        "scorer_runtime": arm["scorer_runtime"],
        "checkpoint": arm["checkpoint"]["path"],
        "bridge": bridge_report,
        "adapter": adapter_report,
        "passes_deployment_bridge_smoke": bool(
            bridge_report["finite_grad_rate"] >= 0.999
            and bridge_report["positive_grad_rate"] >= 0.999
            and adapter_report["finite_grad_rate"] >= 0.999
            and adapter_report["positive_grad_rate"] >= 0.999
            and adapter_report["improved_rate"] >= args.min_improved_rate
            and adapter_report["max_delta_within_trust_region"]
            and adapter_report["integration_contract"]["reranking"] is False
            and adapter_report["integration_contract"]["every_step_ddpm_guidance"] is False
            and adapter_report["called_from_inference_mode"] is True
            and adapter_report["returned_requires_grad"] is False
        ),
    }


def build(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    cfg = load_json(Path(args.config))
    runtime = TacQualityGuidanceRuntime(device=args.device)
    arms = []
    for task, task_arms in cfg["tasks"].items():
        for arm_name, arm in task_arms.items():
            if not arm.get("guidance_enabled", False):
                arms.append(
                    {
                        "task": task,
                        "arm": arm_name,
                        "guidance_enabled": False,
                        "passes_deployment_bridge_smoke": True,
                        "skip_reason": "baseline arm has no scorer and no TacQuality guidance",
                    }
                )
                continue
            arms.append(check_arm(task, arm_name, cfg, runtime, args))
    guided = [row for row in arms if row.get("guidance_enabled", True)]
    result = {
        "purpose": "Verify guided rollout arms can run final-action TacQuality guidance through Foresight bridge.",
        "scientific_evidence": False,
        "config": str(args.config),
        "git_commit": git_commit(),
        "seed": args.seed,
        "guidance_mode": "final_clean_action_trust_region_refinement",
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "arms": arms,
        "checks": {
            "all_guided_arms_present": len(guided) == 4,
            "all_guided_arms_pass_deployment_bridge_smoke": all(
                row["passes_deployment_bridge_smoke"] for row in guided
            ),
            "baseline_arms_recorded": sum(1 for row in arms if not row.get("guidance_enabled", True)) == 2,
        },
    }
    result["overall_pass"] = all(result["checks"].values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Deployment Bridge Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- guidance_mode: `{result['guidance_mode']}`",
        f"- git_commit: `{result['git_commit']}`",
        "",
        "## Arms",
        "",
        "| task | arm | scorer | pass | improved_rate | bridge_grad |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in result["arms"]:
        lines.append(
            "| {task} | {arm} | {scorer} | {passed} | {improved} | {grad} |".format(
                task=row["task"],
                arm=row["arm"],
                scorer=row.get("scorer_runtime"),
                passed=row["passes_deployment_bridge_smoke"],
                improved=row.get("adapter", {}).get("improved_rate", ""),
                grad=row.get("bridge", {}).get("positive_grad_rate", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This smoke verifies deployment wiring only: rollout arm config -> scorer runtime -> Foresight bridge -> DP adapter.",
            "It remains synthetic and does not replace formal baseline-vs-guided HDF5/robot rollout gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--min_improved_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=61)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_deployment_bridge_smoke.json"
    md_path = out_dir / "tac_quality_deployment_bridge_smoke.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
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
