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
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import (  # noqa: E402
    DistilledTacQualityEnergyRuntime,
    TASK_TO_ID as DISTILLED_TASK_TO_ID,
)
from TFAC_V5.tac_quality_dp_guidance_controller import from_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_dp_integration_adapter import (  # noqa: E402
    ActionNormalizer,
    TacQualityDPIntegrationAdapter,
    summarize_tensor,
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


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class DistilledDeploymentAdapter:
    """Minimal adapter for running distilled scorer through the same bridge."""

    def __init__(self, task: str, scorer: DistilledTacQualityEnergyRuntime, action_normalizer: ActionNormalizer):
        self.task = task
        self.scorer = scorer
        self.action_normalizer = action_normalizer
        self.controller = from_guidance_profile(task, clamp_norm_action=False)

    def score_from_prediction(self, tactile: Dict[str, torch.Tensor], action_raw: torch.Tensor) -> torch.Tensor:
        task_id = torch.full(
            (action_raw.shape[0],),
            DISTILLED_TASK_TO_ID[self.task],
            dtype=torch.long,
            device=self.scorer.device,
        )
        return self.scorer.score(
            tactile["left_marker_seq"],
            right_marker_seq=tactile.get("right_marker_seq"),
            eef_action_seq=tactile.get("eef_action_seq"),
            joint_action_seq=action_raw,
            task_id=task_id,
            mode="energy_clipped",
        )

    def guide_final_action(self, action_norm: torch.Tensor, foresight_predict_fn) -> Tuple[torch.Tensor, Dict[str, Any]]:
        action_raw = self.action_normalizer.denormalize(action_norm).detach()

        def score_fn(candidate_raw: torch.Tensor) -> torch.Tensor:
            return self.score_from_prediction(foresight_predict_fn(candidate_raw), candidate_raw)

        guided_raw, report = self.controller.guide(action_raw, score_fn)
        guided_norm = self.action_normalizer.normalize(guided_raw)
        base_score = score_fn(action_raw.detach().clone().requires_grad_(True)).detach()
        guided_score = score_fn(guided_raw.detach().clone().requires_grad_(True)).detach()
        report.update(
            {
                "task": self.task,
                "score_mode": "energy_clipped",
                "adapter_policy": "final_clean_action_trust_region_refinement",
                "scorer_runtime": "DistilledTacQualityEnergyRuntime",
                "base_score_recomputed": summarize_tensor(base_score),
                "guided_score_recomputed": summarize_tensor(guided_score),
                "raw_action_delta": summarize_tensor((guided_raw - action_raw).flatten(1).norm(dim=1)),
                "normalized_action_delta": summarize_tensor((guided_norm - action_norm).flatten(1).norm(dim=1)),
                "integration_contract": {
                    "guidance_location": "after DP denoising has produced clean/final action",
                    "reranking": False,
                    "every_step_ddpm_guidance": False,
                    "recompute_foresight_each_guidance_call": True,
                },
            }
        )
        return guided_norm.detach(), report


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


def make_adapter(task: str, arm: Mapping[str, Any], runtime: TacQualityGuidanceRuntime, normalizer: ActionNormalizer, device: str):
    scorer_name = arm["scorer_runtime"]
    if scorer_name in {"InsertionRiskScorerRuntime", "PTGProxyScorerV2Runtime"}:
        return TacQualityDPIntegrationAdapter(task, runtime=runtime, action_normalizer=normalizer)
    if scorer_name == "DistilledTacQualityEnergyRuntime":
        return DistilledDeploymentAdapter(
            task,
            DistilledTacQualityEnergyRuntime(arm["checkpoint"]["path"], device=device),
            normalizer,
        )
    raise KeyError(scorer_name)


def check_arm(
    task: str,
    arm_name: str,
    arm: Mapping[str, Any],
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
    adapter = make_adapter(task, arm, runtime, normalizer, str(device))
    _, adapter_report = adapter.guide_final_action(action_norm, bridge)
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
            arms.append(check_arm(task, arm_name, arm, runtime, args))
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
