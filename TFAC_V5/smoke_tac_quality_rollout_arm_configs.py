"""Smoke-test TacQuality rollout arm configs for gradient-guidance readiness.

This is an engineering smoke test, not task-effectiveness evidence.  It verifies
that every guided arm in tac_quality_rollout_arm_configs.json can:

  1. instantiate its configured scorer runtime and checkpoint;
  2. produce finite per-sample scores;
  3. backpropagate finite, non-zero gradients to the tactile/action tensors that
     a DP + Foresight guidance loop would optimize through.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import DistilledTacQualityEnergyRuntime  # noqa: E402
from TFAC_V5.insertion_risk_scorer_runtime import InsertionRiskScorerRuntime  # noqa: E402
from TFAC_V5.ptg_proxy_scorer_v2_runtime import PTGProxyScorerV2Runtime, TASK_TO_ID  # noqa: E402
from TFAC_V5.action_aware_scorer_runtime import ActionAwareScorerRuntime, TASK_TO_ID as ACTION_AWARE_TASK_TO_ID  # noqa: E402


DEFAULT_CONFIG = Path("/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_rollout_arm_config_smoke")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def tensor_norm(t: Optional[torch.Tensor]) -> Optional[float]:
    if t is None:
        return None
    return float(t.detach().flatten(1).norm(dim=1).mean().cpu())


def all_finite(items: Sequence[torch.Tensor]) -> bool:
    return all(torch.isfinite(item).all().item() for item in items)


def has_nonzero_grad(items: Sequence[torch.Tensor], eps: float) -> bool:
    return all(item.detach().norm().item() > eps for item in items)


def synthetic_inputs(device: torch.device, batch: int, window: int) -> Dict[str, torch.Tensor]:
    left = torch.randn(batch, window, 9, 9, 2, device=device, requires_grad=True)
    right = torch.randn(batch, window, 9, 9, 2, device=device, requires_grad=True)
    eef = torch.randn(batch, window, 6, device=device, requires_grad=True)
    joint = torch.randn(batch, window, 7, device=device, requires_grad=True)
    return {"left": left, "right": right, "eef": eef, "joint": joint}


def runtime_for_arm(runtime_name: str, checkpoint: str, device: str):
    if runtime_name == "InsertionRiskScorerRuntime":
        return InsertionRiskScorerRuntime(checkpoint, device=device)
    if runtime_name == "PTGProxyScorerV2Runtime":
        return PTGProxyScorerV2Runtime(checkpoint, device=device)
    if runtime_name == "DistilledTacQualityEnergyRuntime":
        return DistilledTacQualityEnergyRuntime(checkpoint, device=device)
    if runtime_name == "ActionAwareScorerRuntime":
        return ActionAwareScorerRuntime(checkpoint, device=device)
    raise KeyError(f"Unsupported scorer runtime: {runtime_name}")


def score_arm(task: str, runtime_name: str, runtime, x: Dict[str, torch.Tensor]) -> torch.Tensor:
    if runtime_name == "InsertionRiskScorerRuntime":
        return runtime.score(x["left"], x["joint"], mode="energy_clipped")

    task_id = torch.full((x["left"].shape[0],), TASK_TO_ID[task], dtype=torch.long, device=runtime.device)
    if runtime_name == "PTGProxyScorerV2Runtime":
        return runtime.weighted_energy_score(
            x["left"],
            right_marker_seq=x["right"],
            eef_action_seq=x["eef"],
            joint_action_seq=x["joint"],
            task_id=task_id,
            quality_weight=0.75,
            binary_weight=0.1,
            reason_weight=0.0,
            clip=True,
        )
    if runtime_name == "DistilledTacQualityEnergyRuntime":
        return runtime.score(
            x["left"],
            right_marker_seq=x["right"],
            eef_action_seq=x["eef"],
            joint_action_seq=x["joint"],
            task_id=task_id,
            mode="energy_clipped",
        )
    if runtime_name == "ActionAwareScorerRuntime":
        task_id = torch.full(
            (x["left"].shape[0],),
            ACTION_AWARE_TASK_TO_ID[task],
            dtype=torch.long,
            device=runtime.device,
        )
        return runtime.score(x["left"], x["eef"], task_id, mode="quality")
    raise KeyError(runtime_name)


def gradient_targets(runtime_name: str, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    if runtime_name == "InsertionRiskScorerRuntime":
        return [x["left"], x["joint"]]
    if runtime_name == "ActionAwareScorerRuntime":
        return [x["left"], x["eef"]]
    return [x["left"], x["right"], x["eef"], x["joint"]]


def check_guided_arm(
    task: str,
    arm_name: str,
    arm: Dict[str, Any],
    device: str,
    batch: int,
    window: int,
    eps: float,
) -> Dict[str, Any]:
    runtime_name = arm["scorer_runtime"]
    checkpoint = arm["checkpoint"]["path"]
    runtime = runtime_for_arm(runtime_name, checkpoint, device=device)
    x = synthetic_inputs(runtime.device, batch=batch, window=window)
    score = score_arm(task, runtime_name, runtime, x)
    targets = gradient_targets(runtime_name, x)
    grads = torch.autograd.grad(score.sum(), targets, retain_graph=False, allow_unused=False)
    if runtime_name == "InsertionRiskScorerRuntime":
        grad_names = ["left_marker", "joint_action"]
    elif runtime_name == "ActionAwareScorerRuntime":
        grad_names = ["left_marker", "eef_action"]
    else:
        grad_names = ["left_marker", "right_marker", "eef_action", "joint_action"]
    grad_norms = {name: tensor_norm(grad) for name, grad in zip(grad_names, grads)}
    finite = bool(torch.isfinite(score).all().item() and all_finite(grads))
    nonzero = bool(has_nonzero_grad(grads, eps))
    return {
        "task": task,
        "arm": arm_name,
        "guidance_enabled": True,
        "scorer_runtime": runtime_name,
        "checkpoint": checkpoint,
        "device": str(runtime.device),
        "score_mean": float(score.detach().mean().cpu()),
        "score_std": float(score.detach().std(unbiased=False).cpu()),
        "score_all_finite": bool(torch.isfinite(score).all().item()),
        "grad_norms": grad_norms,
        "grad_all_finite": bool(all_finite(grads)),
        "grad_all_nonzero": nonzero,
        "passes_rollout_arm_gradient_smoke": bool(finite and nonzero),
    }


def build_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    cfg = load_json(Path(args.config))
    arm_results: List[Dict[str, Any]] = []
    for task, arms in cfg["tasks"].items():
        for arm_name, arm in arms.items():
            if not arm.get("guidance_enabled", False):
                arm_results.append(
                    {
                        "task": task,
                        "arm": arm_name,
                        "guidance_enabled": False,
                        "scorer_runtime": None,
                        "passes_rollout_arm_gradient_smoke": True,
                        "skip_reason": "baseline arm has no TacQuality guidance scorer",
                    }
                )
                continue
            arm_results.append(
                check_guided_arm(
                    task,
                    arm_name,
                    arm,
                    device=args.device,
                    batch=args.batch_size,
                    window=args.window,
                    eps=args.grad_eps,
                )
            )

    guided = [row for row in arm_results if row["guidance_enabled"]]
    result = {
        "purpose": "Smoke-test machine-readable rollout arm scorer configs for differentiable DP guidance readiness.",
        "scientific_evidence": False,
        "config": str(args.config),
        "git_commit": git_commit(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "window": args.window,
        "grad_eps": args.grad_eps,
        "arms": arm_results,
        "checks": {
            "all_guided_arms_present": len(guided) == 6,
            "optional_action_aware_arms_present": sum(
                1 for row in guided if row.get("scorer_runtime") == "ActionAwareScorerRuntime"
            )
            == 2,
            "all_guided_arms_pass_gradient_smoke": all(
                row["passes_rollout_arm_gradient_smoke"] for row in guided
            ),
            "baseline_arms_recorded": sum(1 for row in arm_results if not row["guidance_enabled"]) == 2,
        },
    }
    result["overall_pass"] = all(result["checks"].values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Rollout Arm Config Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- config: `{result['config']}`",
        f"- git_commit: `{result['git_commit']}`",
        "",
        "## Arms",
        "",
        "| task | arm | scorer | guidance | pass | score_mean | grad_norms |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in result["arms"]:
        lines.append(
            "| {task} | {arm} | {scorer} | {guided} | {passed} | {score} | {grads} |".format(
                task=row["task"],
                arm=row["arm"],
                scorer=row.get("scorer_runtime"),
                guided=row["guidance_enabled"],
                passed=row["passes_rollout_arm_gradient_smoke"],
                score=row.get("score_mean", ""),
                grads=json.dumps(row.get("grad_norms", {}), ensure_ascii=False),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This smoke only proves that the configured scorer runtimes are loadable and differentiable on synthetic inputs.",
            "It does not prove real task improvement; the formal HDF5 rollout gates remain the required scientific evidence.",
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
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_eps", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build_smoke(args)
    json_path = out_dir / "tac_quality_rollout_arm_config_smoke.json"
    md_path = out_dir / "tac_quality_rollout_arm_config_smoke.md"
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
