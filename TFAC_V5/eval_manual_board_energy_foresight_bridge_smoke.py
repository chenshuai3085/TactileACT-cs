"""Foresight-bridge smoke for the manual-board TacQualityEnergy scorer.

The previous real-window smoke verifies:

  real tactile/action -> manual TacQualityEnergy -> d score / d action

This diagnostic verifies the next DP-guidance contract layer for the same
manual-board checkpoint:

  action -> Foresight bridge -> predicted tactile -> manual TacQualityEnergy
         -> d score / d action

It uses the existing synthetic Foresight bridge contract so the test remains
fast and deterministic.  It is not a real Foresight accuracy test, not DP
denoising, and not a robot rollout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFAC_V5.distilled_tac_quality_energy_runtime import (  # noqa: E402
    DistilledTacQualityEnergyRuntime,
    TASK_TO_ID,
)
from TFAC_V5.tac_quality_foresight_bridge import (  # noqa: E402
    ForesightBridgeConfig,
    ForesightTacQualityBridge,
    SyntheticLatentForesight,
)
from TFAC_V5.tac_quality_guidance_config import get_guidance_profile  # noqa: E402
from TFAC_V5.tac_quality_trust_region_guidance import TacQualityTrustRegionRefiner, TrustRegionConfig  # noqa: E402


DEFAULT_CKPT = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy/distilled_tac_quality_energy_final.pt")
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/manual_board_tac_quality_energy_foresight_bridge_smoke")


def summarize_tensor(x: torch.Tensor) -> Dict[str, float]:
    arr = x.detach().float().flatten()
    if arr.numel() == 0:
        return {"n": 0}
    return {
        "n": int(arr.numel()),
        "mean": float(arr.mean().cpu()),
        "std": float(arr.std(unbiased=False).cpu()),
        "min": float(arr.min().cpu()),
        "max": float(arr.max().cpu()),
    }


def task_refiner(task: str, action_step_scale: float) -> TacQualityTrustRegionRefiner:
    refine = get_guidance_profile(task).refinement
    return TacQualityTrustRegionRefiner(
        TrustRegionConfig(
            steps=refine.refine_steps,
            step_size=refine.action_step * action_step_scale,
            max_total_delta=refine.max_total_delta * action_step_scale,
            accept_only_improved=refine.accept_only_improved,
        )
    )


def synthetic_bridge(task: str, args: argparse.Namespace, device: torch.device) -> ForesightTacQualityBridge:
    action_dim = args.action_dim
    fs_norm: Mapping[str, Any] = {
        "action_mean": torch.zeros(action_dim, device=device),
        "action_std": torch.ones(action_dim, device=device),
        "qpos_mean": torch.zeros(action_dim, device=device),
        "qpos_std": torch.ones(action_dim, device=device),
    }
    qpos = torch.zeros(1, action_dim, device=device)
    images = [
        torch.zeros(1, 3, 64, 64, device=device),
        torch.zeros(1, 3, 64, 64, device=device),
    ]
    marker_window = torch.zeros(1, args.window, 9, 9, 2, device=device)
    foresight = SyntheticLatentForesight(
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        pred_steps=args.pred_steps,
    ).to(device)
    return ForesightTacQualityBridge(
        foresight,
        fs_norm,
        qpos_raw=qpos,
        foresight_images=images,
        marker_window_norm=marker_window,
        config=ForesightBridgeConfig(
            task=task,
            window=args.window,
            action_chunk=args.horizon,
            latent_dim=args.latent_dim,
            board_right_source="mirror_left",
        ),
    )


def score_manual(
    runtime: DistilledTacQualityEnergyRuntime,
    task: str,
    tactile: Dict[str, torch.Tensor],
    action: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    task_id = torch.full(
        (action.shape[0],),
        TASK_TO_ID[task],
        dtype=torch.long,
        device=runtime.device,
    )
    return runtime.score(
        tactile["left_marker_seq"],
        right_marker_seq=tactile.get("right_marker_seq"),
        eef_action_seq=tactile.get("eef_action_seq"),
        joint_action_seq=action,
        task_id=task_id,
        mode=mode,
    )


def grad_probe(score_fn, action: torch.Tensor) -> Dict[str, object]:
    x = action.detach().clone().requires_grad_(True)
    score = score_fn(x)
    grad = torch.autograd.grad(score.sum(), x, retain_graph=False)[0]
    grad_norm = grad.flatten(1).norm(dim=1)
    return {
        "score": summarize_tensor(score),
        "grad_norm": summarize_tensor(grad_norm),
        "finite_grad_rate": float(torch.isfinite(grad).flatten(1).all(dim=1).float().mean().cpu()),
        "positive_grad_rate": float((grad_norm > 1e-8).float().mean().cpu()),
    }


def refiner_finite_grad_rate(report: Dict[str, object]) -> float:
    logs = report.get("logs", [])
    if not logs:
        return 0.0
    values = [float(row.get("finite_grad_rate", 0.0)) for row in logs]
    return min(values)


def run_task(task: str, args: argparse.Namespace, runtime: DistilledTacQualityEnergyRuntime) -> Dict[str, object]:
    device = runtime.device
    bridge = synthetic_bridge(task, args, device)
    action = torch.randn(args.batch_size, args.horizon, args.action_dim, device=device) * args.action_std

    def score_fn(candidate: torch.Tensor) -> torch.Tensor:
        tactile = bridge(candidate)
        return score_manual(runtime, task, tactile, candidate, args.score_mode)

    tactile = bridge(action.detach().clone().requires_grad_(True))
    probe = grad_probe(score_fn, action)
    refined, report = task_refiner(task, args.action_step_scale).refine(action, score_fn)
    final_score = score_fn(refined.detach().clone().requires_grad_(True)).detach()
    base_score = score_fn(action.detach().clone().requires_grad_(True)).detach()
    result = {
        "task": task,
        "tactile_shapes": {key: list(value.shape) for key, value in tactile.items()},
        "bridge_score_gradient": probe,
        "adapter_report": report,
        "refiner_finite_grad_rate_min": refiner_finite_grad_rate(report),
        "base_score_recomputed": summarize_tensor(base_score),
        "final_score_recomputed": summarize_tensor(final_score),
        "raw_action_delta": summarize_tensor((refined - action).flatten(1).norm(dim=1)),
    }
    result["passes"] = bool(
        probe["finite_grad_rate"] >= args.min_finite_grad_rate
        and probe["positive_grad_rate"] >= args.min_positive_grad_rate
        and result["refiner_finite_grad_rate_min"] >= args.min_finite_grad_rate
        and report["improved_rate"] >= args.min_improved_rate
        and report["max_delta_within_trust_region"]
    )
    return result


def write_markdown(result: Dict[str, object], path: Path) -> None:
    ins = result["insertion"]
    board = result["board"]
    lines = [
        "# Manual-board TacQualityEnergy Foresight-bridge Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- score_mode: `{result['score_mode']}`",
        f"- action_step_scale: `{result['config']['action_step_scale']}`",
        f"- device: `{result['device']}`",
        "",
        "## Insertion",
        "",
        f"- pass: `{ins['passes']}`",
        f"- improved_rate: `{ins['adapter_report']['improved_rate']}`",
        f"- bridge_positive_grad_rate: `{ins['bridge_score_gradient']['positive_grad_rate']}`",
        f"- bridge_grad_norm_mean: `{ins['bridge_score_gradient']['grad_norm']['mean']}`",
        f"- score_delta_mean: `{ins['adapter_report']['score_delta']['mean']}`",
        "",
        "## Board",
        "",
        f"- pass: `{board['passes']}`",
        f"- improved_rate: `{board['adapter_report']['improved_rate']}`",
        f"- bridge_positive_grad_rate: `{board['bridge_score_gradient']['positive_grad_rate']}`",
        f"- bridge_grad_norm_mean: `{board['bridge_score_gradient']['grad_norm']['mean']}`",
        f"- score_delta_mean: `{board['adapter_report']['score_delta']['mean']}`",
        "",
        "## Interpretation",
        "",
        "This verifies the differentiable contract for the manual-board checkpoint: action -> Foresight-style predicted tactile -> TacQualityEnergy -> action gradient.",
        "It is a synthetic-bridge contract test, not a real Foresight accuracy or robot rollout result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--pred_steps", type=int, default=4)
    parser.add_argument("--action_std", type=float, default=0.1)
    parser.add_argument("--score_mode", default="energy_clipped")
    parser.add_argument("--action_step_scale", type=float, default=0.5)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_improved_rate", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=52)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = DistilledTacQualityEnergyRuntime(str(args.checkpoint), device=args.device)
    insertion = run_task("insertion", args, runtime)
    board = run_task("board", args, runtime)
    result = {
        "purpose": "Synthetic Foresight-bridge action-gradient smoke for the manual-board TacQualityEnergy scorer.",
        "scope": "Differentiable contract test only; not real Foresight accuracy, DP denoising, or robot rollout validation.",
        "checkpoint": str(args.checkpoint),
        "score_mode": args.score_mode,
        "device": str(runtime.device),
        "config": {
            "batch_size": int(args.batch_size),
            "horizon": int(args.horizon),
            "window": int(args.window),
            "action_dim": int(args.action_dim),
            "latent_dim": int(args.latent_dim),
            "pred_steps": int(args.pred_steps),
            "action_std": float(args.action_std),
            "action_step_scale": float(args.action_step_scale),
            "min_finite_grad_rate": float(args.min_finite_grad_rate),
            "min_positive_grad_rate": float(args.min_positive_grad_rate),
            "min_improved_rate": float(args.min_improved_rate),
            "seed": int(args.seed),
        },
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "insertion": insertion,
        "board": board,
    }
    result["overall_pass"] = bool(insertion["passes"] and board["passes"])
    json_path = args.output_dir / "manual_board_energy_foresight_bridge_smoke.json"
    md_path = args.output_dir / "manual_board_energy_foresight_bridge_smoke.md"
    result["output_json"] = str(json_path)
    result["output_markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "insertion_pass": insertion["passes"],
                "insertion_improved_rate": insertion["adapter_report"]["improved_rate"],
                "board_pass": board["passes"],
                "board_improved_rate": board["adapter_report"]["improved_rate"],
                "json": str(json_path),
                "md": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
