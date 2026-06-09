"""Run real-Foresight dry-run smoke for all TacQuality guided server arms.

This is an engineering gate for the final-action classifier-guidance server.
It verifies that both tasks and both guided arms can instantiate:

  DP/Foresight checkpoints -> ForesightTacQualityBridge
  -> TacQualityServingGuidance -> d score / d action

It is still not robot evidence.  It is meant to catch deployment wiring issues
before collecting the formal baseline/default/distilled rollout data.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from for_show_xiaomi.serve_dp_tac_quality_guided import dry_run_guidance_smoke  # noqa: E402


DEFAULT_SERVING_PACKET = Path(
    "/home/chenshuai/Project/output/tac_quality_serving_packet/"
    "auto_discovered/tac_quality_serving_packet.json"
)
DEFAULT_ROLLOUT_ARM_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/"
    "tac_quality_rollout_arm_configs.json"
)
DEFAULT_OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guided_server_real_foresight_smoke")


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def ckpt_name(path: str) -> str:
    return Path(path).name


def make_namespace(args: argparse.Namespace, task: str, arm: str, packet: Dict[str, Any], out_path: Path) -> argparse.Namespace:
    inputs = packet["packets"][task]["inputs"]
    return argparse.Namespace(
        task=task,
        arm=arm,
        ckpt_dir=inputs["dp_ckpt_dir"],
        ckpt_name=ckpt_name(packet["auto_pairs"][task]["dp"]["checkpoint"]),
        foresight_dir=inputs["foresight_dir"],
        foresight_ckpt=inputs["foresight_ckpt"],
        rollout_arm_config=str(args.rollout_arm_config),
        host="0.0.0.0",
        port=0,
        gpu=args.gpu,
        scheduler="ddim",
        num_inference_steps=None,
        action_horizon=8,
        action_skip=0,
        max_timesteps=1,
        seed=args.seed,
        dp_norm_mode="minmax",
        no_ema=False,
        send_guidance_report=False,
        dry_run_guidance_smoke=True,
        synthetic_foresight_for_smoke=False,
        smoke_output=str(out_path),
        min_finite_grad_rate=args.min_finite_grad_rate,
        min_positive_grad_rate=args.min_positive_grad_rate,
    )


def check_report(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    report = row.get("report", {})
    return bool(
        row.get("dry_run_guidance_smoke_pass") is True
        and report.get("finite_grad_rate", 0.0) >= args.min_finite_grad_rate
        and report.get("positive_grad_rate", 0.0) >= args.min_positive_grad_rate
        and report.get("max_delta_within_trust_region") is True
        and report.get("called_from_inference_mode") is True
        and report.get("returned_requires_grad") is False
        and report.get("integration_contract", {}).get("reranking") is False
        and report.get("integration_contract", {}).get("every_step_ddpm_guidance") is False
    )


def build(args: argparse.Namespace) -> Dict[str, Any]:
    packet = load_json(Path(args.serving_packet))
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    arms: List[Dict[str, Any]] = []
    for task in ["insertion", "board"]:
        for arm in ["default_guided", "distilled_guided"]:
            arm_out = out_dir / f"{task}_{arm}_real_foresight_smoke.json"
            arm_args = make_namespace(args, task, arm, packet, arm_out)
            row = dry_run_guidance_smoke(arm_args)
            row["task"] = task
            row["arm"] = arm
            row["smoke_output"] = str(arm_out)
            row["passes_all_arm_real_foresight_smoke"] = check_report(row, args)
            arms.append(row)
    result = {
        "purpose": "Real-Foresight dry-run smoke for all TacQuality guided server arms.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "serving_packet": str(args.serving_packet),
        "rollout_arm_config": str(args.rollout_arm_config),
        "tag": args.tag,
        "device_request": "cpu" if args.gpu < 0 else f"cuda:{args.gpu}",
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "arms": arms,
        "checks": {
            "n_guided_arms": len(arms),
            "all_four_guided_arms_present": len(arms) == 4,
            "all_guided_arms_pass_real_foresight_smoke": all(
                row["passes_all_arm_real_foresight_smoke"] for row in arms
            ),
        },
    }
    result["overall_pass"] = all(result["checks"].values())
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Guided Server Real-Foresight Smoke",
        "",
        f"- overall_pass: `{result['overall_pass']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- not_reranking: `{result['not_reranking']}`",
        f"- not_every_step_ddpm_guidance: `{result['not_every_step_ddpm_guidance']}`",
        f"- git_commit: `{result['git_commit']}`",
        "",
        "| task | arm | pass | improved_rate | finite_grad | positive_grad | raw_delta_mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["arms"]:
        report = row.get("report", {})
        raw_delta = report.get("raw_action_delta", {})
        lines.append(
            "| {task} | {arm} | {passed} | {improved} | {finite} | {positive} | {delta} |".format(
                task=row["task"],
                arm=row["arm"],
                passed=row["passes_all_arm_real_foresight_smoke"],
                improved=report.get("improved_rate"),
                finite=report.get("finite_grad_rate"),
                positive=report.get("positive_grad_rate"),
                delta=raw_delta.get("mean"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This smoke uses the real DP/Foresight paths from the auto-discovered serving packet.",
            "It verifies deployment wiring for the default and distilled guided arms used by the formal three-arm scorer ablation.",
            "It does not replace real baseline-vs-guided robot/production rollout gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving_packet", default=str(DEFAULT_SERVING_PACKET))
    parser.add_argument("--rollout_arm_config", default=str(DEFAULT_ROLLOUT_ARM_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="auto_discovered_all_arms")
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--min_finite_grad_rate", type=float, default=0.999)
    parser.add_argument("--min_positive_grad_rate", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_guided_server_real_foresight_smoke.json"
    md_path = out_dir / "tac_quality_guided_server_real_foresight_smoke.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "overall_pass": result["overall_pass"],
                "n_guided_arms": result["checks"]["n_guided_arms"],
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
