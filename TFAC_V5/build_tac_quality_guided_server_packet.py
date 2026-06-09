"""Build launch commands for TacQuality-guided DP server dry-runs.

This packet is deliberately honest about the current deployment state:

  - baseline DP server commands can be launched with existing serve_dp_policy.py;
  - TacQuality final-action gradient guidance has helper/bridge/preflight code;
  - a production server entrypoint still needs to call that helper inside the
    DP loop before this can run as a real guided server.

It prevents confusing old reranking servers with classifier guidance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict


OUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_guided_server_packet")
DEFAULT_SERVING_PACKET = Path(
    "/home/chenshuai/Project/output/tac_quality_serving_packet/"
    "auto_discovered/tac_quality_serving_packet.json"
)
BASELINE_SERVER = Path("for_show_xiaomi/serve_dp_policy.py")
GUIDED_SERVER_ENTRYPOINT = Path("for_show_xiaomi/serve_dp_tac_quality_guided.py")


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


def ckpt_name(path: str) -> str:
    return Path(path).name


def baseline_command(task: str, packet: Dict[str, Any], port: int, gpu: int) -> str:
    inputs = packet["packets"][task]["inputs"]
    return (
        f"python -m for_show_xiaomi.serve_dp_policy "
        f"--ckpt_dir {inputs['dp_ckpt_dir']} "
        f"--ckpt_name {ckpt_name(packet['auto_pairs'][task]['dp']['checkpoint'])} "
        f"--port {port} --gpu {gpu} --action_horizon 8"
    )


def guided_command_template(task: str, arm: str, packet: Dict[str, Any], port: int, gpu: int) -> str:
    inputs = packet["packets"][task]["inputs"]
    return (
        f"python -m for_show_xiaomi.serve_dp_tac_quality_guided "
        f"--task {task} --arm {arm} "
        f"--ckpt_dir {inputs['dp_ckpt_dir']} "
        f"--ckpt_name {ckpt_name(packet['auto_pairs'][task]['dp']['checkpoint'])} "
        f"--foresight_dir {inputs['foresight_dir']} "
        f"--foresight_ckpt {inputs['foresight_ckpt']} "
        f"--rollout_arm_config {inputs['rollout_arm_config']} "
        f"--port {port} --gpu {gpu} --action_horizon 8"
    )


def integration_steps() -> str:
    return "\n".join(
        [
            "1. Copy the baseline server loop from for_show_xiaomi/serve_dp_policy.py or wrap it in a new entrypoint.",
            "2. Load Foresight with the auto-discovered foresight_dir/foresight_ckpt.",
            "3. Build ForesightTacQualityBridge from current qpos, raw images, marker window, and fs_norm.",
            "4. Build TacQualityServingGuidance from task/arm and DP norm_stats.",
            "5. After DDPM produces the final clean action chunk, call helper.guide_action_chunk(action_norm, bridge).",
            "6. Denormalize the returned guided_action_norm and send the selected receding-horizon action.",
            "7. Log report fields: base_score, guided_score, improved_rate, delta_norm, called_from_inference_mode.",
        ]
    )


def build(args: argparse.Namespace) -> Dict[str, Any]:
    packet = load_json(Path(args.serving_packet))
    tasks: Dict[str, Any] = {}
    for idx, task in enumerate(["insertion", "board"]):
        base_port = args.base_port + 10 * idx
        tasks[task] = {
            "baseline_command": baseline_command(task, packet, base_port, args.gpu),
            "default_guided_command_template": guided_command_template(task, "default_guided", packet, base_port + 1, args.gpu),
            "distilled_guided_command_template": guided_command_template(task, "distilled_guided", packet, base_port + 2, args.gpu),
            "selected_dp": packet["auto_pairs"][task]["dp"],
            "selected_foresight": packet["auto_pairs"][task]["foresight"],
        }
    result = {
        "purpose": "Launch packet for TacQuality-guided DP server dry-runs.",
        "scientific_evidence": False,
        "git_commit": git_commit(),
        "serving_packet": str(args.serving_packet),
        "serving_packet_ready": bool(packet.get("serving_ready")),
        "baseline_server": file_info(BASELINE_SERVER),
        "guided_server_entrypoint": file_info(GUIDED_SERVER_ENTRYPOINT),
        "guided_server_ready": bool(GUIDED_SERVER_ENTRYPOINT.exists()),
        "not_reranking": True,
        "not_every_step_ddpm_guidance": True,
        "tasks": tasks,
        "integration_steps": integration_steps(),
    }
    result["launch_packet_ready"] = bool(
        result["serving_packet_ready"]
        and result["baseline_server"]["exists"]
        and result["guided_server_ready"]
    )
    result["next_step"] = (
        "Implement for_show_xiaomi/serve_dp_tac_quality_guided.py using TacQualityServingGuidance and ForesightTacQualityBridge."
        if not result["guided_server_ready"]
        else "Run real baseline-vs-guided robot/production rollouts with the generated commands."
    )
    return result


def write_markdown(result: Dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Guided Server Launch Packet",
        "",
        f"- launch_packet_ready: `{result['launch_packet_ready']}`",
        f"- guided_server_ready: `{result['guided_server_ready']}`",
        f"- scientific_evidence: `{result['scientific_evidence']}`",
        f"- next_step: {result['next_step']}",
        "",
        "## Commands",
        "",
    ]
    for task, info in result["tasks"].items():
        lines.extend(
            [
                f"### {task}",
                "",
                "Baseline:",
                "",
                "```bash",
                info["baseline_command"],
                "```",
                "",
                "TacQuality default guided template:",
                "",
                "```bash",
                info["default_guided_command_template"],
                "```",
                "",
                "TacQuality distilled guided template:",
                "",
                "```bash",
                info["distilled_guided_command_template"],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Integration Steps",
            "",
            "```text",
            result["integration_steps"],
            "```",
            "",
            "## Boundary",
            "",
            "The guided command templates point to the final-action TacQuality guidance server entrypoint.",
            "Do not use the older reranking servers as substitutes for classifier guidance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving_packet", default=str(DEFAULT_SERVING_PACKET))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="auto_discovered")
    parser.add_argument("--base_port", type=int, default=8766)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    result = build(args)
    json_path = out_dir / "tac_quality_guided_server_packet.json"
    md_path = out_dir / "tac_quality_guided_server_packet.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(
        json.dumps(
            {
                "launch_packet_ready": result["launch_packet_ready"],
                "guided_server_ready": result["guided_server_ready"],
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
