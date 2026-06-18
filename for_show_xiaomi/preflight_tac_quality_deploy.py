#!/usr/bin/env python3
"""Preflight checks for TacQuality-guided board/insertion deployment."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("/home/chenshuai/Project/output/tac_quality_deploy_preflight")
ROLLOUT_CONFIG = Path(
    "/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_marker_joint_20260618.json"
)
BOARD_DP_RUN = Path(
    "/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619"
)
BOARD_FORESIGHT_DIR = Path(
    "/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload"
)
BOARD_FORCE_ROOT = Path("/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_scorer")
INSERTION_DP_RUN = Path("/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210")
INSERTION_VAE = Path("/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt")
INSERTION_FORESIGHT_DIR = Path("/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401")
INSERTION_ROOT = Path("/home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def path_check(path: Path, *, kind: str = "file", min_bytes: int = 1) -> dict[str, Any]:
    exists = path.exists()
    is_kind = path.is_dir() if kind == "dir" else path.is_file()
    size = path.stat().st_size if exists and path.is_file() else None
    ok = bool(exists and is_kind and (kind == "dir" or (size is not None and size >= min_bytes)))
    return {
        "path": str(path),
        "kind": kind,
        "exists": bool(exists),
        "size": size,
        "ok": ok,
    }


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, int(port))) == 0


def shell_output(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.stdout.strip()
    except Exception as exc:
        return str(exc)


def check_rollout_config(config: dict[str, Any]) -> dict[str, Any]:
    insertion = config.get("tasks", {}).get("insertion", {}).get("default_guided", {})
    board = config.get("tasks", {}).get("board", {}).get("marker_joint_guided", {})
    board_energy = board.get("refiner", {}).get("energy", {})
    return {
        "path": str(ROLLOUT_CONFIG),
        "recommended_board_arm": config.get("recommended_board_arm"),
        "insertion_default_runtime": insertion.get("scorer_runtime"),
        "insertion_default_checkpoint": (insertion.get("checkpoint") or {}).get("path"),
        "board_marker_joint_runtime": board.get("scorer_runtime"),
        "board_marker_joint_checkpoint": (board.get("checkpoint") or {}).get("path"),
        "board_marker_joint_energy_source": board_energy.get("source"),
        "checks": {
            "recommended_board_arm_marker_joint": config.get("recommended_board_arm") == "marker_joint_guided",
            "insertion_runtime_ok": insertion.get("scorer_runtime") == "InsertionRiskScorerRuntime",
            "board_runtime_ok": board.get("scorer_runtime") == "ForceBandTacQualityEnergyRuntime",
            "board_energy_marker_joint": "marker_joint_action" in str(board_energy.get("source")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--fail_on_busy_port", action="store_true",
                        help="Return nonzero if any deployment port is already listening.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(ROLLOUT_CONFIG)
    paths = {
        "board_dp_config": path_check(BOARD_DP_RUN / "config.json"),
        "board_dp_best": path_check(BOARD_DP_RUN / "dp_best.pth", min_bytes=1024),
        "board_foresight_args": path_check(BOARD_FORESIGHT_DIR / "args.json"),
        "board_foresight_ckpt": path_check(BOARD_FORESIGHT_DIR / "foresight_best.ckpt", min_bytes=1024),
        "rollout_config": path_check(ROLLOUT_CONFIG),
        "board_scorer": path_check(Path(config["tasks"]["board"]["marker_joint_guided"]["checkpoint"]["path"]), min_bytes=1024),
        "insertion_dp_config": path_check(INSERTION_DP_RUN / "config.json"),
        "insertion_dp_final": path_check(INSERTION_DP_RUN / "dp_final.pth", min_bytes=1024),
        "insertion_vae": path_check(INSERTION_VAE, min_bytes=1024),
        "insertion_foresight_args": path_check(INSERTION_FORESIGHT_DIR / "args.json"),
        "insertion_foresight_ckpt": path_check(INSERTION_FORESIGHT_DIR / "foresight_best.ckpt", min_bytes=1024),
        "insertion_scorer": path_check(Path(config["tasks"]["insertion"]["default_guided"]["checkpoint"]["path"]), min_bytes=1024),
        "board_rollout_root": path_check(BOARD_FORCE_ROOT, kind="dir"),
        "insertion_rollout_root": path_check(INSERTION_ROOT, kind="dir"),
    }
    BOARD_FORCE_ROOT.mkdir(parents=True, exist_ok=True)
    INSERTION_ROOT.mkdir(parents=True, exist_ok=True)
    paths["board_rollout_root"] = path_check(BOARD_FORCE_ROOT, kind="dir")
    paths["insertion_rollout_root"] = path_check(INSERTION_ROOT, kind="dir")

    ports = {str(port): {"listening": port_open(port)} for port in [8765, 8766, 8785, 8786]}
    config_check = check_rollout_config(config)
    path_ok = all(item["ok"] for item in paths.values())
    config_ok = all(config_check["checks"].values())
    busy_ports = [port for port, item in ports.items() if item["listening"]]
    ports_ok = not busy_ports or not args.fail_on_busy_port
    result = {
        "paths": paths,
        "rollout_config": config_check,
        "ports": ports,
        "busy_ports": busy_ports,
        "gpu": shell_output([
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]),
        "serving_processes": shell_output(["pgrep", "-af", "serve_dp_tac_quality_guided|serve_board_dp_foresight_guided|serve_dp_policy"]),
        "path_ok": bool(path_ok),
        "config_ok": bool(config_ok),
        "ports_ok": bool(ports_ok),
    }
    result["preflight_pass"] = bool(path_ok and config_ok and ports_ok)

    json_path = out_dir / "tac_quality_deploy_preflight.json"
    md_path = out_dir / "tac_quality_deploy_preflight.md"
    result["json"] = str(json_path)
    result["markdown"] = str(md_path)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(result, md_path)
    print(json.dumps({
        "preflight_pass": result["preflight_pass"],
        "path_ok": result["path_ok"],
        "config_ok": result["config_ok"],
        "busy_ports": busy_ports,
        "json": str(json_path),
        "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))
    if not result["preflight_pass"]:
        raise SystemExit(1)


def write_md(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Deploy Preflight",
        "",
        f"- preflight_pass: `{result['preflight_pass']}`",
        f"- path_ok: `{result['path_ok']}`",
        f"- config_ok: `{result['config_ok']}`",
        f"- ports_ok: `{result['ports_ok']}`",
        f"- busy_ports: `{result['busy_ports']}`",
        "",
        "## Rollout Config",
        "",
    ]
    for key, value in result["rollout_config"].items():
        if key == "checks":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Config Checks", "", "| check | pass |", "|---|---|"])
    for key, value in result["rollout_config"]["checks"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Path Checks", "", "| item | ok | path | size |", "|---|---|---|---:|"])
    for name, item in result["paths"].items():
        lines.append(f"| `{name}` | `{item['ok']}` | `{item['path']}` | {item.get('size')} |")
    lines.extend([
        "",
        "## Runtime",
        "",
        f"- gpu: `{result.get('gpu')}`",
        f"- serving_processes: `{result.get('serving_processes')}`",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
