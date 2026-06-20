#!/usr/bin/env python3
"""Audit server-side rollout log schema for TacQuality real evidence.

The audit can run in two modes:

1. Synthetic smoke mode: create a temporary rollout with ServerRolloutLogger and
   verify the files/columns needed by downstream evaluators.
2. Existing log mode: inspect a real trial directory or rollout root.

It does not count synthetic logs as real robot evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from for_show_xiaomi.server_rollout_logger import ServerRolloutLogger


DEFAULT_OUTPUT_DIR = "/home/chenshuai/Project/output/tac_quality_server_rollout_schema_audit"
CORE_COLUMNS = {
    "step",
    "t",
    "wall_time",
    "ft_fz",
    "ft_f_mag",
    "left_fz",
    "left_f_mag",
    "right_fz",
    "right_f_mag",
    "left_marker_mag_mean",
    "left_marker_mag_max",
    "left_marker_contact_area",
    "action_0",
    "action_norm_0",
}
GUIDANCE_COLUMNS = {
    "guidance_score_delta",
    "guidance_accept_rate",
    "guidance_contact_gate_value",
}
METADATA_KEYS = {
    "created_at",
    "log_side",
    "episode",
    "trial",
    "host",
    "port",
    "task",
    "arm",
    "group",
    "server_metadata",
    "steps",
    "stop_reason",
    "summary",
    "artifacts",
}
ARTIFACT_KEYS = {"force_trace_csv", "force_trace_npz", "force_curve_png"}


def read_csv_columns(path: Path) -> tuple[list[str], int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), len(rows)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def make_obs(step: int) -> dict[str, Any]:
    marker = np.zeros((9, 9, 2), dtype=np.float32)
    marker[..., 0] = 0.1 + 0.01 * step
    marker[..., 1] = 0.2 + 0.02 * step
    force = np.array([1.0 + step, -0.5, 6.0 + 0.2 * step, 0.1, 0.2, 0.3], dtype=np.float32)
    return {
        "ft": force,
        "qpos": np.linspace(0.0, 1.0, 7, dtype=np.float32) + step * 0.01,
        "eef": np.linspace(0.1, 0.8, 7, dtype=np.float32) + step * 0.01,
        "tac": {
            "left": {
                "force6d": force + 0.5,
                "marker_offset": marker,
            },
            "right": {
                "force6d": force + 1.0,
                "marker_offset": marker * 0.5,
            },
        },
    }


def make_guidance_report(step: int) -> dict[str, Any]:
    return {
        "score_delta": 0.01 * (step + 1),
        "accept_rate": 1.0,
        "contact_gate_value": 0.8,
        "finite_grad_rate": 1.0,
        "positive_grad_rate": 1.0,
        "nested": {"trust_region_pass_rate": 1.0},
    }


def create_synthetic_rollout(root: Path, *, task: str, arm: str, port: int, steps: int) -> Path:
    logger = ServerRolloutLogger(
        root,
        episode=0,
        host="127.0.0.1",
        port=port,
        task=task,
        arm=arm,
        server_metadata={
            "protocol": "schema_smoke",
            "task": task,
            "arm": arm,
            "guidance": "schema_smoke",
            "reranking": False,
            "synthetic_smoke": True,
            "not_real_robot_evidence": True,
        },
    )
    for step in range(steps):
        action = np.linspace(0.0, 0.6, 7, dtype=np.float32) + step * 0.01
        action_norm = np.linspace(-0.5, 0.5, 7, dtype=np.float32) + step * 0.01
        logger.record(
            step=step,
            obs=make_obs(step),
            action=action,
            action_norm=action_norm,
            guidance_report=make_guidance_report(step),
        )
    logger.metadata["synthetic_smoke"] = True
    logger.metadata["not_real_robot_evidence"] = True
    logger.finalize(steps=steps, stop_reason="schema_smoke")
    return logger.trial_dir


def discover_trial_dirs(path: Path) -> list[Path]:
    if (path / "force_trace.csv").exists():
        return [path]
    return sorted(p.parent for p in path.rglob("force_trace.csv"))


def audit_trial(trial_dir: Path, *, require_guidance: bool) -> dict[str, Any]:
    csv_path = trial_dir / "force_trace.csv"
    npz_path = trial_dir / "force_trace.npz"
    meta_path = trial_dir / "metadata.json"
    png_path = trial_dir / "force_curve.png"
    columns, n_rows = read_csv_columns(csv_path) if csv_path.exists() else ([], 0)
    col_set = set(columns)
    meta = load_json(meta_path)
    artifact = meta.get("artifacts") if isinstance(meta.get("artifacts"), dict) else {}
    server_meta = meta.get("server_metadata") if isinstance(meta.get("server_metadata"), dict) else {}
    missing_core = sorted(CORE_COLUMNS - col_set)
    missing_guidance = sorted(GUIDANCE_COLUMNS - col_set) if require_guidance else []
    missing_metadata = sorted(METADATA_KEYS - set(meta))
    missing_artifacts = sorted(ARTIFACT_KEYS - set(artifact))
    synthetic = bool(
        meta.get("synthetic_smoke")
        or meta.get("not_real_robot_evidence")
        or server_meta.get("synthetic_smoke")
        or server_meta.get("not_real_robot_evidence")
        or "synthetic" in str(trial_dir).lower()
    )
    checks = {
        "force_trace_csv_exists": csv_path.exists(),
        "force_trace_npz_exists": npz_path.exists(),
        "metadata_exists": meta_path.exists(),
        "force_curve_png_exists": png_path.exists(),
        "nonempty_csv": n_rows > 0,
        "core_columns_present": not missing_core,
        "guidance_columns_present": not missing_guidance,
        "metadata_keys_present": not missing_metadata,
        "artifact_keys_present": not missing_artifacts,
        "metadata_task_present": bool(meta.get("task")),
        "metadata_arm_present": bool(meta.get("arm")),
        "metadata_port_present": meta.get("port") is not None,
        "metadata_server_protocol_present": bool(server_meta.get("protocol")),
    }
    return {
        "trial_dir": str(trial_dir),
        "synthetic": synthetic,
        "n_rows": n_rows,
        "n_columns": len(columns),
        "columns": columns,
        "metadata": {
            "task": meta.get("task"),
            "arm": meta.get("arm"),
            "group": meta.get("group"),
            "port": meta.get("port"),
            "steps": meta.get("steps"),
            "stop_reason": meta.get("stop_reason"),
            "server_protocol": server_meta.get("protocol"),
            "server_guidance": server_meta.get("guidance"),
            "reranking": server_meta.get("reranking"),
        },
        "missing_core_columns": missing_core,
        "missing_guidance_columns": missing_guidance,
        "missing_metadata_keys": missing_metadata,
        "missing_artifact_keys": missing_artifacts,
        "checks": checks,
        "schema_pass": all(checks.values()),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_trials": len(rows),
        "schema_pass": bool(rows) and all(row["schema_pass"] for row in rows),
        "synthetic_count": sum(1 for row in rows if row["synthetic"]),
        "real_count": sum(1 for row in rows if not row["synthetic"]),
        "failed_trials": [row["trial_dir"] for row in rows if not row["schema_pass"]],
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Server Rollout Schema Audit",
        "",
        f"- schema_pass: `{result['summary']['schema_pass']}`",
        f"- n_trials: `{result['summary']['n_trials']}`",
        f"- real_count: `{result['summary']['real_count']}`",
        f"- synthetic_count: `{result['summary']['synthetic_count']}`",
        "",
        "Synthetic schema-smoke logs do not count as real robot evidence.",
        "",
        "## Trials",
        "",
        "| trial | pass | rows | synthetic | task | arm | guidance |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in result["trials"]:
        meta = row["metadata"]
        lines.append(
            f"| `{row['trial_dir']}` | `{row['schema_pass']}` | {row['n_rows']} | "
            f"`{row['synthetic']}` | `{meta.get('task')}` | `{meta.get('arm')}` | "
            f"`{meta.get('server_guidance')}` |"
        )
    lines.extend(["", "## Failed Checks", ""])
    for row in result["trials"]:
        failed = [key for key, value in row["checks"].items() if not value]
        if not failed:
            continue
        lines.append(f"- `{row['trial_dir']}`: {failed}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Existing trial dir or rollout root to audit.")
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--tag", default="schema_smoke")
    parser.add_argument("--task", choices=["board", "insertion"], default="board")
    parser.add_argument("--arm", default="marker_joint_s12_guided")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--require_guidance", action="store_true", default=True)
    parser.add_argument("--no_require_guidance", action="store_false", dest="require_guidance")
    parser.add_argument("--keep_synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.expanduser() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir: Path | None = None
    try:
        if args.input:
            trial_dirs = discover_trial_dirs(args.input.expanduser())
            synthetic_root = None
        else:
            tmp_dir = Path(tempfile.mkdtemp(prefix="tac_quality_schema_", dir="/tmp"))
            synthetic_root = tmp_dir / "rollouts"
            trial_dirs = [create_synthetic_rollout(synthetic_root, task=args.task, arm=args.arm, port=args.port, steps=args.steps)]
        trials = [audit_trial(path, require_guidance=args.require_guidance) for path in trial_dirs]
        result = {
            "input": str(args.input) if args.input else None,
            "synthetic_root": str(synthetic_root) if not args.input else None,
            "require_guidance": bool(args.require_guidance),
            "summary": summarize(trials),
            "trials": trials,
            "evidence_boundary": "Synthetic schema-smoke logs only validate logger/evaluator schema; they are not real robot evidence.",
        }
        json_path = out_dir / "tac_quality_server_rollout_schema_audit.json"
        md_path = out_dir / "tac_quality_server_rollout_schema_audit.md"
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        write_markdown(result, md_path)
        print(json.dumps({
            "json": str(json_path),
            "markdown": str(md_path),
            "schema_pass": result["summary"]["schema_pass"],
            "n_trials": result["summary"]["n_trials"],
            "synthetic_count": result["summary"]["synthetic_count"],
            "real_count": result["summary"]["real_count"],
        }, ensure_ascii=False, indent=2))
        if not result["summary"]["schema_pass"]:
            raise SystemExit(1)
    finally:
        if tmp_dir and tmp_dir.exists() and not args.keep_synthetic:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
