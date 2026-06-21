#!/usr/bin/env python3
"""Create paired real-rollout manifest for TacQuality baseline/guided tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "/home/chenshuai/Project/output/tac_quality_real_rollout_manifest"
DEFAULT_BOARD_ROOT = "/home/chenshuai/Project/output/board_force_rollouts/260617_only_force_aware_scorer"
DEFAULT_INSERTION_ROOT = "/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer"


TASK_DEFAULTS = {
    "board": {
        "baseline_port": 8765,
        "guided_port": 8769,
        "baseline_arm": "baseline",
        "guided_arm": "force_aware_guided",
        "root": DEFAULT_BOARD_ROOT,
        "pair_prefix": "board_force_aware",
    },
    "insertion": {
        "baseline_port": 8785,
        "guided_port": 8786,
        "baseline_arm": "baseline",
        "guided_arm": "good_margin_guided",
        "root": DEFAULT_INSERTION_ROOT,
        "pair_prefix": "insertion",
    },
}


def parse_tasks(text: str) -> list[str]:
    tasks = [item.strip() for item in text.split(",") if item.strip()]
    invalid = [task for task in tasks if task not in TASK_DEFAULTS]
    if invalid:
        raise SystemExit(f"Unknown task(s): {invalid}. Valid: {sorted(TASK_DEFAULTS)}")
    return tasks


def trial_sequence(n_pairs: int, order: str) -> list[tuple[int, str]]:
    if order == "all_baseline_then_guided":
        return [(i, "baseline") for i in range(1, n_pairs + 1)] + [
            (i, "guided") for i in range(1, n_pairs + 1)
        ]
    if order == "guided_first_interleaved":
        return [(i, group) for i in range(1, n_pairs + 1) for group in ["guided", "baseline"]]
    return [(i, group) for i in range(1, n_pairs + 1) for group in ["baseline", "guided"]]


def make_client_command(row: dict[str, Any], host: str, manifest_csv: str = "<manifest_csv>") -> str:
    return (
        "cd /home/chenshuai/Project/TactileACT-cs && "
        "python for_show_xiaomi/ws_client.py "
        f"--host {host} "
        f"--port {row['server_port']} "
        "--disable_force_log "
        f"--rollout_pair_id {row['pair_id']} "
        f"--rollout_trial_order {row['trial_order']} "
        f"--rollout_task {row['task']} "
        f"--rollout_group {row['group']} "
        f"--rollout_server_arm {row['server_arm']} "
        f"--rollout_manifest_csv {manifest_csv}"
    )


def make_rows(args: argparse.Namespace, *, manifest_csv: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = parse_tasks(args.tasks)
    trial_order = 1
    overrides = {
        "board": {
            "baseline_port": args.board_baseline_port,
            "guided_port": args.board_guided_port,
            "baseline_arm": args.board_baseline_arm,
            "guided_arm": args.board_guided_arm,
            "root": args.board_root,
            "pair_prefix": args.board_pair_prefix,
        },
        "insertion": {
            "baseline_port": args.insertion_baseline_port,
            "guided_port": args.insertion_guided_port,
            "baseline_arm": args.insertion_baseline_arm,
            "guided_arm": args.insertion_guided_arm,
            "root": args.insertion_root,
            "pair_prefix": args.insertion_pair_prefix,
        },
    }
    for task in tasks:
        cfg = {**TASK_DEFAULTS[task], **{k: v for k, v in overrides[task].items() if v is not None}}
        n_pairs = args.board_pairs if task == "board" else args.insertion_pairs
        for pair_idx, group in trial_sequence(n_pairs, args.order):
            pair_id = f"{cfg['pair_prefix']}_{pair_idx:03d}"
            arm = cfg[f"{group}_arm"]
            port = cfg[f"{group}_port"]
            root = Path(str(cfg["root"]))
            group_root = root / group
            row = {
                "trial_order": trial_order,
                "task": task,
                "pair_id": pair_id,
                "group": group,
                "server_port": port,
                "server_arm": arm,
                "server_log_root": str(root),
                "expected_group_dir": str(group_root),
                "expected_trial_dir_pattern": str(group_root / f"*_port{port}_episode*"),
                "client_command": "",
                "run_status": "pending",
                "real_robot": "yes",
                "success": "" if task == "insertion" else "n/a",
                "stopped_early": "" if task == "insertion" else "n/a",
                "bounce_count": "" if task == "insertion" else "n/a",
                "retry_count": "" if task == "insertion" else "n/a",
                "notes": "",
            }
            rows.append(row)
            trial_order += 1
    for row in rows:
        row["client_command"] = make_client_command(row, args.client_host, manifest_csv=manifest_csv)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "trial_order",
        "task",
        "pair_id",
        "group",
        "server_port",
        "server_arm",
        "server_log_root",
        "expected_group_dir",
        "expected_trial_dir_pattern",
        "client_command",
        "run_status",
        "real_robot",
        "success",
        "stopped_early",
        "bounce_count",
        "retry_count",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, Any]], path: Path, args: argparse.Namespace, csv_path: Path, json_path: Path) -> None:
    tasks = parse_tasks(args.tasks)
    board_root = next((row["server_log_root"] for row in rows if row["task"] == "board"), TASK_DEFAULTS["board"]["root"])
    board_guided_arm = next((row["server_arm"] for row in rows if row["task"] == "board" and row["group"] == "guided"), TASK_DEFAULTS["board"]["guided_arm"])
    insertion_root = next((row["server_log_root"] for row in rows if row["task"] == "insertion"), TASK_DEFAULTS["insertion"]["root"])
    insertion_guided_arm = next((row["server_arm"] for row in rows if row["task"] == "insertion" and row["group"] == "guided"), TASK_DEFAULTS["insertion"]["guided_arm"])
    eval_lines = [
        "conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_tac_quality_real_rollouts.py \\",
        f"  --board_root {board_root} \\",
        f"  --insertion_root {insertion_root} \\",
        "  --output_dir /home/chenshuai/Project/output/tac_quality_real_rollout_eval \\",
        f"  --tag {args.tag}_eval \\",
        "  --board_expected_baseline_arm baseline \\",
        f"  --board_expected_guided_arm {board_guided_arm} \\",
        "  --insertion_expected_baseline_arm baseline \\",
        f"  --insertion_expected_guided_arm {insertion_guided_arm} \\",
    ]
    if "board" not in tasks:
        eval_lines.append("  --skip_board \\")
    if "insertion" not in tasks:
        eval_lines.append("  --skip_insertion \\")
    eval_lines.append("  --pairing_strategy explicit")
    lines = [
        "# TacQuality Real Rollout Manifest",
        "",
        "Purpose: run paired real robot baseline/guided trials for TacQuality classifier/scorer guidance.",
        "This manifest is execution evidence bookkeeping, not a performance result.",
        "",
        "## Files",
        "",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        "",
        "## Protocol",
        "",
        "1. Start the matching server command from `for_show_xiaomi/guide_forshow.sh`.",
        "2. Run the `client_command` for each row in `trial_order`; each command forwards `pair_id` and manifest metadata to the server log.",
        "3. After every trial, confirm that a new server-side `force_trace.csv` exists under `expected_group_dir`.",
        "4. After all trials, run `apply_rollout_manifest_metadata.py` as a consistency/fallback pass for any missing metadata.",
        "5. For insertion, fill `success`, `stopped_early`, `bounce_count`, and `retry_count` in the manifest CSV, rerun metadata apply, then evaluate.",
        "6. Run `eval_tac_quality_real_rollouts.py`; final evidence uses explicit manifest `pair_id` pairing by default, and only non-synthetic paired logs count as real evidence.",
        "",
        "## Planned Trials",
        "",
        "| order | task | pair | group | port | arm | status |",
        "|---:|---|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['trial_order']} | `{row['task']}` | `{row['pair_id']}` | "
            f"`{row['group']}` | {row['server_port']} | `{row['server_arm']}` | "
            f"`{row['run_status']}` |"
        )
    lines += [
        "",
        "## Evaluation Commands",
        "",
        "```bash",
        "cd /home/chenshuai/Project/TactileACT-cs",
        "conda run --no-capture-output -n TactileACT python for_show_xiaomi/apply_rollout_manifest_metadata.py \\",
        f"  --manifest_csv {csv_path}",
        *eval_lines,
        "```",
        "",
        "## Notes",
        "",
        f"- tasks: `{args.tasks}`",
        f"- board_pairs: `{args.board_pairs}`",
        f"- insertion_pairs: `{args.insertion_pairs}`",
        f"- board_root: `{board_root}`",
        f"- board_guided_arm: `{board_guided_arm}`",
        f"- insertion_root: `{insertion_root}`",
        f"- insertion_guided_arm: `{insertion_guided_arm}`",
        f"- order: `{args.order}`",
        f"- client_host: `{args.client_host}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="current_forceaware_goodmargin_manifest")
    parser.add_argument("--tasks", default="board,insertion")
    parser.add_argument("--board_pairs", type=int, default=3)
    parser.add_argument("--insertion_pairs", type=int, default=3)
    parser.add_argument("--board_root", default=None)
    parser.add_argument("--board_baseline_port", type=int, default=None)
    parser.add_argument("--board_guided_port", type=int, default=None)
    parser.add_argument("--board_baseline_arm", default=None)
    parser.add_argument("--board_guided_arm", default=None)
    parser.add_argument("--board_pair_prefix", default=None)
    parser.add_argument("--insertion_root", default=None)
    parser.add_argument("--insertion_baseline_port", type=int, default=None)
    parser.add_argument("--insertion_guided_port", type=int, default=None)
    parser.add_argument("--insertion_baseline_arm", default=None)
    parser.add_argument("--insertion_guided_arm", default=None)
    parser.add_argument("--insertion_pair_prefix", default=None)
    parser.add_argument(
        "--order",
        choices=["interleaved", "guided_first_interleaved", "all_baseline_then_guided"],
        default="interleaved",
    )
    parser.add_argument("--client_host", default="${GPU_SERVER_IP}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tac_quality_rollout_manifest.csv"
    json_path = out_dir / "tac_quality_rollout_manifest.json"
    md_path = out_dir / "tac_quality_rollout_manifest.md"
    rows = make_rows(args, manifest_csv=str(csv_path))

    write_csv(rows, csv_path)
    summary = {
        "tag": args.tag,
        "tasks": parse_tasks(args.tasks),
        "board_pairs": int(args.board_pairs),
        "insertion_pairs": int(args.insertion_pairs),
        "order": args.order,
        "client_host": args.client_host,
        "n_trials": len(rows),
        "overrides": {
            "board_root": args.board_root,
            "board_baseline_port": args.board_baseline_port,
            "board_guided_port": args.board_guided_port,
            "board_baseline_arm": args.board_baseline_arm,
            "board_guided_arm": args.board_guided_arm,
            "board_pair_prefix": args.board_pair_prefix,
            "insertion_root": args.insertion_root,
            "insertion_baseline_port": args.insertion_baseline_port,
            "insertion_guided_port": args.insertion_guided_port,
            "insertion_baseline_arm": args.insertion_baseline_arm,
            "insertion_guided_arm": args.insertion_guided_arm,
            "insertion_pair_prefix": args.insertion_pair_prefix,
        },
        "rows": rows,
        "csv": str(csv_path),
        "markdown": str(md_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(rows, md_path, args, csv_path, json_path)

    print(json.dumps({
        "csv": str(csv_path),
        "json": str(json_path),
        "markdown": str(md_path),
        "n_trials": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
