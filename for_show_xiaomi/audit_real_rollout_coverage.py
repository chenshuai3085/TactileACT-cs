#!/usr/bin/env python3
"""Audit real rollout coverage for TacQuality baseline/guided evidence.

This is a read-only bookkeeping check.  It answers:

1. Which planned manifest rows already have server-side force_trace.csv?
2. Are baseline/guided pair IDs complete for board and insertion?
3. Are insertion outcome labels complete enough for final evaluation?
4. Can eval_tac_quality_real_rollouts.py be treated as real evidence yet?
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST_CSV = (
    "/home/chenshuai/Project/output/tac_quality_real_rollout_manifest/"
    "current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv"
)
DEFAULT_OUTPUT_DIR = "/home/chenshuai/Project/output/tac_quality_real_rollout_coverage"
DEFAULT_TAG = "current_s12_good_margin_coverage"
OUTCOME_KEYS = ["success", "stopped_early", "bounce_count", "retry_count"]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_sort_key(row: dict[str, str]) -> tuple[int, str]:
    try:
        return int(float(row.get("trial_order") or 0)), row.get("pair_id") or ""
    except ValueError:
        return 0, row.get("pair_id") or ""


def has_value(value: Any) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"", "nan", "none", "null", "n/a"}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def trial_candidates(row: dict[str, str]) -> list[Path]:
    pattern = (row.get("expected_trial_dir_pattern") or "").strip()
    if pattern:
        candidates = [Path(p) for p in glob.glob(pattern)]
    else:
        group_dir = Path(row.get("expected_group_dir") or "")
        port = str(row.get("server_port") or "").strip()
        candidates = [Path(p) for p in glob.glob(str(group_dir / f"*_port{port}_episode*"))]
    return sorted(p for p in candidates if (p / "force_trace.csv").exists())


def metadata_status(row: dict[str, str], trial_dir: Path | None) -> dict[str, Any]:
    if trial_dir is None:
        return {
            "metadata_exists": False,
            "metadata_path": None,
            "task_match": None,
            "arm_match": None,
            "port_match": None,
            "pair_id_match": None,
            "synthetic": None,
            "outcome_complete": False,
        }
    path = trial_dir / "metadata.json"
    meta = load_json(path)
    server_meta = meta.get("server_metadata") if isinstance(meta.get("server_metadata"), dict) else {}
    task_actual = str(meta.get("task") or server_meta.get("task") or "")
    arm_actual = str(meta.get("arm") or server_meta.get("arm") or "")
    port_actual = str(meta.get("port") or server_meta.get("port") or "")
    pair_actual = str(meta.get("pair_id") or "")
    synthetic = bool(
        str(meta.get("synthetic_smoke", "")).lower() in {"1", "true", "yes"}
        or str(meta.get("not_real_robot_evidence", "")).lower() in {"1", "true", "yes"}
        or str(meta.get("log_side", "")).lower() == "synthetic"
        or "synthetic" in str(server_meta.get("protocol", "")).lower()
        or "synthetic" in str(trial_dir).lower()
    )
    outcomes = {key: meta.get(key) for key in OUTCOME_KEYS}
    manifest_outcomes = {key: row.get(key) for key in OUTCOME_KEYS}
    outcome_complete = all(has_value(outcomes.get(key)) or has_value(manifest_outcomes.get(key)) for key in OUTCOME_KEYS)
    return {
        "metadata_exists": path.exists(),
        "metadata_path": str(path),
        "task_actual": task_actual,
        "arm_actual": arm_actual,
        "port_actual": port_actual,
        "pair_id_actual": pair_actual,
        "task_match": (not task_actual) or task_actual == str(row.get("task")),
        "arm_match": (not arm_actual) or arm_actual == str(row.get("server_arm")),
        "port_match": (not port_actual) or port_actual == str(row.get("server_port")),
        "pair_id_match": (not pair_actual) or pair_actual == str(row.get("pair_id")),
        "synthetic": synthetic,
        "outcome_complete": outcome_complete,
        "metadata_outcomes": outcomes,
        "manifest_outcomes": manifest_outcomes,
    }


def audit_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    audited = []
    used: set[Path] = set()
    for row in sorted(rows, key=row_sort_key):
        candidates = [p for p in trial_candidates(row) if p not in used]
        status = "missing"
        trial_dir: Path | None = None
        if len(candidates) == 1:
            status = "matched"
            trial_dir = candidates[0]
            used.add(trial_dir)
        elif len(candidates) > 1:
            status = "ambiguous"
        meta = metadata_status(row, trial_dir)
        if status == "matched":
            metadata_checks = [
                meta["metadata_exists"],
                meta["task_match"],
                meta["arm_match"],
                meta["port_match"],
                meta["pair_id_match"],
                not meta["synthetic"],
            ]
            if not all(item is True for item in metadata_checks):
                status = "metadata_problem"
        audited.append({
            "trial_order": row.get("trial_order"),
            "task": row.get("task"),
            "pair_id": row.get("pair_id"),
            "group": row.get("group"),
            "server_port": row.get("server_port"),
            "server_arm": row.get("server_arm"),
            "expected_group_dir": row.get("expected_group_dir"),
            "expected_trial_dir_pattern": row.get("expected_trial_dir_pattern"),
            "client_command": row.get("client_command"),
            "status": status,
            "candidate_count": len(candidates),
            "candidates": [str(p) for p in candidates[:10]],
            "trial_dir": str(trial_dir) if trial_dir else None,
            "force_trace": str(trial_dir / "force_trace.csv") if trial_dir else None,
            "metadata": meta,
        })
    return audited


def summarize(audited: list[dict[str, Any]], min_pairs: int) -> dict[str, Any]:
    planned: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    observed: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    statuses: dict[str, int] = defaultdict(int)
    pairs: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    insertion_outcome_complete_by_pair: dict[str, bool] = defaultdict(lambda: True)

    for row in audited:
        task = str(row["task"])
        group = str(row["group"])
        pair_id = str(row["pair_id"])
        planned[task][group] += 1
        statuses[str(row["status"])] += 1
        if row["status"] == "matched":
            observed[task][group] += 1
            pairs[task][pair_id][group] = True
            if task == "insertion":
                insertion_outcome_complete_by_pair[pair_id] = (
                    insertion_outcome_complete_by_pair[pair_id]
                    and bool(row["metadata"].get("outcome_complete"))
                )
        else:
            pairs[task][pair_id][group] = False
            if task == "insertion":
                insertion_outcome_complete_by_pair[pair_id] = False

    pair_summary: dict[str, Any] = {}
    for task, task_pairs in pairs.items():
        complete_ids = []
        incomplete_ids = []
        for pair_id, groups in sorted(task_pairs.items()):
            complete = bool(groups.get("baseline")) and bool(groups.get("guided"))
            if task == "insertion":
                complete = complete and bool(insertion_outcome_complete_by_pair.get(pair_id))
            (complete_ids if complete else incomplete_ids).append(pair_id)
        pair_summary[task] = {
            "complete_pair_ids": complete_ids,
            "incomplete_pair_ids": incomplete_ids,
            "n_complete_pairs": len(complete_ids),
            "min_pairs": int(min_pairs),
            "ready": len(complete_ids) >= int(min_pairs),
            "insertion_requires_outcome_metadata": task == "insertion",
        }

    board_ready = bool(pair_summary.get("board", {}).get("ready", False))
    insertion_ready = bool(pair_summary.get("insertion", {}).get("ready", False))
    return {
        "planned_counts": {task: dict(groups) for task, groups in planned.items()},
        "observed_counts": {task: dict(groups) for task, groups in observed.items()},
        "status_counts": dict(statuses),
        "pair_summary": pair_summary,
        "board_ready_for_real_eval": board_ready,
        "insertion_ready_for_real_eval": insertion_ready,
        "real_rollout_evidence_complete": board_ready and insertion_ready,
        "note": "Insertion pairs require both baseline/guided force traces and complete outcome metadata.",
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# TacQuality Real Rollout Coverage Audit",
        "",
        "This is a read-only coverage audit for real robot baseline/guided evidence.",
        "",
        "## Summary",
        "",
        f"- manifest_csv: `{result['manifest_csv']}`",
        f"- board_ready_for_real_eval: `{result['summary']['board_ready_for_real_eval']}`",
        f"- insertion_ready_for_real_eval: `{result['summary']['insertion_ready_for_real_eval']}`",
        f"- real_rollout_evidence_complete: `{result['summary']['real_rollout_evidence_complete']}`",
        f"- status_counts: `{json.dumps(result['summary']['status_counts'], ensure_ascii=False)}`",
        "",
        "## Counts",
        "",
        f"- planned_counts: `{json.dumps(result['summary']['planned_counts'], ensure_ascii=False)}`",
        f"- observed_counts: `{json.dumps(result['summary']['observed_counts'], ensure_ascii=False)}`",
        "",
        "## Pair Coverage",
        "",
        "| task | complete pairs | incomplete pairs | ready |",
        "|---|---:|---:|---:|",
    ]
    for task, item in result["summary"]["pair_summary"].items():
        lines.append(
            f"| `{task}` | `{item['n_complete_pairs']}` | "
            f"`{len(item['incomplete_pair_ids'])}` | `{item['ready']}` |"
        )
    lines.extend([
        "",
        "## Missing Or Problem Rows",
        "",
        "| order | task | pair | group | port | arm | status | candidates |",
        "|---:|---|---|---|---:|---|---|---:|",
    ])
    problem_rows = [row for row in result["rows"] if row["status"] != "matched"]
    for row in problem_rows:
        lines.append(
            f"| {row['trial_order']} | `{row['task']}` | `{row['pair_id']}` | "
            f"`{row['group']}` | {row['server_port']} | `{row['server_arm']}` | "
            f"`{row['status']}` | {row['candidate_count']} |"
        )
    if problem_rows:
        lines.extend([
            "",
            "## Commands To Run For Missing Or Problem Rows",
            "",
            "Run these client commands on the robot/client machine after starting the matching server blocks.",
            "They are copied from the manifest so they include pair IDs and evaluation metadata.",
            "",
        ])
        for row in problem_rows:
            command = str(row.get("client_command") or "").strip()
            if not command:
                continue
            lines.extend([
                f"### Trial {row['trial_order']}: {row['task']} {row['group']} {row['pair_id']}",
                "",
                "```bash",
                command,
                "```",
                "",
                "Expected server-side output pattern:",
                "",
                "```text",
                str(row.get("expected_trial_dir_pattern") or row.get("expected_group_dir") or ""),
                "```",
                "",
            ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- `missing` means no server-side `force_trace.csv` matched the planned manifest row.",
        "- `ambiguous` means multiple unmatched trial dirs matched one manifest row.",
        "- `metadata_problem` means a matched trial exists but task/arm/port/pair_id/synthetic checks failed.",
        "- The command section is operational help only; it does not change coverage status.",
        "- Final TacQuality performance evidence requires non-synthetic complete pairs for both board and insertion.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", default=DEFAULT_MANIFEST_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--min_pairs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest_csv).expanduser()
    rows = read_manifest(manifest)
    audited = audit_rows(rows)
    summary = summarize(audited, min_pairs=args.min_pairs)
    out_dir = Path(args.output_dir).expanduser() / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "tac_quality_real_rollout_coverage.json"
    md_path = out_dir / "tac_quality_real_rollout_coverage.md"
    result = {
        "manifest_csv": str(manifest),
        "output_json": str(json_path),
        "output_markdown": str(md_path),
        "min_pairs": int(args.min_pairs),
        "summary": summary,
        "rows": audited,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "board_ready_for_real_eval": summary["board_ready_for_real_eval"],
        "insertion_ready_for_real_eval": summary["insertion_ready_for_real_eval"],
        "real_rollout_evidence_complete": summary["real_rollout_evidence_complete"],
        "status_counts": summary["status_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
