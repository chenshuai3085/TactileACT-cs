#!/usr/bin/env python3
"""Apply TacQuality rollout manifest pair IDs to server-side trial metadata."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


OUTCOME_KEYS = ["success", "stopped_early", "bounce_count", "retry_count", "notes"]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "n/a"}:
        return None
    if text in {"1", "true", "yes", "y", "success", "succeeded", "pass"}:
        return True
    if text in {"0", "false", "no", "n", "fail", "failed"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value!r}")


def parse_count(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null", "n/a"}:
        return None
    return int(float(text))


def row_outcomes(row: dict[str, str]) -> dict[str, Any]:
    outcomes: dict[str, Any] = {}
    success = parse_bool(row.get("success"))
    stopped = parse_bool(row.get("stopped_early"))
    bounce = parse_count(row.get("bounce_count"))
    retry = parse_count(row.get("retry_count"))
    notes = (row.get("notes") or "").strip()
    if success is not None:
        outcomes["success"] = success
        outcomes["task_success"] = success
    if stopped is not None:
        outcomes["stopped_early"] = stopped
        outcomes["early_stop"] = stopped
    if bounce is not None:
        outcomes["bounce_count"] = bounce
        outcomes["bounces"] = bounce
    if retry is not None:
        outcomes["retry_count"] = retry
        outcomes["retries"] = retry
    if notes:
        outcomes["notes"] = notes
    return outcomes


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def trial_dirs_for_row(row: dict[str, str]) -> list[Path]:
    pattern = (row.get("expected_trial_dir_pattern") or "").strip()
    if pattern:
        return sorted(Path(p).expanduser() for p in glob.glob(pattern))
    group_dir = (row.get("expected_group_dir") or "").strip()
    port = str(row.get("server_port") or "").strip()
    if not group_dir:
        return []
    fallback = str(Path(group_dir).expanduser() / f"*_port{port}_episode*")
    return sorted(Path(p) for p in glob.glob(fallback))


def has_force_trace(trial_dir: Path) -> bool:
    return (trial_dir / "force_trace.csv").exists()


def row_sort_key(row: dict[str, str]) -> int:
    try:
        return int(float(row.get("trial_order") or 0))
    except ValueError:
        return 0


def validate_manifest(rows: list[dict[str, str]]) -> list[str]:
    errors = []
    required = ["trial_order", "task", "pair_id", "group", "server_port", "server_arm"]
    seen_orders = set()
    for idx, row in enumerate(rows):
        for key in required:
            if not str(row.get(key) or "").strip():
                errors.append(f"row {idx}: missing {key}")
        order = str(row.get("trial_order") or "").strip()
        if order in seen_orders:
            errors.append(f"duplicate trial_order: {order}")
        seen_orders.add(order)
    return errors


def apply_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest = Path(args.manifest_csv).expanduser()
    rows = sorted(read_manifest(manifest), key=row_sort_key)
    errors = validate_manifest(rows)
    if errors:
        raise SystemExit("; ".join(errors))

    used: set[Path] = set()
    results: list[dict[str, Any]] = []
    for row in rows:
        candidates = [p for p in trial_dirs_for_row(row) if has_force_trace(p)]
        candidates = [p for p in candidates if p not in used]
        candidates = sorted(candidates, key=lambda p: str(p))
        result: dict[str, Any] = {
            "trial_order": row.get("trial_order"),
            "task": row.get("task"),
            "pair_id": row.get("pair_id"),
            "group": row.get("group"),
            "server_port": row.get("server_port"),
            "server_arm": row.get("server_arm"),
            "candidate_count": len(candidates),
        }
        if not candidates:
            result.update({"status": "missing", "reason": "no unmatched force_trace.csv for manifest row"})
            results.append(result)
            continue
        if len(candidates) > 1 and args.strict_unique_candidate:
            result.update({
                "status": "ambiguous",
                "reason": "multiple unmatched trial dirs match manifest row",
                "candidates": [str(p) for p in candidates],
            })
            results.append(result)
            continue

        trial_dir = candidates[0]
        used.add(trial_dir)
        meta_path = trial_dir / "metadata.json"
        meta = load_json(meta_path)
        server_meta = meta.get("server_metadata") if isinstance(meta.get("server_metadata"), dict) else {}
        arm = str(server_meta.get("arm") or meta.get("arm") or "")
        port = str(meta.get("port") or "")
        task = str(meta.get("task") or server_meta.get("task") or "")
        violations = []
        if arm and arm != str(row.get("server_arm")):
            violations.append({"field": "server_arm", "expected": row.get("server_arm"), "actual": arm})
        if port and port != str(row.get("server_port")):
            violations.append({"field": "server_port", "expected": row.get("server_port"), "actual": port})
        if task and task != str(row.get("task")):
            violations.append({"field": "task", "expected": row.get("task"), "actual": task})
        if violations and not args.allow_metadata_mismatch:
            result.update({
                "status": "metadata_mismatch",
                "trial_dir": str(trial_dir),
                "metadata": str(meta_path),
                "violations": violations,
            })
            results.append(result)
            continue

        outcomes = row_outcomes(row)
        before = {key: meta.get(key) for key in ["pair_id", "manifest_trial_order", *OUTCOME_KEYS] if key in meta}
        meta.update({
            "pair_id": row.get("pair_id"),
            "manifest_trial_order": int(float(row.get("trial_order") or 0)),
            "manifest_group": row.get("group"),
            "manifest_task": row.get("task"),
            "manifest_server_port": int(float(row.get("server_port") or 0)),
            "manifest_server_arm": row.get("server_arm"),
            "manifest_source_csv": str(manifest),
            "pair_id_source": "tac_quality_rollout_manifest",
        })
        if outcomes:
            meta.update(outcomes)
            meta["outcome_metadata_source"] = "tac_quality_rollout_manifest"
            meta["outcome_metadata_complete"] = bool(
                "success" in outcomes and "stopped_early" in outcomes
            )
        if not args.dry_run:
            write_json(meta_path, meta)
        result.update({
            "status": "would_update" if args.dry_run else "updated",
            "trial_dir": str(trial_dir),
            "metadata": str(meta_path),
            "before": before,
            "applied_pair_id": row.get("pair_id"),
            "applied_outcomes": outcomes,
            "violations_allowed": violations,
        })
        results.append(result)

    missing = [r for r in results if r["status"] == "missing"]
    ambiguous = [r for r in results if r["status"] == "ambiguous"]
    mismatched = [r for r in results if r["status"] == "metadata_mismatch"]
    summary = {
        "manifest_csv": str(manifest),
        "dry_run": bool(args.dry_run),
        "allow_missing": bool(args.allow_missing),
        "strict_unique_candidate": bool(args.strict_unique_candidate),
        "allow_metadata_mismatch": bool(args.allow_metadata_mismatch),
        "n_manifest_rows": len(rows),
        "updated": sum(1 for r in results if r["status"] == "updated"),
        "would_update": sum(1 for r in results if r["status"] == "would_update"),
        "missing": len(missing),
        "ambiguous": len(ambiguous),
        "metadata_mismatch": len(mismatched),
        "ok": not ambiguous and not mismatched and (not missing or args.allow_missing),
        "results": results,
    }
    out_path = Path(args.output_json).expanduser() if args.output_json else manifest.with_name("manifest_metadata_apply_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["output_json"] = str(out_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_missing", action="store_true",
                        help="Allow manifest rows without a matching force_trace.csv.")
    parser.add_argument("--strict_unique_candidate", action="store_true",
                        help="Fail if more than one unmatched trial dir matches a manifest row. By default, rows are bound to the earliest unmatched trial in manifest order.")
    parser.add_argument("--allow_metadata_mismatch", action="store_true",
                        help="Write metadata even when logged task/arm/port disagree with the manifest row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_manifest(args)
    print(json.dumps({
        "output_json": summary["output_json"],
        "ok": summary["ok"],
        "updated": summary["updated"],
        "would_update": summary["would_update"],
        "missing": summary["missing"],
        "ambiguous": summary["ambiguous"],
        "metadata_mismatch": summary["metadata_mismatch"],
    }, ensure_ascii=False, indent=2))
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
