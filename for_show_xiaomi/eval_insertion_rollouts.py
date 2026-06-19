#!/usr/bin/env python3
"""Summarize server-side insertion rollout logs.

The insertion server logs one directory per rollout:

  root/baseline/<trial>/force_trace.csv
  root/guided/<trial>/force_trace.csv

This evaluator is intentionally conservative.  Force/marker/action metrics can
be computed from the server-side CSV, but success/bounce/retry claims require
human or robot metadata in ``metadata.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ACTION_KEY_RE = re.compile(r"^action_\d+$")


def finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def stats(values) -> dict[str, float]:
    arr = finite(values)
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = sorted({k for row in rows for k in row})
    out: dict[str, np.ndarray] = {}
    for key in keys:
        vals = []
        ok = True
        for row in rows:
            try:
                vals.append(float(row.get(key, "nan")))
            except ValueError:
                ok = False
                break
        if ok:
            out[key] = np.asarray(vals, dtype=np.float64)
    return out


def load_metadata(trial_dir: Path) -> dict[str, Any]:
    path = trial_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_boolish(value: Any) -> float:
    if value is None:
        return float("nan")
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return float("nan")
    if text in {"1", "true", "yes", "y", "success", "succeeded", "pass"}:
        return 1.0
    if text in {"0", "false", "no", "n", "fail", "failed"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return float("nan")


def metadata_value(meta: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in meta:
            return meta[key]
    server_meta = meta.get("server_metadata")
    if isinstance(server_meta, dict):
        for key in keys:
            if key in server_meta:
                return server_meta[key]
    return None


def add_signal_stats(row: dict[str, Any], data: dict[str, np.ndarray], key: str) -> None:
    if key not in data:
        return
    s = stats(data[key])
    for stat_key, value in s.items():
        row[f"{key}_{stat_key}"] = value
    arr = finite(data[key])
    delta = np.abs(np.diff(arr))
    row[f"{key}_delta_abs_mean"] = float(delta.mean()) if len(delta) else 0.0
    row[f"{key}_delta_abs_p95"] = float(np.percentile(delta, 95)) if len(delta) else 0.0


def summarize_trace(csv_path: Path) -> dict[str, Any]:
    trial_dir = csv_path.parent
    data = read_csv(csv_path)
    meta = load_metadata(trial_dir)
    server_meta = meta.get("server_metadata") if isinstance(meta.get("server_metadata"), dict) else {}
    row: dict[str, Any] = {
        "trial_dir": str(trial_dir),
        "csv": str(csv_path),
        "group": infer_group(trial_dir),
        "pair_id": meta.get("pair_id"),
        "port": meta.get("port"),
        "trial": meta.get("trial"),
        "episode": meta.get("episode"),
        "steps": meta.get("steps"),
        "stop_reason": meta.get("stop_reason"),
        "server_arm": server_meta.get("arm"),
        "server_guidance": server_meta.get("guidance"),
    }
    row["success"] = parse_boolish(metadata_value(meta, "success", "task_success"))
    row["stopped_early"] = parse_boolish(metadata_value(meta, "stopped_early", "early_stop"))
    row["bounce_count"] = parse_boolish(metadata_value(meta, "bounce_count", "bounces"))
    row["retry_count"] = parse_boolish(metadata_value(meta, "retry_count", "retries"))

    for key in [
        "ft_fz",
        "ft_f_mag",
        "left_fz",
        "left_f_mag",
        "right_fz",
        "right_f_mag",
        "left_marker_mag_mean",
        "left_marker_mag_max",
        "right_marker_mag_mean",
        "right_marker_mag_max",
        "guidance_score_delta_mean",
        "guidance_accept_rate",
        "guidance_contact_gate_value",
    ]:
        add_signal_stats(row, data, key)

    action_keys = sorted(k for k in data if ACTION_KEY_RE.match(k))
    if action_keys:
        action = np.stack([data[k] for k in action_keys], axis=-1)
        if len(action) > 1:
            step = np.linalg.norm(np.diff(action, axis=0), axis=-1)
            accel = np.linalg.norm(np.diff(np.diff(action, axis=0), axis=0), axis=-1) if len(action) > 2 else np.asarray([])
            row["action_step_norm_mean"] = float(finite(step).mean()) if len(finite(step)) else 0.0
            row["action_step_norm_p95"] = float(np.percentile(finite(step), 95)) if len(finite(step)) else 0.0
            row["action_accel_norm_mean"] = float(finite(accel).mean()) if len(finite(accel)) else 0.0
            row["action_accel_norm_p95"] = float(np.percentile(finite(accel), 95)) if len(finite(accel)) else 0.0
    return row


def infer_group(trial_dir: Path) -> str:
    for part in reversed(trial_dir.parts):
        lowered = part.lower()
        if "baseline" in lowered or lowered in {"base", "no_guidance"}:
            return "baseline"
        if "guided" in lowered or lowered in {"ptg", "ptg_guided"}:
            return "guided"
    return "unknown"


def discover(root: Path) -> list[Path]:
    if root.is_file() and root.name == "force_trace.csv":
        return [root]
    return sorted(root.rglob("force_trace.csv"))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for row in rows for k in row})
    preferred = [
        "group",
        "pair_id",
        "trial_dir",
        "port",
        "steps",
        "stop_reason",
        "success",
        "stopped_early",
        "bounce_count",
        "retry_count",
        "ft_fz_mean",
        "ft_f_mag_mean",
        "left_f_mag_mean",
        "right_f_mag_mean",
        "left_marker_mag_mean_mean",
        "action_step_norm_mean",
        "action_accel_norm_mean",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=preferred + [k for k in keys if k not in preferred])
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    metric_keys = [
        "success",
        "stopped_early",
        "bounce_count",
        "retry_count",
        "ft_f_mag_mean",
        "ft_f_mag_p95",
        "ft_f_mag_delta_abs_mean",
        "left_f_mag_mean",
        "left_marker_mag_mean_mean",
        "left_marker_mag_mean_delta_abs_mean",
        "action_step_norm_mean",
        "action_accel_norm_mean",
        "guidance_score_delta_mean_mean",
        "guidance_accept_rate_mean",
    ]
    for name in sorted({row["group"] for row in rows}):
        subset = [row for row in rows if row["group"] == name]
        item: dict[str, Any] = {"n_trials": len(subset)}
        for key in metric_keys:
            vals = finite([row.get(key, np.nan) for row in subset])
            if len(vals):
                item[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                    "n": int(len(vals)),
                }
        out[name] = item
    return out


def check_expected_arms(
    rows: list[dict[str, Any]],
    *,
    expected_baseline_arm: str | None,
    expected_guided_arm: str | None,
) -> dict[str, Any]:
    expected = {
        "baseline": expected_baseline_arm,
        "guided": expected_guided_arm,
    }
    violations = []
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        group = str(row.get("group") or "unknown")
        arm = row.get("server_arm")
        arm_text = "" if arm is None else str(arm)
        counts.setdefault(group, {})
        counts[group][arm_text] = counts[group].get(arm_text, 0) + 1
        wanted = expected.get(group)
        if wanted and arm_text != wanted:
            violations.append({
                "trial_dir": row.get("trial_dir"),
                "group": group,
                "server_arm": arm,
                "expected_arm": wanted,
            })
    return {
        "expected_baseline_arm": expected_baseline_arm,
        "expected_guided_arm": expected_guided_arm,
        "arm_counts_by_group": counts,
        "violations": violations,
        "ok": not violations,
    }


def metadata_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "complete": False}
    keys = ["success", "stopped_early", "bounce_count", "retry_count"]
    coverage = {key: int(np.isfinite([row.get(key, np.nan) for row in rows]).sum()) for key in keys}
    complete = coverage["success"] == n and coverage["stopped_early"] == n
    return {
        "n": n,
        **coverage,
        "success_and_stopped_early_complete": bool(complete),
        "missing_examples": [
            row["trial_dir"]
            for row in rows
            if not (np.isfinite(row.get("success", np.nan)) and np.isfinite(row.get("stopped_early", np.nan)))
        ][:10],
    }


def write_group_csv(grouped_summary: dict[str, Any], path: Path) -> None:
    rows = []
    for name, item in grouped_summary.items():
        row = {"group": name, "n_trials": item.get("n_trials", 0)}
        for key, value in item.items():
            if isinstance(value, dict):
                for stat_key, stat_value in value.items():
                    row[f"{key}_{stat_key}"] = stat_value
        rows.append(row)
    keys = sorted({k for row in rows for k in row})
    preferred = ["group", "n_trials"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=preferred + [k for k in keys if k not in preferred])
        writer.writeheader()
        writer.writerows(rows)


def write_metadata_template(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a fill-in sheet for human/robot outcome labels.

    Server-side traces contain force, marker, action, and guidance signals, but
    they cannot know whether the insertion succeeded unless the tester records
    the outcome.  This template keeps that boundary explicit.
    """

    fields = [
        "trial_dir",
        "group",
        "pair_id",
        "success",
        "stopped_early",
        "bounce_count",
        "retry_count",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "trial_dir": row.get("trial_dir", ""),
                "group": row.get("group", ""),
                "pair_id": row.get("pair_id", ""),
                "success": "" if not np.isfinite(row.get("success", np.nan)) else row.get("success"),
                "stopped_early": "" if not np.isfinite(row.get("stopped_early", np.nan)) else row.get("stopped_early"),
                "bounce_count": "" if not np.isfinite(row.get("bounce_count", np.nan)) else row.get("bounce_count"),
                "retry_count": "" if not np.isfinite(row.get("retry_count", np.nan)) else row.get("retry_count"),
                "notes": "",
            })


def write_plot(rows: list[dict[str, Any]], traces: list[Path], path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not traces:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    for csv_path in traces:
        data = read_csv(csv_path)
        t = data.get("t")
        if t is None:
            continue
        group = infer_group(csv_path.parent)
        label = f"{group}/{csv_path.parent.name}"
        for ax, key, title in [
            (axes[0, 0], "ft_f_mag", "robot |Fxyz|"),
            (axes[0, 1], "left_f_mag", "left tactile |Fxyz|"),
            (axes[1, 0], "left_marker_mag_mean", "left marker mean"),
            (axes[1, 1], "action_step_norm", "action step norm"),
        ]:
            if key == "action_step_norm":
                action_keys = sorted(k for k in data if ACTION_KEY_RE.match(k))
                if not action_keys:
                    continue
                action = np.stack([data[k] for k in action_keys], axis=-1)
                y = np.r_[0.0, np.linalg.norm(np.diff(action, axis=0), axis=-1)] if len(action) > 1 else np.zeros(len(t))
            else:
                y = data.get(key)
            if y is not None and np.isfinite(y).any():
                ax.plot(t[: len(y)], y, linewidth=1.0, alpha=0.75, label=label)
            ax.set_title(title)
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time (s)")
        if len(traces) <= 12:
            ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Insertion Server Rollout Summary",
        "",
        f"- root: `{result['root']}`",
        f"- n_trials: `{result['n_trials']}`",
        f"- metadata success/stopped coverage: `{result['metadata_coverage']['success_and_stopped_early_complete']}`",
        f"- summary_csv: `{result['summary_csv']}`",
        f"- group_summary_csv: `{result['group_summary_csv']}`",
        f"- metadata_template_csv: `{result['metadata_template_csv']}`",
        f"- overview_plot: `{result.get('overview_plot')}`",
        "",
        "## Group Summary",
        "",
        "| group | n | success | stopped early | bounce count | retry count | force mean | marker mean | action step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["group_summary"].items():
        lines.append(
            f"| `{name}` | {item.get('n_trials', 0)} | "
            f"{item.get('success', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('stopped_early', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('bounce_count', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('retry_count', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_f_mag_mean', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('left_marker_mag_mean_mean', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('action_step_norm_mean', {}).get('mean', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## Evidence Boundary",
        "",
        "This report can summarize server-side force/marker/action traces directly.",
        "Success, bounce, and retry claims are valid only when metadata coverage is complete.",
        "If metadata coverage is false, fill `metadata_template_csv` and copy the labels into each trial's `metadata.json` before using outcome metrics.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/chenshuai/Project/output/insertion_rollouts/default_insertion_risk_scorer")
    parser.add_argument("--output_dir", default="/home/chenshuai/Project/output/insertion_rollout_eval")
    parser.add_argument("--tag", default="insertion_default_risk_scorer")
    parser.add_argument("--require_metadata", action="store_true",
                        help="Exit nonzero unless success and stopped_early metadata exist for all trials.")
    parser.add_argument("--expected_baseline_arm", default=None,
                        help="If set, fail when any baseline-group trace was logged by a different server arm.")
    parser.add_argument("--expected_guided_arm", default=None,
                        help="If set, fail when any guided-group trace was logged by a different server arm.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    traces = discover(root)
    if not traces:
        raise FileNotFoundError(f"No force_trace.csv found under {root}")
    rows = [summarize_trace(path) for path in traces]
    arm_check = check_expected_arms(
        rows,
        expected_baseline_arm=args.expected_baseline_arm,
        expected_guided_arm=args.expected_guided_arm,
    )
    if not arm_check["ok"]:
        raise RuntimeError(
            "Unexpected insertion rollout arm(s) detected; refusing to mix scorer variants in one evaluation: "
            + json.dumps(arm_check["violations"], ensure_ascii=False)
        )
    out_dir = Path(args.output_dir) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "insertion_rollout_summary.csv"
    group_csv = out_dir / "insertion_rollout_group_summary.csv"
    metadata_template_csv = out_dir / "metadata_template.csv"
    write_csv(rows, summary_csv)
    group_summary = grouped(rows)
    write_group_csv(group_summary, group_csv)
    write_metadata_template(rows, metadata_template_csv)
    result = {
        "root": str(root),
        "n_trials": len(rows),
        "traces": [str(p) for p in traces],
        "summary_csv": str(summary_csv),
        "group_summary_csv": str(group_csv),
        "metadata_template_csv": str(metadata_template_csv),
        "group_summary": group_summary,
        "metadata_coverage": metadata_coverage(rows),
        "expected_arm_check": arm_check,
        "rows": rows,
    }
    result["overview_plot"] = write_plot(rows, traces, out_dir / "insertion_rollout_overview.png")
    result_path = out_dir / "insertion_rollout_summary.json"
    md_path = out_dir / "insertion_rollout_summary.md"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, md_path)
    if args.require_metadata and not result["metadata_coverage"]["success_and_stopped_early_complete"]:
        raise SystemExit(
            "Missing success/stopped_early metadata. Fill metadata.json for each trial before claiming outcome."
        )
    print(json.dumps({
        "output": str(result_path),
        "n_trials": len(rows),
        "metadata_complete": result["metadata_coverage"]["success_and_stopped_early_complete"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
