#!/usr/bin/env python3
"""Summarize force traces recorded by ws_client.py during board wipe tests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def finite(values):
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
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keys = sorted({k for row in rows for k in row})
    out = {}
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


def summarize_trace(csv_path: Path) -> dict[str, Any]:
    trial_dir = csv_path.parent
    data = read_csv(csv_path)
    meta = load_metadata(trial_dir)
    row: dict[str, Any] = {
        "trial_dir": str(trial_dir),
        "csv": str(csv_path),
        "port": meta.get("port"),
        "trial": meta.get("trial"),
        "steps": meta.get("steps"),
        "stop_reason": meta.get("stop_reason"),
        "server_protocol": (meta.get("server_metadata") or {}).get("protocol"),
        "server_arm": (meta.get("server_metadata") or {}).get("arm"),
        "server_guidance": (meta.get("server_metadata") or {}).get("guidance"),
    }
    for key in ("ft_fz", "ft_f_mag", "left_fz", "left_f_mag", "right_fz", "right_f_mag"):
        if key not in data:
            continue
        s = stats(data[key])
        for stat_key, value in s.items():
            row[f"{key}_{stat_key}"] = value
        delta = np.abs(np.diff(finite(data[key])))
        row[f"{key}_delta_abs_mean"] = float(delta.mean()) if len(delta) else 0.0
        row[f"{key}_delta_abs_p95"] = float(np.percentile(delta, 95)) if len(delta) else 0.0
    return row


def discover(root: Path):
    if root.is_file() and root.name == "force_trace.csv":
        return [root]
    return sorted(root.rglob("force_trace.csv"))


def write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for row in rows for k in row})
    preferred = [
        "trial_dir", "port", "trial", "steps", "stop_reason",
        "server_protocol", "server_arm", "server_guidance",
        "ft_fz_mean", "ft_fz_std", "ft_fz_p95", "ft_fz_delta_abs_mean",
        "ft_f_mag_mean", "ft_f_mag_p95", "ft_f_mag_delta_abs_mean",
        "left_fz_mean", "left_f_mag_mean", "right_fz_mean", "right_f_mag_mean",
    ]
    fieldnames = preferred + [k for k in keys if k not in preferred]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_overview_plot(rows: list[dict[str, Any]], traces: list[Path], path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    n = len(traces)
    if n == 0:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for csv_path in traces:
        data = read_csv(csv_path)
        t = data.get("t")
        if t is None:
            continue
        label = csv_path.parent.name
        for ax, key in [(axes[0], "ft_fz"), (axes[1], "ft_f_mag")]:
            y = data.get(key)
            if y is not None and np.isfinite(y).any():
                ax.plot(t, y, linewidth=1.0, alpha=0.75, label=label)
    axes[0].set_ylabel("robot ft Fz")
    axes[1].set_ylabel("robot |Fxyz|")
    axes[1].set_xlabel("time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        if n <= 12:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Board Force Rollout Summary",
        "",
        f"- root: `{result['root']}`",
        f"- n_trials: `{result['n_trials']}`",
        f"- summary_csv: `{result['summary_csv']}`",
        f"- overview_plot: `{result.get('overview_plot')}`",
        "",
        "| trial | port | steps | stop | Fz mean | Fz p95 | |F| mean | dF mean |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{Path(row['trial_dir']).name}` | {row.get('port')} | {row.get('steps')} | "
            f"{row.get('stop_reason')} | {row.get('ft_fz_mean', float('nan')):.4f} | "
            f"{row.get('ft_fz_p95', float('nan')):.4f} | "
            f"{row.get('ft_f_mag_mean', float('nan')):.4f} | "
            f"{row.get('ft_fz_delta_abs_mean', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/chenshuai/Project/output/board_force_rollouts")
    parser.add_argument("--output_dir", default="/home/chenshuai/Project/output/board_force_rollout_eval")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    traces = discover(root)
    if not traces:
        raise FileNotFoundError(f"No force_trace.csv files under {root}")
    rows = [summarize_trace(p) for p in traces]
    tag = args.tag or root.name
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "board_force_rollout_summary.csv"
    summary_json = out_dir / "board_force_rollout_summary.json"
    summary_md = out_dir / "board_force_rollout_summary.md"
    overview_png = out_dir / "board_force_overview.png"
    write_summary_csv(rows, summary_csv)
    overview = write_overview_plot(rows, traces, overview_png)
    result = {
        "root": str(root),
        "n_trials": len(rows),
        "summary_csv": str(summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "overview_plot": overview,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, summary_md)
    print(json.dumps({k: result[k] for k in ["n_trials", "summary_csv", "summary_md", "overview_plot"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
