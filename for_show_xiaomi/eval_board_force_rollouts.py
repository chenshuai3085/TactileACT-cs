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


def robust_contact_mask(
    data: dict[str, np.ndarray],
    *,
    source: str = "auto",
    threshold_frac: float = 0.25,
    min_contact_fraction: float = 0.05,
) -> tuple[np.ndarray | None, str | None, float]:
    """Infer the wiping/contact segment from marker or force traces.

    Approach has low/no contact, so whole-episode force statistics can dilute
    the actual wiping quality.  This mask keeps evaluation aligned with the
    scorer target: force should be in range and smooth during contact.
    """

    preferred = [
        "left_marker_mag_mean",
        "right_marker_mag_mean",
        "ft_f_mag",
        "left_f_mag",
        "right_f_mag",
    ]
    candidates = preferred if source == "auto" else [source]
    for key in candidates:
        signal = data.get(key)
        if signal is None:
            continue
        arr = np.asarray(signal, dtype=np.float64)
        finite_mask = np.isfinite(arr)
        vals = arr[finite_mask]
        if len(vals) < 3:
            continue
        lo = float(np.percentile(vals, 10))
        hi = float(np.percentile(vals, 90))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
            continue
        threshold = lo + float(threshold_frac) * (hi - lo)
        mask = finite_mask & (arr >= threshold)
        frac = float(mask.mean()) if len(mask) else 0.0
        if frac >= min_contact_fraction:
            return mask, key, threshold
    return None, None, float("nan")


def add_signal_stats(
    row: dict[str, Any],
    data: dict[str, np.ndarray],
    key: str,
    *,
    prefix: str | None = None,
    mask: np.ndarray | None = None,
) -> None:
    if key not in data:
        return
    values = data[key]
    if mask is not None and len(mask) == len(values):
        values = values[mask]
    s = stats(values)
    name = prefix or key
    for stat_key, value in s.items():
        row[f"{name}_{stat_key}"] = value
    delta = np.abs(np.diff(finite(values)))
    row[f"{name}_delta_abs_mean"] = float(delta.mean()) if len(delta) else 0.0
    row[f"{name}_delta_abs_p95"] = float(np.percentile(delta, 95)) if len(delta) else 0.0


def summarize_trace(
    csv_path: Path,
    *,
    contact_source: str = "auto",
    contact_threshold_frac: float = 0.25,
    min_contact_fraction: float = 0.05,
) -> dict[str, Any]:
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
    metric_keys = (
        "ft_fz", "ft_f_mag",
        "left_fz", "left_f_mag",
        "right_fz", "right_f_mag",
        "left_marker_mag_mean", "left_marker_mag_max", "left_marker_contact_area",
        "right_marker_mag_mean", "right_marker_mag_max", "right_marker_contact_area",
    )
    for key in metric_keys:
        add_signal_stats(row, data, key)

    contact_mask, contact_key, contact_threshold = robust_contact_mask(
        data,
        source=contact_source,
        threshold_frac=contact_threshold_frac,
        min_contact_fraction=min_contact_fraction,
    )
    row["contact_source"] = contact_key
    row["contact_threshold"] = contact_threshold
    row["contact_steps"] = int(contact_mask.sum()) if contact_mask is not None else 0
    row["contact_fraction"] = float(contact_mask.mean()) if contact_mask is not None and len(contact_mask) else 0.0
    if contact_mask is not None:
        for key in metric_keys:
            add_signal_stats(row, data, key, prefix=f"{key}_contact", mask=contact_mask)
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


def normalize_group_name(value: Any) -> str | None:
    if value is None:
        return None
    lowered = str(value).lower()
    if lowered in {"baseline", "base", "no_guidance"} or "baseline" in lowered:
        return "baseline"
    if lowered in {"guided", "default_guided", "ptg", "ptg_guided"} or "guided" in lowered:
        return "guided"
    return str(value)


def group_key(row: dict[str, Any]) -> str:
    arm = row.get("server_arm")
    arm_group = normalize_group_name(arm)
    if arm_group:
        return arm_group
    trial_dir = Path(str(row.get("trial_dir", "")))
    for part in reversed(trial_dir.parts):
        part_group = normalize_group_name(part)
        if part_group in {"baseline", "guided"}:
            return part_group
    port = row.get("port")
    if port is not None:
        return f"port_{port}"
    return "unknown"


def grouped_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({group_key(row) for row in rows})
    out: dict[str, Any] = {}
    metric_keys = [
        "ft_fz_mean",
        "ft_fz_std",
        "ft_fz_p95",
        "ft_fz_delta_abs_mean",
        "ft_f_mag_mean",
        "ft_f_mag_p95",
        "ft_f_mag_delta_abs_mean",
        "ft_fz_contact_mean",
        "ft_fz_contact_p95",
        "ft_fz_contact_delta_abs_mean",
        "ft_f_mag_contact_mean",
        "ft_f_mag_contact_p95",
        "ft_f_mag_contact_delta_abs_mean",
        "left_fz_mean",
        "left_f_mag_mean",
        "right_fz_mean",
        "right_f_mag_mean",
        "left_marker_mag_mean_mean",
        "left_marker_contact_area_mean",
        "left_marker_mag_mean_contact_mean",
        "left_marker_contact_area_contact_mean",
        "contact_fraction",
    ]
    for name in groups:
        subset = [row for row in rows if group_key(row) == name]
        item: dict[str, Any] = {"n_trials": len(subset)}
        for key in metric_keys:
            vals = finite([row.get(key, np.nan) for row in subset])
            if len(vals):
                item[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p50": float(np.percentile(vals, 50)),
                    "p95": float(np.percentile(vals, 95)),
                }
        out[name] = item
    return out


def write_group_summary_csv(grouped: dict[str, Any], path: Path) -> None:
    rows = []
    for name, item in grouped.items():
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


def write_group_curves_plot(rows: list[dict[str, Any]], traces: list[Path], path: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not traces:
        return None

    row_by_csv = {Path(row["csv"]): row for row in rows}
    groups = ["baseline", "guided"]
    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 4.5 * len(groups)), sharex=False)
    if len(groups) == 1:
        axes = np.asarray([axes])

    for gi, group in enumerate(groups):
        group_traces = [p for p in traces if group_key(row_by_csv.get(p, {})) == group]
        for csv_path in group_traces:
            data = read_csv(csv_path)
            t = data.get("t")
            if t is None:
                continue
            label = csv_path.parent.name
            fz = data.get("ft_fz")
            fmag = data.get("ft_f_mag")
            if fz is not None and np.isfinite(fz).any():
                axes[gi, 0].plot(t, fz, linewidth=1.0, alpha=0.75, label=label)
            if fmag is not None and np.isfinite(fmag).any():
                axes[gi, 1].plot(t, fmag, linewidth=1.0, alpha=0.75, label=label)
        axes[gi, 0].set_title(f"{group}: robot ft Fz ({len(group_traces)} traces)")
        axes[gi, 1].set_title(f"{group}: robot |Fxyz| ({len(group_traces)} traces)")
        axes[gi, 0].set_ylabel("Fz")
        axes[gi, 1].set_ylabel("|Fxyz|")
        for ax in axes[gi]:
            ax.set_xlabel("time (s)")
            ax.grid(True, alpha=0.3)
            if 0 < len(group_traces) <= 10:
                ax.legend(loc="best", fontsize=7)

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
        f"- group_summary_csv: `{result['group_summary_csv']}`",
        f"- overview_plot: `{result.get('overview_plot')}`",
        f"- group_curves_plot: `{result.get('group_curves_plot')}`",
        "",
        "## Group Summary",
        "",
        "| group | n | Fz mean | Fz p95 | |F| mean | dF mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["group_summary"].items():
        lines.append(
            f"| `{name}` | {item.get('n_trials', 0)} | "
            f"{item.get('ft_fz_mean', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_fz_p95', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_f_mag_mean', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_fz_delta_abs_mean', {}).get('mean', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## Contact-Phase Group Summary",
        "",
        "Contact phase is inferred from marker magnitude when available, otherwise from force magnitude. This avoids judging approach/lift as wiping contact.",
        "",
        "| group | n | contact frac | contact Fz mean | contact Fz p95 | contact |F| mean | contact dF mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, item in result["group_summary"].items():
        lines.append(
            f"| `{name}` | {item.get('n_trials', 0)} | "
            f"{item.get('contact_fraction', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_fz_contact_mean', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_fz_contact_p95', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_f_mag_contact_mean', {}).get('mean', float('nan')):.4f} | "
            f"{item.get('ft_fz_contact_delta_abs_mean', {}).get('mean', float('nan')):.4f} |"
        )
    lines.extend([
        "",
        "## Trial Summary",
        "",
        "| trial | port | steps | stop | contact source | contact frac | Fz mean | Fz p95 | contact Fz mean | contact dF mean |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in result["rows"]:
        lines.append(
            f"| `{Path(row['trial_dir']).name}` | {row.get('port')} | {row.get('steps')} | "
            f"{row.get('stop_reason')} | {row.get('contact_source')} | "
            f"{row.get('contact_fraction', float('nan')):.4f} | "
            f"{row.get('ft_fz_mean', float('nan')):.4f} | "
            f"{row.get('ft_fz_p95', float('nan')):.4f} | "
            f"{row.get('ft_fz_contact_mean', float('nan')):.4f} | "
            f"{row.get('ft_fz_contact_delta_abs_mean', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/home/chenshuai/Project/output/board_force_rollouts")
    parser.add_argument("--output_dir", default="/home/chenshuai/Project/output/board_force_rollout_eval")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--contact_source", default="auto",
                        help="Signal used to infer contact phase: auto, left_marker_mag_mean, ft_f_mag, etc.")
    parser.add_argument("--contact_threshold_frac", type=float, default=0.25,
                        help="Robust threshold as p10 + frac * (p90 - p10).")
    parser.add_argument("--min_contact_fraction", type=float, default=0.05,
                        help="Minimum fraction required to accept an inferred contact mask.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    traces = discover(root)
    if not traces:
        raise FileNotFoundError(f"No force_trace.csv files under {root}")
    rows = [
        summarize_trace(
            p,
            contact_source=args.contact_source,
            contact_threshold_frac=args.contact_threshold_frac,
            min_contact_fraction=args.min_contact_fraction,
        )
        for p in traces
    ]
    tag = args.tag or root.name
    out_dir = Path(args.output_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "board_force_rollout_summary.csv"
    group_summary_csv = out_dir / "board_force_rollout_group_summary.csv"
    summary_json = out_dir / "board_force_rollout_summary.json"
    summary_md = out_dir / "board_force_rollout_summary.md"
    overview_png = out_dir / "board_force_overview.png"
    group_curves_png = out_dir / "board_force_group_curves.png"
    write_summary_csv(rows, summary_csv)
    group_summary = grouped_summary(rows)
    write_group_summary_csv(group_summary, group_summary_csv)
    overview = write_overview_plot(rows, traces, overview_png)
    group_curves = write_group_curves_plot(rows, traces, group_curves_png)
    result = {
        "root": str(root),
        "n_trials": len(rows),
        "contact_source_arg": args.contact_source,
        "contact_threshold_frac": args.contact_threshold_frac,
        "min_contact_fraction": args.min_contact_fraction,
        "summary_csv": str(summary_csv),
        "group_summary_csv": str(group_summary_csv),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "overview_plot": overview,
        "group_curves_plot": group_curves,
        "group_summary": group_summary,
        "rows": rows,
    }
    summary_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(result, summary_md)
    print(json.dumps({k: result[k] for k in ["n_trials", "summary_csv", "group_summary_csv", "summary_md", "overview_plot", "group_curves_plot"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
