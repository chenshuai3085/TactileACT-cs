#!/usr/bin/env python3
"""Compare DP training run directories for deployment checkpoint choice.

This is a read-only audit.  It compares status JSONs, checkpoint presence, and
best/latest validation behavior so a new training run is not promoted just
because it is newer or longer.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": repr(exc), "_path": str(path)}
    if isinstance(data, dict):
        data.setdefault("_source_path", str(path))
        return data
    return {"_error": "JSON root is not object", "_path": str(path)}


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def get_best(status: dict[str, Any]) -> dict[str, Any]:
    best = status.get("best_val_epoch") or status.get("best") or {}
    if isinstance(best, dict):
        return best
    return {}


def get_latest(status: dict[str, Any]) -> dict[str, Any]:
    latest = status.get("latest") or {}
    if isinstance(latest, dict):
        return latest
    return {}


def val_from(row: dict[str, Any]) -> float | None:
    return as_float(row.get("val") if "val" in row else row.get("val_loss"))


def ckpt_info(run_dir: Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_gb": round(stat.st_size / (1024**3), 3),
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
    }


def summarize_run(name: str, run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "training_status_latest.json"
    status = load_json(status_path)
    best = get_best(status)
    latest = get_latest(status)
    best_val = val_from(best)
    latest_val = val_from(latest)
    latest_over_best = None
    if best_val and latest_val:
        latest_over_best = latest_val / best_val
    trend = status.get("trend") if isinstance(status.get("trend"), dict) else {}
    checkpoints = {
        name: ckpt_info(run_dir, name)
        for name in ("dp_best.pth", "dp_latest.pth", "dp_final.pth", "dp_epoch200.pth")
    }
    return {
        "name": name,
        "run_dir": str(run_dir),
        "status_path": str(status_path),
        "status_loaded": not status.get("_missing") and not status.get("_error"),
        "best_epoch": best.get("epoch"),
        "best_val": best_val,
        "latest_epoch": latest.get("epoch"),
        "latest_val": latest_val,
        "latest_over_best": latest_over_best,
        "epochs_since_best": trend.get("epochs_since_best"),
        "trend_warning": trend.get("warning"),
        "checkpoints": checkpoints,
        "recommended_ckpt": checkpoints["dp_best.pth"]["path"] if checkpoints["dp_best.pth"]["exists"] else None,
        "avoid_ckpts": [
            info["path"]
            for key, info in checkpoints.items()
            if key != "dp_best.pth" and info["exists"]
        ],
    }


def choose_best(runs: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [run for run in runs if run.get("best_val") is not None and run.get("checkpoints", {}).get("dp_best.pth", {}).get("exists")]
    if not candidates:
        return {
            "selected_name": None,
            "selected_run_dir": None,
            "selected_ckpt": None,
            "reason": "No run has both best validation metric and dp_best.pth.",
        }
    selected = min(candidates, key=lambda run: float(run["best_val"]))
    sorted_candidates = sorted(candidates, key=lambda run: float(run["best_val"]))
    margin_to_second = None
    if len(sorted_candidates) > 1:
        margin_to_second = float(sorted_candidates[1]["best_val"]) - float(selected["best_val"])
    return {
        "selected_name": selected["name"],
        "selected_run_dir": selected["run_dir"],
        "selected_ckpt": selected["recommended_ckpt"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_best_val": selected["best_val"],
        "margin_to_second_best_val": margin_to_second,
        "reason": "Lowest episode-level best validation loss among runs with dp_best.pth.",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# DP Run Comparison",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- selected: `{report['selection']['selected_name']}`",
        f"- selected_ckpt: `{report['selection']['selected_ckpt']}`",
        f"- reason: {report['selection']['reason']}",
        "",
        "## Runs",
        "",
        "| name | best epoch | best val | latest epoch | latest val | latest/best | warning | dp_best |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for run in report["runs"]:
        latest_over_best = run.get("latest_over_best")
        latest_over_best_text = f"{latest_over_best:.4f}" if isinstance(latest_over_best, float) else "NA"
        lines.append(
            "| "
            f"{run['name']} | "
            f"{run.get('best_epoch')} | "
            f"{run.get('best_val')} | "
            f"{run.get('latest_epoch')} | "
            f"{run.get('latest_val')} | "
            f"{latest_over_best_text} | "
            f"{run.get('trend_warning')} | "
            f"{run['checkpoints']['dp_best.pth']['exists']} |"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"Use `{report['selection']['selected_ckpt']}` for deployment/evaluation unless a later run beats its episode-level validation metric.",
        "",
        "Avoid promoting `dp_latest.pth`, `dp_final.pth`, or late epoch checkpoints when validation has degraded.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tag", default="dp_run_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [summarize_run(name, Path(run_dir)) for name, run_dir in args.run]
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "runs": runs,
        "selection": choose_best(runs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.tag}.json"
    md_path = args.output_dir / f"{args.tag}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "selected_name": report["selection"]["selected_name"],
        "selected_ckpt": report["selection"]["selected_ckpt"],
        "selected_best_val": report["selection"].get("selected_best_val"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
