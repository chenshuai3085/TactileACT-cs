#!/usr/bin/env python3
"""Monitor a DP training run directory.

This script only reads the training log and system status.  It does not stop,
restart, or modify the training process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(r"^Ep\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+)")
VAL_RE = re.compile(r"\|\s+val=([0-9.eE+-]+)")
BEST_RE = re.compile(r"best=[^=]+=([0-9.eE+-]+|pending)")


def read_rows(train_log: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not train_log.exists():
        return rows
    with train_log.open("r", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            match = EPOCH_RE.search(stripped)
            if not match:
                continue
            epoch, total, train = match.groups()
            val_match = VAL_RE.search(stripped)
            best_match = BEST_RE.search(stripped)
            val = val_match.group(1) if val_match else None
            best = best_match.group(1) if best_match else "pending"
            rows.append(
                {
                    "epoch": int(epoch),
                    "total": int(total),
                    "train": float(train),
                    "val": None if val is None else float(val),
                    "best": None if best == "pending" else float(best),
                }
            )
    return rows


def find_pid(run_dir: Path) -> str | None:
    pattern = "diffusion/train_dp_tac_concat.py"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if (
            "diffusion/train_dp_tac_concat.py" in line
            and str(run_dir) in line
            and "python" in line
            and "tmux new-session" not in line
        ):
            return line.split()[0]
    return None


def gpu_line() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.splitlines()[0].strip()
    return ""


def free_gb(path: str) -> int | None:
    try:
        return int(shutil.disk_usage(path).free // (1024**3))
    except FileNotFoundError:
        return None


def checkpoint_state(run_dir: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in ("dp_best.pth", "dp_latest.pth", "dp_final.pth"):
        path = run_dir / name
        if path.exists():
            state[name] = {
                "size_gb": round(path.stat().st_size / (1024**3), 3),
                "mtime": time.strftime("%F %T", time.localtime(path.stat().st_mtime)),
            }
    for path in sorted(run_dir.glob("dp_epoch*.pth"))[-5:]:
        state[path.name] = {
            "size_gb": round(path.stat().st_size / (1024**3), 3),
            "mtime": time.strftime("%F %T", time.localtime(path.stat().st_mtime)),
        }
    return state


def update_loss_curve(run_dir: Path, log: Any) -> None:
    train_log = run_dir / "train.log"
    if not train_log.exists():
        return
    repo_root = Path(__file__).resolve().parents[2]
    plot_script = repo_root / "scripts" / "utils" / "plot_dp_training_log.py"
    result = subprocess.run(
        [
            "python",
            str(plot_script),
            "--log",
            str(train_log),
            "--out_dir",
            str(run_dir),
            "--smooth",
            "9",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        log.write(
            f"[{dt.datetime.now():%F %T}] warning: plot update failed: "
            f"{result.stderr.strip()}\n"
        )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 8)


def tail_stats(rows: list[dict[str, Any]], count: int) -> dict[str, Any] | None:
    if not rows:
        return None
    tail = rows[-count:]
    vals = [float(row["val"]) for row in tail if row.get("val") is not None]
    trains = [float(row["train"]) for row in tail]
    return {
        "n": len(tail),
        "epoch_start": tail[0]["epoch"],
        "epoch_end": tail[-1]["epoch"],
        "train_mean": _mean(trains),
        "val_n": len(vals),
        "val_mean": _mean(vals),
        "val_min": min(vals) if vals else None,
        "val_max": max(vals) if vals else None,
    }


def trend_state(rows: list[dict[str, Any]], latest: dict[str, Any] | None,
                best: dict[str, Any] | None) -> dict[str, Any]:
    if not rows or latest is None or best is None:
        return {
            "epochs_since_best": None,
            "latest_val_minus_best": None,
            "tail20": None,
            "tail50": None,
            "warning": "insufficient_log_rows",
        }

    epochs_since_best = int(latest["epoch"]) - int(best["epoch"])
    latest_val_row = next((row for row in reversed(rows) if row.get("val") is not None), None)
    if latest_val_row is None or best.get("val") is None:
        return {
            "epochs_since_best": epochs_since_best,
            "latest_val_minus_best": None,
            "latest_val_over_best_ratio": None,
            "latest_val_epoch": None,
            "tail20": tail_stats(rows, 20),
            "tail50": tail_stats(rows, 50),
            "warning": "no_validation_rows_yet",
        }

    latest_val = float(latest_val_row["val"])
    best_val = float(best["val"])
    latest_gap = latest_val - best_val
    tail20 = tail_stats(rows, 20)
    tail50 = tail_stats(rows, 50)

    warning = "healthy"
    if epochs_since_best >= 100 and tail20 and tail20["val_min"] is not None and tail20["val_min"] > best_val * 1.05:
        warning = "strong_plateau_or_overfit_use_best"
    elif epochs_since_best >= 50 and latest_gap > best_val * 0.2:
        warning = "watch_plateau_use_best_for_deploy"
    elif epochs_since_best >= 25:
        warning = "watching_no_recent_best"

    return {
        "epochs_since_best": epochs_since_best,
        "latest_val_minus_best": round(latest_gap, 8),
        "latest_val_over_best_ratio": round(latest_val / best_val, 6) if best_val else None,
        "latest_val_epoch": latest_val_row,
        "tail20": tail20,
        "tail50": tail50,
        "warning": warning,
    }


def build_status(run_dir: Path) -> dict[str, Any]:
    rows = read_rows(run_dir / "train.log")
    latest = rows[-1] if rows else None
    val_rows = [row for row in rows if row.get("val") is not None]
    best = min(val_rows, key=lambda row: row["val"]) if val_rows else (
        min(rows, key=lambda row: row["train"]) if rows else None
    )
    return {
        "timestamp": f"{dt.datetime.now():%F %T}",
        "run_dir": str(run_dir),
        "latest": latest,
        "latest_val_epoch": val_rows[-1] if val_rows else None,
        "best_val_epoch": best,
        "trend": trend_state(rows, latest, best),
        "pid": find_pid(run_dir),
        "gpu": gpu_line(),
        "free_ext_gb": free_gb("/media/chenshuai/EXTERNAL_USB"),
        "free_home_gb": free_gb("/home/chenshuai/Project/output"),
        "checkpoints": checkpoint_state(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--interval_sec", type=int, default=300)
    parser.add_argument("--plot_interval_sec", type=int, default=300)
    parser.add_argument("--stop_when_missing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    status_path = run_dir / "training_status_latest.json"
    monitor_log = run_dir / "monitor_training.log"
    monitor_log.parent.mkdir(parents=True, exist_ok=True)

    with monitor_log.open("a", buffering=1) as log:
        log.write(f"[{dt.datetime.now():%F %T}] monitor started run_dir={run_dir}\n")
        last_plot_time = 0.0
        while True:
            status = build_status(run_dir)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            log.write(
                f"[{status['timestamp']}] latest={status['latest']} "
                f"best={status['best_val_epoch']} pid={status['pid'] or 'none'} "
                f"gpu='{status['gpu']}' free_ext={status['free_ext_gb']}G "
                f"free_home={status['free_home_gb']}G\n"
            )
            if args.stop_when_missing and not status["pid"]:
                log.write(f"[{dt.datetime.now():%F %T}] training process not found; monitor exits\n")
                return
            now = time.time()
            if now - last_plot_time >= max(1, args.plot_interval_sec):
                update_loss_curve(run_dir, log)
                last_plot_time = now
            time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    main()
