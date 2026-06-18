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


EPOCH_RE = re.compile(
    r"^Ep\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+)"
    r".*?\|\s+val=([0-9.eE+-]+).*?best=[^=]+=([0-9.eE+-]+|pending)"
)


def read_rows(train_log: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not train_log.exists():
        return rows
    with train_log.open("r", errors="ignore") as f:
        for line in f:
            match = EPOCH_RE.search(line.strip())
            if not match:
                continue
            epoch, total, train, val, best = match.groups()
            rows.append(
                {
                    "epoch": int(epoch),
                    "total": int(total),
                    "train": float(train),
                    "val": float(val),
                    "best": None if best == "pending" else float(best),
                }
            )
    return rows


def find_pid(run_dir: Path) -> str | None:
    pattern = f"diffusion/train_dp_tac_concat.py.*{run_dir}"
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if "diffusion/train_dp_tac_concat.py" in line:
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


def build_status(run_dir: Path) -> dict[str, Any]:
    rows = read_rows(run_dir / "train.log")
    latest = rows[-1] if rows else None
    best = min(rows, key=lambda row: row["val"]) if rows else None
    return {
        "timestamp": f"{dt.datetime.now():%F %T}",
        "run_dir": str(run_dir),
        "latest": latest,
        "best_val_epoch": best,
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
    parser.add_argument("--stop_when_missing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    status_path = run_dir / "training_status_latest.json"
    monitor_log = run_dir / "monitor_training.log"
    monitor_log.parent.mkdir(parents=True, exist_ok=True)

    with monitor_log.open("a", buffering=1) as log:
        log.write(f"[{dt.datetime.now():%F %T}] monitor started run_dir={run_dir}\n")
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
            time.sleep(max(1, args.interval_sec))


if __name__ == "__main__":
    main()
