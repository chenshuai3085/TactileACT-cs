#!/usr/bin/env python3
"""Read-only watcher for DP train.log files.

The watcher intentionally does not stop or modify training.  It writes compact
status files that are easy to inspect while a long tmux training run continues.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


EPOCH_RE = re.compile(
    r"Ep\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+).*?"
    r"\|\s+val=([0-9.eE+-]+).*?best=([^=]+)=([0-9.eE+-]+|pending)"
)


def parse_rows(log_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_path.exists():
        return rows
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = EPOCH_RE.search(line)
            if not match:
                continue
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "total": int(match.group(2)),
                    "train": float(match.group(3)),
                    "val": float(match.group(4)),
                    "best_metric_name": match.group(5),
                    "best_metric": None
                    if match.group(6) == "pending"
                    else float(match.group(6)),
                }
            )
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(float(r[key]) for r in rows) / len(rows))


def gpu_status() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else ""


def disk_free_gb(path: str) -> int | None:
    try:
        return int(shutil.disk_usage(path).free // (1024**3))
    except FileNotFoundError:
        return None


def build_status(run_dir: Path) -> dict[str, Any]:
    rows = parse_rows(run_dir / "train.log")
    status: dict[str, Any] = {
        "timestamp": time.strftime("%F %T %Z"),
        "run_dir": str(run_dir),
        "epoch_rows": len(rows),
        "gpu": gpu_status(),
        "free_ext_gb": disk_free_gb("/media/chenshuai/EXTERNAL_USB"),
        "free_home_gb": disk_free_gb("/home/chenshuai/Project/output"),
    }
    if rows:
        latest = rows[-1]
        best = min(rows, key=lambda row: float(row["val"]))
        tail20 = rows[-20:]
        tail50 = rows[-50:]
        status.update(
            {
                "latest": latest,
                "best": best,
                "epochs_since_best": int(latest["epoch"]) - int(best["epoch"]),
                "val_over_best": float(latest["val"]) / float(best["val"]),
                "tail20_val_min": min(float(r["val"]) for r in tail20),
                "tail20_val_mean": mean(tail20, "val"),
                "tail20_train_mean": mean(tail20, "train"),
                "tail50_val_min": min(float(r["val"]) for r in tail50),
                "tail50_val_mean": mean(tail50, "val"),
                "tail50_train_mean": mean(tail50, "train"),
            }
        )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--interval_sec", type=int, default=600)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    status_path = run_dir / "training_watch_status.json"
    log_path = run_dir / "training_watch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", buffering=1, encoding="utf-8") as log:
        log.write(f"[{time.strftime('%F %T %Z')}] read-only watcher started\n")
        while True:
            status = build_status(run_dir)
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            latest = status.get("latest")
            best = status.get("best")
            log.write(
                f"[{status['timestamp']}] latest={latest} best={best} "
                f"gpu={status.get('gpu')} free_ext={status.get('free_ext_gb')}G "
                f"free_home={status.get('free_home_gb')}G\n"
            )
            time.sleep(max(1, int(args.interval_sec)))


if __name__ == "__main__":
    main()
