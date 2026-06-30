#!/usr/bin/env python3
from __future__ import annotations
"""Monitor PIH pure-vision DP runs and start the next queued run.

Policy:
- Never stop before epoch 200.
- After epoch 200, stop a run if validation best has not improved for 50 epochs.
- Start the queued 260625 rerun after the current 260630 run finishes/stops,
  but only when the 260625 dataset path is mounted.
"""

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/chenshuai/Project/TactileACT-cs")
MONITOR_LOG = Path("/tmp/pih_dp_queue_260630_then_260625.log")
POLL_SECONDS = 300
MIN_EPOCH = 200
PATIENCE_EPOCHS = 50

EPOCH_RE = re.compile(
    r"Ep (?P<epoch>\d+)/(?P<total>\d+) \| train=(?P<train>[0-9.]+)"
    r"(?: \| val=(?P<val>[0-9.]+))?.*"
)


@dataclass
class Job:
    name: str
    session: str
    script: Path
    log: Path
    dataset: Path
    save_dir: Path
    already_running: bool = False


JOBS = [
    Job(
        name="260630_huaping",
        session="dp_official_pih260630_huaping_stride1_val5_e600",
        script=ROOT / "scripts/train/train_dp_official_pih_260630_huaping_ph16_oh2_stride1_val5_e600.sh",
        log=Path("/media/chenshuai/czy_data22/pih_output/dp_official_vision_pih_260630_huaping_ph16_oh2_stride1_val5_e600/train.log"),
        dataset=Path("/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630"),
        save_dir=Path("/media/chenshuai/czy_data22/pih_output/dp_official_vision_pih_260630_huaping_ph16_oh2_stride1_val5_e600"),
        already_running=True,
    ),
    Job(
        name="260625_caheiban_rerun",
        session="dp_official_pih260625_caheiban_stride1_val5_e600_rerun_min200",
        script=ROOT / "scripts/train/train_dp_official_pih_260625_v8j_caheiban_ph16_oh2_stride1_val5_e600_rerun_min200.sh",
        log=Path("/media/chenshuai/EXTERNAL_USB/pih_output/dp_official_vision_pih_260625_v8j_caheiban_ph16_oh2_stride1_val5_e600_rerun_min200/train.log"),
        dataset=Path("/media/chenshuai/EXTERNAL_USB/pih_dataset/260625_v8j_caheiban/peg_in_hole_0625"),
        save_dir=Path("/media/chenshuai/EXTERNAL_USB/pih_output/dp_official_vision_pih_260625_v8j_caheiban_ph16_oh2_stride1_val5_e600_rerun_min200"),
    ),
]


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%F %T')} {msg}"
    print(line, flush=True)
    with MONITOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=check)


def session_exists(session: str) -> bool:
    result = run(["tmux", "has-session", "-t", session])
    return result.returncode == 0


def start_job(job: Job) -> bool:
    if not job.dataset.is_dir():
        log(f"[{job.name}] dataset not available yet: {job.dataset}")
        return False
    if session_exists(job.session):
        log(f"[{job.name}] session already running: {job.session}")
        return True
    job.save_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["tmux", "new-session", "-d", "-s", job.session, f"bash {job.script}"]
    result = run(cmd)
    if result.returncode != 0:
        log(f"[{job.name}] failed to start: {result.stderr.strip()}")
        return False
    log(f"[{job.name}] started session {job.session}")
    return True


def stop_job(job: Job, reason: str) -> None:
    if session_exists(job.session):
        log(f"[{job.name}] stopping session {job.session}: {reason}")
        run(["tmux", "send-keys", "-t", job.session, "C-c"])
        time.sleep(20)
    else:
        log(f"[{job.name}] session already stopped: {reason}")


def parse_progress(log_path: Path):
    if not log_path.exists():
        return None
    current_epoch = 0
    total_epoch = 0
    val_points = []
    train_loss = None
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if not m:
                continue
            current_epoch = int(m.group("epoch"))
            total_epoch = int(m.group("total"))
            train_loss = float(m.group("train"))
            if m.group("val") is not None:
                val_points.append((current_epoch, float(m.group("val"))))
    if current_epoch == 0:
        return None
    best_epoch = None
    best_val = None
    if val_points:
        best_epoch, best_val = min(val_points, key=lambda x: x[1])
    return {
        "epoch": current_epoch,
        "total": total_epoch,
        "train_loss": train_loss,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "val_points": val_points,
    }


def maybe_cleanup_old_epoch_ckpts(job: Job) -> None:
    usage = shutil.disk_usage(job.save_dir if job.save_dir.exists() else job.save_dir.parent)
    free_gb = usage.free / (1024 ** 3)
    if free_gb >= 40:
        return
    ckpts = sorted(job.save_dir.glob("dp_epoch*.pth"), key=lambda p: p.stat().st_mtime)
    for path in ckpts[:-2]:
        try:
            path.unlink()
            log(f"[{job.name}] low disk free={free_gb:.1f}GB, removed old ckpt {path.name}")
            usage = shutil.disk_usage(job.save_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb >= 60:
                break
        except OSError as exc:
            log(f"[{job.name}] failed to remove {path}: {exc}")


def monitor_job(job: Job) -> None:
    if not job.already_running:
        while not start_job(job):
            time.sleep(POLL_SECONDS)

    while True:
        progress = parse_progress(job.log)
        exists = session_exists(job.session)
        if progress:
            best_epoch = progress["best_epoch"]
            best_val = progress["best_val"]
            log(
                f"[{job.name}] epoch={progress['epoch']}/{progress['total']} "
                f"train={progress['train_loss']:.6f} "
                f"best_val={best_val if best_val is not None else 'NA'} "
                f"best_epoch={best_epoch if best_epoch is not None else 'NA'} "
                f"running={exists}"
            )
            maybe_cleanup_old_epoch_ckpts(job)
            if exists and best_epoch is not None and progress["epoch"] >= MIN_EPOCH:
                if progress["epoch"] - best_epoch >= PATIENCE_EPOCHS:
                    stop_job(
                        job,
                        f"no val improvement for {progress['epoch'] - best_epoch} epochs "
                        f"(best epoch {best_epoch}, best val {best_val:.6f})",
                    )
                    return
            if progress["total"] and progress["epoch"] >= progress["total"]:
                log(f"[{job.name}] reached final epoch")
                return
        else:
            log(f"[{job.name}] waiting for log progress, running={exists}")

        if not exists:
            log(f"[{job.name}] session ended")
            return
        time.sleep(POLL_SECONDS)


def main() -> None:
    log("queue monitor started")
    for job in JOBS:
        monitor_job(job)
    log("queue monitor finished")


if __name__ == "__main__":
    main()
