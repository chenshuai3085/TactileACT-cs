#!/bin/bash
set -euo pipefail

RUN_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000"
REPO_DIR="/home/chenshuai/Project/TactileACT-cs"
LOG_PATH="${RUN_DIR}/train.log"
MONITOR_LOG="${RUN_DIR}/monitor_training.log"
STATUS_TXT="${RUN_DIR}/training_status_latest.txt"
INTERVAL_SEC="${INTERVAL_SEC:-600}"

mkdir -p "${RUN_DIR}"
echo "$$" > "${RUN_DIR}/monitor.pid"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${MONITOR_LOG}"
}

latest_status() {
    /home/chenshuai/miniconda3/envs/TactileACT/bin/python - "$LOG_PATH" "$STATUS_TXT" <<'PY'
import math
import os
import re
import sys
from datetime import datetime

log_path, status_path = sys.argv[1], sys.argv[2]
rows = []
pattern = re.compile(
    r"Ep\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+).*?"
    r"\|\s+val=([0-9.eE+-]+).*?\|\s+best=val_loss=([0-9.eE+-]+)"
)

try:
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append((
                    int(m.group(1)),
                    int(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                ))
except FileNotFoundError:
    rows = []

if not rows:
    text = f"time={datetime.now().isoformat(timespec='seconds')}\nstatus=no_epoch_rows\n"
    print("no_epoch_rows")
else:
    latest = rows[-1]
    best = min(rows, key=lambda x: x[3])
    no_improve = latest[0] - best[0]
    text = "\n".join([
        f"time={datetime.now().isoformat(timespec='seconds')}",
        f"latest_epoch={latest[0]}",
        f"total_epoch={latest[1]}",
        f"latest_train={latest[2]:.6f}",
        f"latest_val={latest[3]:.6f}",
        f"best_epoch={best[0]}",
        f"best_val={best[3]:.6f}",
        f"epochs_since_best={no_improve}",
        f"best_metric_in_log={latest[4]:.6f}",
        "",
    ])
    print(
        f"latest={latest[0]}/{latest[1]} train={latest[2]:.6f} "
        f"val={latest[3]:.6f} best={best[3]:.6f}@{best[0]} "
        f"no_improve={no_improve}"
    )

tmp = status_path + ".tmp"
with open(tmp, "w") as f:
    f.write(text)
os.replace(tmp, status_path)
PY
}

train_pids() {
    pgrep -f "diffusion/train_dp_tac_concat.py" | while read -r pid; do
        ps -p "${pid}" -o args= | grep -F "${RUN_DIR}" >/dev/null && echo "${pid}"
    done || true
}

log "monitor started: interval=${INTERVAL_SEC}s"

while true; do
    pids="$(train_pids | tr '\n' ' ')"
    status="$(latest_status)"
    log "status: ${status} pids='${pids}'"

    if [[ -f "${LOG_PATH}" ]]; then
        /home/chenshuai/miniconda3/envs/TactileACT/bin/python \
            "${REPO_DIR}/scripts/utils/plot_dp_training_log.py" \
            --log "${LOG_PATH}" \
            --out_dir "${RUN_DIR}" \
            --smooth 9 >> "${MONITOR_LOG}" 2>&1 || log "warning: plot update failed"
    fi

    df -h /home | tail -1 | awk '{print "disk_home="$4" free, used="$5}' >> "${STATUS_TXT}" || true
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | \
        awk -F, '{gsub(/^ +| +$/,"",$1); gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$3); gsub(/^ +| +$/,"",$4); print "gpu_mem="$1"/"$2" MiB\ngpu_util="$3"%\ngpu_temp="$4"C"}' >> "${STATUS_TXT}" || true

    if [[ -z "${pids// }" ]]; then
        log "training process is not running; monitor exits"
        exit 0
    fi

    sleep "${INTERVAL_SEC}"
done
