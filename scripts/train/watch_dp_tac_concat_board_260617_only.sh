#!/bin/bash
set -euo pipefail

RUN_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000"
WATCH_LOG="${RUN_DIR}/watch_training.log"

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-600}"
STOP_ON_PLATEAU="${STOP_ON_PLATEAU:-1}"
PATIENCE_EPOCHS="${PATIENCE_EPOCHS:-350}"
MIN_EPOCH_BEFORE_EARLY_STOP="${MIN_EPOCH_BEFORE_EARLY_STOP:-1500}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

mkdir -p "${RUN_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

latest_status() {
    /home/chenshuai/miniconda3/envs/TactileACT/bin/python - "$RUN_DIR/train.log" <<'PY'
import math
import re
import sys

path = sys.argv[1]
rows = []
try:
    with open(path, errors="ignore") as f:
        for line in f:
            header = re.search(r"Ep\s+(\d+)/(\d+)\s+\|\s+train=([0-9.eE+-]+)", line)
            if not header:
                continue
            epoch = int(header.group(1))
            total = int(header.group(2))
            train = float(header.group(3))
            val_match = re.search(r"\|\s+val=([0-9.eE+-]+)", line)
            best_match = re.search(r"\|\s+best=(?:val_loss|train_loss)=([0-9.eE+-]+|pending)", line)
            val = float(val_match.group(1)) if val_match else math.nan
            best = float(best_match.group(1)) if best_match and best_match.group(1) != "pending" else math.inf
            rows.append((epoch, total, train, val, best))
except FileNotFoundError:
    pass

if not rows:
    print("0 0 nan nan nan 0")
    raise SystemExit

valid_for_best = [r for r in rows if math.isfinite(r[3])]
if valid_for_best:
    best_epoch, _, _, best_val, _ = min(valid_for_best, key=lambda x: x[3])
else:
    best_epoch, _, best_train, _, _ = min(rows, key=lambda x: x[2])
    best_val = best_train
latest_epoch, total_epoch, latest_train, latest_val, _ = rows[-1]
print(latest_epoch, total_epoch, latest_train, latest_val, best_val, best_epoch)
PY
}

train_pids() {
    pgrep -f "diffusion/train_dp_tac_concat.py" | while read -r pid; do
        ps -p "${pid}" -o args= | grep -F "dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000" >/dev/null && echo "${pid}"
    done || true
}

free_gb() {
    df -BG /home/chenshuai/Project/output | awk 'NR==2 {gsub("G","",$4); print $4}'
}

gpu_line() {
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true
}

log "watcher started: interval=${CHECK_INTERVAL_SEC}s stop_on_plateau=${STOP_ON_PLATEAU} patience=${PATIENCE_EPOCHS} min_epoch=${MIN_EPOCH_BEFORE_EARLY_STOP}"

while true; do
    read -r latest_epoch total_epoch latest_train latest_val best_val best_epoch < <(latest_status)
    pids="$(train_pids | tr '\n' ' ')"
    free="$(free_gb)"
    gpu="$(gpu_line)"
    log "status: latest=${latest_epoch}/${total_epoch} train=${latest_train} val=${latest_val} best=${best_val}@${best_epoch} free_gb=${free} gpu='${gpu}' pids='${pids}'"

    if [[ -z "${pids// }" ]]; then
        log "training process is not running; watcher exits"
        exit 0
    fi

    if [[ "${latest_epoch}" != "0" && "${total_epoch}" != "0" && "${latest_epoch}" -ge "${total_epoch}" ]]; then
        log "training reached total epochs; watcher exits"
        exit 0
    fi

    if [[ "${latest_epoch}" != "0" && ( "${latest_train}" == "nan" || "${latest_val}" == "nan" ) ]]; then
        log "warning: latest val may be nan before first validation; continuing unless train is nan"
    fi
    if [[ "${latest_train}" == "nan" && "${latest_epoch}" != "0" ]]; then
        log "fatal: train loss is nan; stopping training"
        for pid in ${pids}; do kill "${pid}" 2>/dev/null || true; done
        exit 1
    fi

    if [[ "${free}" -lt "${MIN_FREE_GB}" ]]; then
        log "fatal: free disk below ${MIN_FREE_GB}G; stopping training to avoid corrupt checkpoints"
        for pid in ${pids}; do kill "${pid}" 2>/dev/null || true; done
        exit 1
    fi

    no_improve=$(( latest_epoch - best_epoch ))
    if [[ "${STOP_ON_PLATEAU}" == "1" && "${latest_epoch}" -ge "${MIN_EPOCH_BEFORE_EARLY_STOP}" && "${no_improve}" -ge "${PATIENCE_EPOCHS}" ]]; then
        log "early stopping: no best-val improvement for ${no_improve} epochs"
        for pid in ${pids}; do kill "${pid}" 2>/dev/null || true; done
        sleep 10
        for pid in ${pids}; do kill -9 "${pid}" 2>/dev/null || true; done
        log "training stopped by watcher"
        exit 0
    fi

    sleep "${CHECK_INTERVAL_SEC}"
done
