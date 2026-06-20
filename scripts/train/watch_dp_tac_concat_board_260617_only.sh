#!/bin/bash
set -euo pipefail

RUN_DIR="${RUN_DIR:-/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260620_rerun}"
WATCH_LOG="${RUN_DIR}/watch_training.log"

CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-600}"
STOP_ON_PLATEAU="${STOP_ON_PLATEAU:-1}"
PATIENCE_EPOCHS="${PATIENCE_EPOCHS:-350}"
MIN_EPOCH_BEFORE_EARLY_STOP="${MIN_EPOCH_BEFORE_EARLY_STOP:-1500}"
TAIL_WINDOW="${TAIL_WINDOW:-60}"
TAIL_VAL_OVER_BEST_RATIO="${TAIL_VAL_OVER_BEST_RATIO:-1.05}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

mkdir -p "${RUN_DIR}"
echo "$$" > "${RUN_DIR}/watch.pid"

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
    print("0 0 nan nan nan 0 nan nan")
    raise SystemExit

valid_for_best = [r for r in rows if math.isfinite(r[3])]
if valid_for_best:
    best_epoch, _, _, best_val, _ = min(valid_for_best, key=lambda x: x[3])
else:
    best_epoch, _, best_train, _, _ = min(rows, key=lambda x: x[2])
    best_val = best_train
latest_epoch, total_epoch, latest_train, latest_val, _ = rows[-1]
tail_window = int(__import__("os").environ.get("TAIL_WINDOW", "60"))
tail = rows[-tail_window:]
tail_vals = [r[3] for r in tail if math.isfinite(r[3])]
tail_val_min = min(tail_vals) if tail_vals else math.nan
tail_val_mean = sum(tail_vals) / len(tail_vals) if tail_vals else math.nan
print(latest_epoch, total_epoch, latest_train, latest_val, best_val, best_epoch, tail_val_min, tail_val_mean)
PY
}

train_pids() {
    pgrep -f "diffusion/train_dp_tac_concat.py" | while read -r pid; do
        ps -p "${pid}" -o args= | grep -F "${RUN_DIR}" >/dev/null && echo "${pid}"
    done || true
}

free_gb() {
    df -BG "${RUN_DIR}" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

gpu_line() {
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true
}

log "watcher started: run_dir=${RUN_DIR} interval=${CHECK_INTERVAL_SEC}s stop_on_plateau=${STOP_ON_PLATEAU} patience=${PATIENCE_EPOCHS} min_epoch=${MIN_EPOCH_BEFORE_EARLY_STOP} tail_window=${TAIL_WINDOW} tail_val_ratio=${TAIL_VAL_OVER_BEST_RATIO}"

while true; do
    read -r latest_epoch total_epoch latest_train latest_val best_val best_epoch tail_val_min tail_val_mean < <(latest_status)
    pids="$(train_pids | tr '\n' ' ')"
    free="$(free_gb)"
    gpu="$(gpu_line)"
    log "status: latest=${latest_epoch}/${total_epoch} train=${latest_train} val=${latest_val} best=${best_val}@${best_epoch} tail_val_min=${tail_val_min} tail_val_mean=${tail_val_mean} free_gb=${free} gpu='${gpu}' pids='${pids}'"

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
        if /home/chenshuai/miniconda3/envs/TactileACT/bin/python - "$tail_val_min" "$best_val" "$TAIL_VAL_OVER_BEST_RATIO" <<'PY'
import math
import sys
tail_val_min = float(sys.argv[1])
best_val = float(sys.argv[2])
ratio = float(sys.argv[3])
raise SystemExit(0 if math.isfinite(tail_val_min) and math.isfinite(best_val) and tail_val_min > best_val * ratio else 1)
PY
        then
            log "early stopping: no best-val improvement for ${no_improve} epochs and tail val is > ${TAIL_VAL_OVER_BEST_RATIO}x best"
            for pid in ${pids}; do kill "${pid}" 2>/dev/null || true; done
            sleep 10
            for pid in ${pids}; do kill -9 "${pid}" 2>/dev/null || true; done
            log "training stopped by watcher; keep dp_best.pth for deployment/offline tests"
            exit 0
        fi
        log "plateau watch: no best-val improvement for ${no_improve} epochs, but tail val is not consistently worse than threshold; continuing"
    fi

    sleep "${CHECK_INTERVAL_SEC}"
done
