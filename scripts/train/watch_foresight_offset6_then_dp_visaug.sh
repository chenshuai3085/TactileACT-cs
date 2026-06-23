#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

FORESIGHT_BASE="/home/chenshuai/Project/output/foresight_ckpt"
FORESIGHT_NAME="latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_offset6_e100_bs16_preload"
FORESIGHT_LOG_DIR="/home/chenshuai/Project/output/foresight_logs"
WATCH_LOG="${FORESIGHT_LOG_DIR}/watch_foresight_offset6_then_dp_visaug.log"
NEXT_SCRIPT="scripts/train/train_dp_tac_concat_board_260609_260610_action_offset6_vision_aug.sh"
VISAUG_DP_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_action_offset6_visaug_e1000"

mkdir -p "${FORESIGHT_LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

find_foresight_dir() {
    local candidate
    for candidate in "${FORESIGHT_BASE}/${FORESIGHT_NAME}" "${FORESIGHT_BASE}/${FORESIGHT_NAME}"_*; do
        if [[ -f "${candidate}/foresight_best.ckpt" && -f "${candidate}/foresight_last.ckpt" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

log "watcher started"
log "wait_for=offset6 foresight finished checkpoint"
log "next=${NEXT_SCRIPT}"
log "next_save_dir=${VISAUG_DP_DIR}"

while true; do
    if pgrep -f "pretrain_latent_foresight_multistep.py.*config_pretrain_foresight_board_multistep16_boardvae_marker_only_offset6" >/dev/null 2>&1; then
        latest_dir="$(ls -dt "${FORESIGHT_BASE}/${FORESIGHT_NAME}"* 2>/dev/null | head -1 || true)"
        if [[ -n "${latest_dir}" && -f "${latest_dir}/pretrain_history.pkl" ]]; then
            log "offset6 foresight still running. latest_dir=${latest_dir}"
        else
            log "offset6 foresight still running."
        fi
        sleep 300
        continue
    fi

    if foresight_dir="$(find_foresight_dir)"; then
        log "offset6 foresight finished: ${foresight_dir}"
        if pgrep -f "diffusion/train_dp_tac_concat.py.*action_offset 6.*vision_aug" >/dev/null 2>&1; then
            log "vision-aug offset6 DP already running; exiting watcher"
            exit 0
        fi
        if [[ -f "${VISAUG_DP_DIR}/dp_final.pth" ]]; then
            log "vision-aug offset6 DP already has dp_final.pth; nothing to start"
            exit 0
        fi
        log "starting vision-aug offset6 DP"
        bash "${NEXT_SCRIPT}" 2>&1 | tee -a "${WATCH_LOG}"
        status=${PIPESTATUS[0]}
        log "vision-aug offset6 DP finished with code ${status}"
        exit "${status}"
    fi

    log "offset6 foresight not finished yet"
    sleep 300
done
