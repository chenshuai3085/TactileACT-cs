#!/usr/bin/env bash
set -euo pipefail

# Single-GPU queue for the 260707 card tactile run:
#   1) train task-local card TactileVAE with temporal_window=2
#   2) train tactile+vision DP with temporal_stride=1, tac_history=2, epochs=600

cd /home/chenshuai/Project/TactileACT-cs

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
GPU=${GPU:-0}
VAE_EPOCHS=${VAE_EPOCHS:-150}
DP_EPOCHS=${DP_EPOCHS:-600}
LOG_DIR="${OUTPUT_ROOT}/queue_logs"
LOG_FILE="${LOG_DIR}/card_260707_tactile_full_queue_$(date +%Y%m%d_%H%M%S).log"
CARD_VAE="${OUTPUT_ROOT}/tactile_vae_card_260707_left_tw2_ld16_s1_e${VAE_EPOCHS}/best_tactile_vae.pt"

mkdir -p "${LOG_DIR}"

{
  echo "=== run_card_260707_tactile_full_queue ==="
  echo "start_time=$(date '+%F %T')"
  echo "gpu=${GPU}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "vae_epochs=${VAE_EPOCHS}"
  echo "dp_epochs=${DP_EPOCHS}"
  echo "card_vae=${CARD_VAE}"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${LOG_FILE}"

echo "[queue] Step 1/2: train card TactileVAE tw2" | tee -a "${LOG_FILE}"
OUTPUT_ROOT="${OUTPUT_ROOT}" GPU="${GPU}" EPOCHS="${VAE_EPOCHS}" SAMPLE_STRIDE=1 \
  bash scripts/pretrain/run_tactile_vae_card_260707_left_tw2.sh 2>&1 | tee -a "${LOG_FILE}"

if [[ ! -f "${CARD_VAE}" ]]; then
  echo "[queue] Missing expected VAE checkpoint: ${CARD_VAE}" | tee -a "${LOG_FILE}"
  exit 1
fi

echo "[queue] Step 2/2: train DP tactile+vision stride1 tac_history2" | tee -a "${LOG_FILE}"
OUTPUT_ROOT="${OUTPUT_ROOT}" GPU="${GPU}" EPOCHS="${DP_EPOCHS}" VAE_EPOCHS="${VAE_EPOCHS}" CARD_VAE="${CARD_VAE}" \
  bash scripts/train/train_dp_tac_concat_card_260707_left_cardvae_ph16_oh2_stride1_tachist2_dynamic32768_e600.sh 2>&1 | tee -a "${LOG_FILE}"

echo "[queue] Done at $(date '+%F %T')" | tee -a "${LOG_FILE}"
