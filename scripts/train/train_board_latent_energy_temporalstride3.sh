#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONDA_ENV=${CONDA_ENV:-TactileACT}
GPU=${GPU:-0}
OUTPUT_DIR=${OUTPUT_DIR:-/home/chenshuai/Project/output/board_latent_energy/ce_margin_temporalstride3_e40}
LOG_DIR=${LOG_DIR:-/home/chenshuai/Project/output/board_latent_energy_logs}
EPOCHS=${EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-128}
STRIDE=${STRIDE:-8}
TEMPORAL_STRIDE=${TEMPORAL_STRIDE:-3}
CONTACT_QUANTILE=${CONTACT_QUANTILE:-0.50}
MIN_CONTACT_RATIO=${MIN_CONTACT_RATIO:-0.25}

mkdir -p "${LOG_DIR}"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/board_latent_energy_temporalstride3_${RUN_ID}.log"

echo "=== train_board_latent_energy_temporalstride3 ==="
echo "start_time=$(date '+%F %T')"
echo "output_dir=${OUTPUT_DIR}"
echo "log=${LOG_FILE}"
echo "gpu=${GPU}"
echo "chunk=start,start+3,...,start+45"
echo "window_start_stride=${STRIDE}"
echo "temporal_stride=${TEMPORAL_STRIDE}"
echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"

CUDA_VISIBLE_DEVICES="${GPU}" \
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  TFAC_V5/board_latent_energy/train.py \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --chunk_len 16 \
  --stride "${STRIDE}" \
  --temporal_stride "${TEMPORAL_STRIDE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --contact_quantile "${CONTACT_QUANTILE}" \
  --min_contact_ratio "${MIN_CONTACT_RATIO}" \
  2>&1 | tee "${LOG_FILE}"
