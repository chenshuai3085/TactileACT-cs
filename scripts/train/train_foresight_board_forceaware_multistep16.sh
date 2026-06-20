#!/bin/bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONFIG="${CONFIG:-TFAC_V5/config_pretrain_foresight_board_forceaware_multistep16.json}"
LOG_DIR="${LOG_DIR:-/home/chenshuai/Project/output/foresight_ckpt/logs}"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/forceaware_foresight_${STAMP}.log"

echo "Config: ${CONFIG}"
echo "Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
conda run --no-capture-output -n TactileACT python -u \
  TFAC_V5/pretrain_latent_foresight_multistep_force.py \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
