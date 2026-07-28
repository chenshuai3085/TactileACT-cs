#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONFIG=${CONFIG:-TFAC_V5/config_pretrain_foresight_card_positive_cardvae_marker_only_temporalstride2.json}
CONDA_ENV=${CONDA_ENV:-TactileACT}
GPU=${GPU:-0}
LOG_DIR=${LOG_DIR:-/home/chenshuai/Project/output/foresight_logs}

mkdir -p "${LOG_DIR}"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/foresight_card_temporalstride2_${RUN_ID}.log"

echo "=== train_foresight_card_temporalstride2 ==="
echo "start_time=$(date '+%F %T')"
echo "config=${CONFIG}"
echo "log=${LOG_FILE}"
echo "gpu=${GPU}"
echo "alignment=obs[t] + action[t,t+2,...,t+30] -> tactile[t,t+2,...,t+30]"
echo "serving_alignment=none for temporalstride2 DP"
echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"

CUDA_VISIBLE_DEVICES="${GPU}" \
conda run --no-capture-output -n "${CONDA_ENV}" python -u \
  TFAC_V5/pretrain_latent_foresight_multistep.py \
  --config "${CONFIG}" \
  2>&1 | tee "${LOG_FILE}"
