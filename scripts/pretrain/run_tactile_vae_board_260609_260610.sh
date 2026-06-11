#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONDA_ENV=${CONDA_ENV:-TactileACT}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-512}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-2}
OUTPUT_DIR=${OUTPUT_DIR:-/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150}
LOG_DIR=${LOG_DIR:-/home/chenshuai/Project/output/tactile_vae_logs}

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/tactile_vae_board_260609_260610_$(date +%Y%m%d_%H%M%S).log"

echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"
echo "GPU: ${GPU}"

CUDA_VISIBLE_DEVICES="${GPU}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -u TFAC_V5/pretrain_tactile_vae.py \
    --data_dirs \
      /home/chenshuai/data/dataset/260609/wipe_pos_straight_z124_125_150_20260609 \
      /home/chenshuai/data/dataset/260609/z_too_high \
      /home/chenshuai/data/dataset/260610/z_too_low \
      /home/chenshuai/data/dataset/260610/z_too_oscillate \
    --sides left \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr 1e-4 \
    --temporal_window 8 \
    --latent_dim 16 \
    --kl_weight 1e-6 \
    --direction_weight 0.2 \
    --sample_stride "${SAMPLE_STRIDE}" \
    --val_ratio 0.1 \
    --save_every 50 \
    --vis_every 10 \
    --vis_samples 4 \
    --num_workers 4 \
    --seed 42 \
  2>&1 | tee "${LOG_FILE}"
