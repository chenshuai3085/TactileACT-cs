#!/usr/bin/env bash
set -euo pipefail

# Task-local marker TactileVAE for 2026-06-30 huaping data.
# The external disk is treated as a read-only data source; all checkpoints and
# logs are written to the internal NVMe output root.

cd /home/chenshuai/Project/TactileACT-cs

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-512}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-2}
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

DATA_DIRS=(
  "/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/260630_v8j_huaping/peg_in_hole_0630"
)
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
OUTPUT_DIR="${OUTPUT_ROOT}/tactile_vae_huaping_260630_left_tw8_ld16_s2_e150"
LOG_DIR="${OUTPUT_ROOT}/tactile_vae_logs"
LOG_FILE="${LOG_DIR}/tactile_vae_huaping_260630_left_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cat > "${OUTPUT_DIR}/README_task.txt" <<EOF
task=huaping
date=2026-07-05
model=TactileVAE marker_offset, old TFAC_V5/tactile_vae.py compatible
side=left
temporal_window=8
latent_dim=16
sample_stride=${SAMPLE_STRIDE}
epochs=${EPOCHS}
data_used=${DATA_DIRS[*]}
data_excluded=/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/260630_v8j_huaping/peg_in_hole_0630/无变化
output_root=${OUTPUT_ROOT}
notes=Use only the 100 root huaping episodes; the external disk is read-only and outputs are stored on internal NVMe.
EOF

{
  echo "=== run_tactile_vae_huaping_260630_left ==="
  echo "start_time=$(date '+%F %T')"
  echo "gpu=${GPU}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "data_dirs=${DATA_DIRS[*]}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "excluded=无变化"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_CMD[@]}" TFAC_V5/pretrain_tactile_vae.py \
  --data_dirs "${DATA_DIRS[@]}" \
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
  2>&1 | tee -a "${LOG_FILE}"
