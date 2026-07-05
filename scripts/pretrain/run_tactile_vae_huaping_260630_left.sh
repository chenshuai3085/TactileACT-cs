#!/usr/bin/env bash
set -euo pipefail

# Task-local marker TactileVAE for 2026-06-30 huaping data.
# Only the 100 root episodes are used. The "无夹取位置变化" subdirectory is
# intentionally excluded because pretrain_tactile_vae.py reads only direct
# episode_*.hdf5 files from DATA_DIRS.

cd /home/chenshuai/Project/TactileACT-cs

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-512}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-2}
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

DATA_DIRS=(
  "/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630"
)
OUTPUT_DIR="/media/chenshuai/czy_data22/pih_output/tactile_vae_huaping_260630_left_tw8_ld16_s2_e150"
LOG_DIR="/media/chenshuai/czy_data22/pih_output/tactile_vae_logs"
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
data_excluded=/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630/无夹取位置变化
notes=Use only the 100 root episodes for the new huaping task. This VAE is intended for frozen-marker-latent tactile DP/foresight/guidance on huaping data.
EOF

{
  echo "=== run_tactile_vae_huaping_260630_left ==="
  echo "start_time=$(date '+%F %T')"
  echo "gpu=${GPU}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "data_dirs=${DATA_DIRS[*]}"
  echo "excluded=无夹取位置变化"
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
