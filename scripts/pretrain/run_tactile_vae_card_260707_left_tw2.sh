#!/usr/bin/env bash
set -euo pipefail

# Card-swipe marker TactileVAE for the 260707 v8j card dataset.
# The downstream policy requested by the user uses tac_history=2, so this VAE
# is trained with temporal_window=2.

cd /home/chenshuai/Project/TactileACT-cs

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-512}
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

DATA_ROOT="/media/chenshuai/czy_data22/pih_dataset/260707_v8j_card/peg_in_hole_0707"
DATA_DIRS=(
  "${DATA_ROOT}/success"
  "${DATA_ROOT}/ercicharu"
)

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
OUTPUT_DIR="${OUTPUT_ROOT}/tactile_vae_card_260707_left_tw2_ld16_s1_e${EPOCHS}"
LOG_DIR="${OUTPUT_ROOT}/tactile_vae_logs"
LOG_FILE="${LOG_DIR}/tactile_vae_card_260707_left_tw2_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

cat > "${OUTPUT_DIR}/README_task.txt" <<EOF
task=card_swipe
date=2026-07-07
model=TactileVAE marker_offset, old TFAC_V5/tactile_vae.py compatible
side=left
temporal_window=2
latent_dim=16
sample_stride=${SAMPLE_STRIDE}
epochs=${EPOCHS}
data_used=${DATA_DIRS[*]}
data_excluded=${DATA_ROOT}/pengzhuang has no direct episode_*.hdf5 at setup time; collision/failure data is not included in this positive policy/VAE run.
output_root=${OUTPUT_ROOT}
notes=Task-local card TactileVAE for downstream DP tac_history=2 and temporal_stride=1.
EOF

{
  echo "=== run_tactile_vae_card_260707_left_tw2 ==="
  echo "start_time=$(date '+%F %T')"
  echo "gpu=${GPU}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "output_root=${OUTPUT_ROOT}"
  echo "log_file=${LOG_FILE}"
  echo "data_dirs=${DATA_DIRS[*]}"
  echo "temporal_window=2"
  echo "sample_stride=${SAMPLE_STRIDE}"
  echo "epochs=${EPOCHS}"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_CMD[@]}" TFAC_V5/pretrain_tactile_vae.py \
  --data_dirs "${DATA_DIRS[@]}" \
  --sides left \
  --output_dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr 1e-4 \
  --temporal_window 2 \
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
