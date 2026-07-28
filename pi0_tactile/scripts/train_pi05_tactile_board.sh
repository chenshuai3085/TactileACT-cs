#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONDA_ENV=${CONDA_ENV:-TactileACT}
GPU=${GPU:-0}
DATASET_DIR=${DATASET_DIR:-/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617}
PI0_WEIGHTS=${PI0_WEIGHTS:-}
VAE_CHECKPOINT=${VAE_CHECKPOINT:-/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt}
FORESIGHT_CHECKPOINT=${FORESIGHT_CHECKPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-/home/chenshuai/Project/output/pi05_tactile_board}
TOKENIZER_BACKEND=${TOKENIZER_BACKEND:-ascii}

mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n "${CONDA_ENV}" python -u -m pi0_tactile.train \
  --dataset_dir "${DATASET_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --pi0_weights "${PI0_WEIGHTS}" \
  --vae_checkpoint "${VAE_CHECKPOINT}" \
  --foresight_checkpoint "${FORESIGHT_CHECKPOINT}" \
  --pi05 \
  --action_dim 32 \
  --robot_action_dim 7 \
  --action_horizon 20 \
  --camera_names base_0_rgb,left_wrist_0_rgb \
  --tokenizer_backend "${TOKENIZER_BACKEND}" \
  --tac_history 8 \
  --foresight_horizon 10 \
  --foresight_predict_horizon 1 \
  --lambda_foresight 0.1 \
  --foresight_t_threshold 0.3 \
  --foresight_warmup_steps 1000 \
  --batch_size "${BATCH_SIZE:-8}" \
  --num_train_steps "${NUM_TRAIN_STEPS:-30000}" \
  --lr "${LR:-2.5e-5}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --warmup_steps "${WARMUP_STEPS:-500}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --val_ratio "${VAL_RATIO:-0.1}" \
  --log_every "${LOG_EVERY:-50}" \
  --val_every "${VAL_EVERY:-500}" \
  --save_every "${SAVE_EVERY:-2000}" \
  --gpu "${GPU}" \
  "$@"
