#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/chenshuai/Project/TactileACT-cs"
PYTHON="/home/chenshuai/miniconda3/envs/TactileACT/bin/python"

cd "$ROOT"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec "$PYTHON" -u TFAC_V5/train_multitask_tacvae_5task.py \
  --manifest outputs/multitask_tacvae_5task_20260803/manifest.jsonl \
  --out_dir outputs/multitask_tacvae_5task_20260804/train \
  --epochs 300 \
  --steps_per_epoch 256 \
  --batch_size 500 \
  --eval_batch_size 2048 \
  --window_stride 2 \
  --val_window_stride 8 \
  --latent_channels 5 \
  --inr_hidden 64 \
  --kl_weight 1e-6 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --grad_clip 1.0 \
  --patience 30 \
  --seed 42 \
  --num_workers 4 \
  --eval_num_workers 4 \
  --device cuda
