#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/chenshuai/miniconda3/envs/TactileACT/bin/python
REPO=/home/chenshuai/Project/TactileACT-cs
OUTPUT=/home/chenshuai/Project/output/tactile_vae_v2_no_intensity_rank_board_260609_260610_left_tw8_ld16_s2_e150

cd "$REPO"
exec "$PYTHON" TFAC_V5/pretrain_tactile_vae_v2.py \
  --data_dirs \
    /media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609 \
    /media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high \
    /media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low \
    /media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate \
  --output_dir "$OUTPUT" \
  --sides left \
  --latent_dim 16 \
  --temporal_window 8 \
  --decoder_hidden 128 \
  --decoder_heads 4 \
  --decoder_layers 2 \
  --kl_weight 1e-6 \
  --direction_weight 0.2 \
  --intensity_weight 0 \
  --rank_weight 0 \
  --epochs 150 \
  --batch_size 512 \
  --lr 1e-4 \
  --weight_decay 0 \
  --sample_stride 2 \
  --val_ratio 0.1 \
  --seed 42 \
  --save_every 50 \
  --vis_every 10 \
  --num_workers 4
