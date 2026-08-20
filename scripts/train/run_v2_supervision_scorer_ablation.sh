#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/chenshuai/miniconda3/envs/TactileACT/bin/python
REPO=/home/chenshuai/Project/TactileACT-cs
FULL_VAE=/home/chenshuai/Project/output/tactile_vae_v2_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae_v2.pt
ABLATE_VAE=/home/chenshuai/Project/output/tactile_vae_v2_no_intensity_rank_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae_v2.pt
ROOT=/home/chenshuai/Project/output/v2_intensity_rank_scorer_ablation_20260820

COMMON=(
  --device cuda:0
  --seed 42
  --chunk_len 16
  --stride 8
  --temporal_stride 1
  --future_offset 1
  --epochs 40
  --batch_size 128
  --class_dir expert=/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/wipe_pos_straight_z124_125_150_20260609
  --class_dir pressure_too_small=/media/chenshuai/EXTERNAL_USB/pih_dataset/260609/z_too_high
  --class_dir pressure_too_large=/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_low
  --class_dir pressure_unstable=/media/chenshuai/EXTERNAL_USB/pih_dataset/260610/z_too_oscillate
)

cd "$REPO"
"$PYTHON" TFAC_V5/board_latent_energy/train.py \
  "${COMMON[@]}" --tactile_vae_ckpt "$FULL_VAE" --output_dir "$ROOT/full_v2"
"$PYTHON" TFAC_V5/board_latent_energy/train.py \
  "${COMMON[@]}" --tactile_vae_ckpt "$ABLATE_VAE" --output_dir "$ROOT/without_intensity_ranking"
