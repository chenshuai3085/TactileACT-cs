#!/bin/bash
set -euo pipefail

# Train deployment-compatible tactile DP concat policy only on the
# 2026-06-17 board-wiping collection.
# Images use the existing 200x266 fp16 cache. Tactile features use the
# board-trained TactileVAE as a frozen encoder.

DATASET_DIR="/media/chenshuai/EXTERNAL_USB/pih_dataset/260617_v8l_caheiban/peg_in_hole_0617"
BOARD_VAE="/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
SAVE_DIR="/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260621_codex"
IMAGE_CACHE_DIR="/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

cd /home/chenshuai/Project/TactileACT-cs
mkdir -p "${SAVE_DIR}"

{
    echo "=== train_dp_tac_concat_board_260617_only_e2000_20260621_codex ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "board_vae=${BOARD_VAE}"
    echo "save_dir=${SAVE_DIR}"
    echo "image_cache_dir=${IMAGE_CACHE_DIR}"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${SAVE_DIR}/run_command.txt"

exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

"${PYTHON_CMD[@]}" diffusion/train_dp_tac_concat.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --camera_names global,wrist \
    --proprio_key proprio_joint \
    --action_key actions/joint_abs \
    --tac_side left \
    --tac_history 8 \
    --vae_checkpoint "${BOARD_VAE}" \
    --vae_latent_dim 16 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --resize_shape 200,266 \
    --crop_shape 200,266 \
    --epochs 2000 \
    --batch_size 64 \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --warmup_steps 1000 \
    --num_train_timesteps 100 \
    --num_inference_steps 100 \
    --diffusion_step_embed_dim 128 \
    --down_dims 512,1024,2048 \
    --image_cache_dir "${IMAGE_CACHE_DIR}" \
    --num_workers 0 \
    --max_steps_per_epoch 128 \
    --max_val_windows 2048 \
    --val_ratio 0.1 \
    --val_interval 5 \
    --log_interval 20 \
    --save_freq 50 \
    --latest_freq 10 \
    --topk_k 3 \
    --seed 4 \
    --gpu 0
