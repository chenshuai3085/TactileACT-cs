#!/bin/bash
set -euo pipefail

# Board-wiping DP training with dynamic window sampling.
# Each epoch resamples 32768 windows from the full training window pool.
# This is much faster than full-window epochs while avoiding the old fixed-subset issue.

DATASET_DIR="/home/chenshuai/data/dataset/260609/wipe_pos_straight_z124_125_150_20260609,/home/chenshuai/data/dataset/260609/z_too_high,/home/chenshuai/data/dataset/260610/z_too_low,/home/chenshuai/data/dataset/260610/z_too_oscillate"
BOARD_VAE="/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
SAVE_DIR="/media/chenshuai/SANDISK ELE/tactileact_output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh4_dynamic32768_e1000"
IMAGE_CACHE_DIR="/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)
RESUME_ARGS=()
if [[ -f "${SAVE_DIR}/dp_latest.pth" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${SAVE_DIR}/dp_latest.pth")
fi

cd /home/chenshuai/Project/TactileACT-cs
mkdir -p "${SAVE_DIR}"

{
    echo "=== train_dp_tac_concat_board_260609_260610_obs4_ph16_dynamic32768_e1000 ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "board_vae=${BOARD_VAE}"
    echo "save_dir=${SAVE_DIR}"
    echo "image_cache_dir=${IMAGE_CACHE_DIR}"
    echo "pred_horizon=16"
    echo "obs_horizon=4"
    echo "action_offset=0"
    echo "dynamic_train_windows=32768"
    echo "max_train_windows=none"
    echo "max_val_windows=4096"
    echo "vision_aug=false"
    echo "vision_enhance=false"
    echo "epochs=1000"
    echo "save_freq=10"
    echo "latest_freq=10"
    echo "topk_k=0"
    echo "resume_checkpoint=${RESUME_ARGS[*]:-none}"
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
    --obs_horizon 4 \
    --n_action_steps 8 \
    --action_offset 0 \
    "${RESUME_ARGS[@]}" \
    --resize_shape 200,266 \
    --crop_shape 200,266 \
    --epochs 1000 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --warmup_steps 500 \
    --num_train_timesteps 100 \
    --num_inference_steps 100 \
    --diffusion_step_embed_dim 128 \
    --down_dims 512,1024,2048 \
    --image_cache_dir "${IMAGE_CACHE_DIR}" \
    --num_workers 4 \
    --dynamic_train_windows 32768 \
    --max_val_windows 4096 \
    --val_ratio 0.1 \
    --val_interval 5 \
    --log_interval 100 \
    --save_freq 10 \
    --latest_freq 10 \
    --topk_k 0 \
    --seed 3 \
    --gpu 0
