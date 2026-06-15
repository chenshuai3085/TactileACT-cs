#!/bin/bash
set -e

# Train tactile DP concat policy on board-wiping data.
# Uses the board-trained TactileVAE as a frozen tactile encoder.

DATASET_DIR="/home/chenshuai/data/dataset/260609/wipe_pos_straight_z124_125_150_20260609,/home/chenshuai/data/dataset/260609/z_too_high,/home/chenshuai/data/dataset/260610/z_too_low,/home/chenshuai/data/dataset/260610/z_too_oscillate"
BOARD_VAE="/home/chenshuai/Project/output/tactile_vae_board_260609_260610_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
SAVE_DIR="/home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2"
IMAGE_CACHE_DIR="/home/chenshuai/Project/output/cache/dp_board_rawimg200x266_fp16"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

cd /home/chenshuai/Project/TactileACT-cs

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
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --warmup_steps 500 \
    --num_train_timesteps 100 \
    --num_inference_steps 100 \
    --diffusion_step_embed_dim 128 \
    --down_dims 512,1024,2048 \
    --image_cache_dir "${IMAGE_CACHE_DIR}" \
    --build_image_cache \
    --num_workers 4 \
    --max_train_windows 8192 \
    --max_val_windows 1024 \
    --val_ratio 0.1 \
    --val_interval 2 \
    --log_interval 10 \
    --save_freq 2 \
    --topk_k 3 \
    --seed 1 \
    --gpu 0
