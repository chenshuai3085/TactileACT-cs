#!/bin/bash
set -euo pipefail

# Pure-vision official-aligned DP baseline for board wiping.
# temporal_stride=3 trains action labels A(t), A(t+3), ..., A(t+45)
# so the deployed policy moves faster than the dense stride-1 baseline.

DATA_ROOT="/mnt/robot-bionic-control-data/chenshuai18"
DATASET_DIR="${DATA_ROOT}/data/dataset/260609/wipe_pos_straight_z124_125_150_20260609,${DATA_ROOT}/data/dataset/260609/z_too_high,${DATA_ROOT}/data/dataset/260610/z_too_low,${DATA_ROOT}/data/dataset/260610/z_too_oscillate"
SAVE_DIR="${DATA_ROOT}/tactileact_output/dp_official_vision_board_260609_260610_ph16_oh2_temporalstride3_e600"

cd /root/vtm
mkdir -p "${SAVE_DIR}"

{
    echo "=== train_dp_official_board_260609_260610_ph16_oh2_temporalstride3_e600_server ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "save_dir=${SAVE_DIR}"
    echo "pred_horizon=16"
    echo "obs_horizon=2"
    echo "n_action_steps=8"
    echo "temporal_stride=3"
    echo "epochs=600"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${SAVE_DIR}/run_command.txt"

exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

python diffusion/train_dp_official.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --camera_names global,wrist \
    --proprio_key proprio_joint \
    --action_key actions/joint_abs \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --temporal_stride 3 \
    --resize_shape 240,320 \
    --crop_shape 216,288 \
    --epochs 600 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --warmup_steps 500 \
    --num_train_timesteps 100 \
    --num_inference_steps 100 \
    --diffusion_step_embed_dim 128 \
    --down_dims 512,1024,2048 \
    --seed 1 \
    --save_freq 50 \
    --gpu 0
