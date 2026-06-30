#!/bin/bash
set -euo pipefail

# Pure-vision official-aligned DP baseline for 2026-06-30 peg-in-hole data.
# Dataset root only contains the selected 1-100 episodes; the excluded
# "无夹取位置变化" subdirectory is not read by train_dp_official.py.
# temporal_stride=1 keeps dense action labels A(t), A(t+1), ..., A(t+15).

DATASET_DIR="/media/chenshuai/czy_data22/pih_dataset/260630_v8j_huaping/peg_in_hole_0630"
SAVE_DIR="/media/chenshuai/czy_data22/pih_output/dp_official_vision_pih_260630_huaping_ph16_oh2_stride1_val5_e600"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

cd /home/chenshuai/Project/TactileACT-cs
mkdir -p "${SAVE_DIR}"

{
    echo "=== train_dp_official_pih_260630_huaping_ph16_oh2_stride1_val5_e600 ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "excluded_subdir=${DATASET_DIR}/无夹取位置变化"
    echo "save_dir=${SAVE_DIR}"
    echo "pred_horizon=16"
    echo "obs_horizon=2"
    echo "n_action_steps=8"
    echo "temporal_stride=1"
    echo "val_ratio=0.05"
    echo "epochs=600"
    echo "early_stop=false"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${SAVE_DIR}/run_command.txt"

exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

"${PYTHON_CMD[@]}" diffusion/train_dp_official.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --camera_names global,wrist \
    --proprio_key proprio_joint \
    --action_key actions/joint_abs \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --temporal_stride 1 \
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
    --val_ratio 0.05 \
    --val_interval 5 \
    --seed 1 \
    --save_freq 50 \
    --gpu 0
