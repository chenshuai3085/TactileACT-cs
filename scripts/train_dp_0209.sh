#!/bin/bash
# Train Diffusion Policy on 0209-0210 dataset

python diffusion/train_dp_official.py \
    --dataset_dir /sharedata/chenshuai/data/dataset/02090210_success \
    --save_dir /sharedata/chenshuai/ckpt/dp_only_vision_drgger_02090210 \
    --camera_names global,wrist \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --epochs 600 \
    --batch_size 64 \
    --lr 1e-4 \
    --seed 1 \
    --save_freq 50 \
    --gpu 5