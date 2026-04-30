#!/bin/bash
# Train DP + Frozen TactileVAE (concat) on full 0209-0210 data (320 episodes)
# 4-GPU DataParallel training
# TactileVAE output (144-dim) concatenated to vision+qpos features

python diffusion/train_dp_tac_concat.py \
    --dataset_dir /sharedata/chenshuai/data/dataset/0209-0210 \
    --save_dir /sharedata/chenshuai/ckpt/dp_tac_concat_02090210 \
    --camera_names global,wrist \
    --tac_side left \
    --tac_history 8 \
    --vae_checkpoint /sharedata/chenshuai/ckpt/TactileVae/best_tactile_vae.pt \
    --vae_latent_dim 16 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --epochs 100 \
    --batch_size 256 \
    --lr 1e-4 \
    --seed 1 \
    --save_freq 50 \
    --gpu 0,1,2,3
