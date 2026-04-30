#!/bin/bash
# Train Diffusion Policy with DAgger-style β-decay sampling
# β: 1.0 → 0.6 (warmup 50 epochs, then linear decay)
# Success (expert) + Bounce (recovery) data

python diffusion/train_dp_dagger.py \
    --success_dir /sharedata/chenshuai/data/dataset/02090210_success \
    --bounce_dir /sharedata/chenshuai/data/dataset/02090210_bounce \
    --save_dir /sharedata/chenshuai/ckpt/dp_dagger_02090210 \
    --camera_names global,wrist \
    --beta_start 1.0 \
    --beta_end 0.6 \
    --beta_warmup_epochs 50 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --epochs 600 \
    --batch_size 64 \
    --lr 1e-4 \
    --seed 1 \
    --save_freq 50 \
    --gpu 5
