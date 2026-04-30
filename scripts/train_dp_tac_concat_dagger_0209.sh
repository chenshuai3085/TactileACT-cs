#!/bin/bash
# Train DP + Frozen TactileVAE (concat) + DAgger β-decay
# 4-GPU DataParallel, total batch=256 (64/GPU), lr=2e-4 (sqrt-scaled)
# β: 1.0 → 0.6 over training (warmup 50 eps pure success)

python diffusion/train_dp_tac_concat_dagger.py \
    --success_dir /sharedata/chenshuai/data/dataset/02090210_success \
    --bounce_dir /sharedata/chenshuai/data/dataset/02090210_bounce \
    --save_dir /sharedata/chenshuai/ckpt/dp_tac_concat_dagger_02090210 \
    --camera_names global,wrist \
    --tac_side left \
    --tac_history 8 \
    --vae_checkpoint /sharedata/chenshuai/ckpt/TactileVae/best_tactile_vae.pt \
    --vae_latent_dim 16 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --beta_start 1.0 \
    --beta_end 0.6 \
    --beta_warmup_epochs 50 \
    --epochs 300 \
    --batch_size 256 \
    --lr 2e-4 \
    --warmup_steps 1000 \
    --seed 1 \
    --save_freq 50 \
    --gpu 0,1,2,3
