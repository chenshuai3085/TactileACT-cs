#!/bin/bash
# DP + Foresight Joint Training (0209-0210 truncated data)
# Uses pretrained Foresight as initialization, adds auxiliary foresight loss to DP training.
# Run on server 172.16.0.109

cd /home/chenshuai/Project/TactileACT-cs

python diffusion/train_dp_foresight_joint.py \
    --dataset_dir /sharedata/chenshuai/data/dataset/0209-0210/truncated \
    --save_dir /sharedata/chenshuai/ckpt/dp_foresight_joint_0209 \
    --camera_names global,wrist \
    --proprio_key proprio_joint \
    --action_key actions/joint_abs \
    --tac_side left \
    --tac_history 8 \
    --vae_checkpoint /sharedata/chenshuai/ckpt/TactileVae/best_tactile_vae.pt \
    --vae_latent_dim 16 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --resize_shape 240,320 \
    --crop_shape 216,288 \
    --epochs 300 \
    --batch_size 128 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --warmup_steps 500 \
    --num_train_timesteps 100 \
    --diffusion_step_embed_dim 128 \
    --down_dims 512,1024,2048 \
    --seed 1 \
    --save_freq 50 \
    --gpu 0,1,2,3 \
    --foresight_ckpt /sharedata/chenshuai/ckpt/latent_foresight_0209/foresight_best.ckpt \
    --foresight_dir /sharedata/chenshuai/ckpt/latent_foresight_0209 \
    --lambda_foresight 0.1 \
    --foresight_horizon 10 \
    --foresight_warmup_epochs 10 \
    --foresight_t_threshold 50
