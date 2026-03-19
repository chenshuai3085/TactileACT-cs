#!/bin/bash
# Stage 2: Train Tactile Feasibility Score (TFS)
# Contrastive learning scorer: (obs + tactile, action) → feasibility score
# Requires: pre-trained VT alignment (Stage 0)

python -m tactile_foresight.training.train_tfs \
  --dataset_dir /home/chenshuai/data/dataset/260309_0310 \
  --save_dir /home/chenshuai/Project/output/tfs_xiaomi \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --horizons 4,8,12 \
  --chunk_size 20 \
  --proprio_key proprio_eef \
  --action_key actions/joint_abs \
  --tac_side left \
  --tac_img_key img \
  --tac_mask_ratio 0.0 \
  --alignment_ckpt /home/chenshuai/Project/output/vt_align_xiaomi/alignment_best.pth \
  --aligned_dim 256 \
  --score_dim 128 \
  --obs_hidden_dim 256 \
  --obs_num_layers 2 \
  --obs_nheads 4 \
  --action_dim 7 \
  --action_hidden_dim 256 \
  --dropout 0.1 \
  --noisy_action \
  --noise_steps 100 \
  --noise_ratio 0.5 \
  --epochs 1000 \
  --batch_size 32 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 20 \
  --save_freq 100 \
  --log_freq 10 \
  --plot_freq 50 \
  --num_workers 4 \
  --seed 42
