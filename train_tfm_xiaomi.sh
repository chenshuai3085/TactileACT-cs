#!/bin/bash
# Train Tactile Foresight Model (TFM) — 基于预提取 DINOv2 特征
# 直接在 DINOv2 768-dim 空间预测未来触觉 embedding
# 前置: 先运行 precompute_dino_xiaomi.sh 提取 DINOv2 特征

python -m tactile_foresight.training.train_tfm \
  --feature_dir /home/chenshuai/data/dataset/260309_0310_dino_features \
  --save_dir /home/chenshuai/Project/output/tfm_xiaomi \
  --num_episodes 337 \
  --start_episode 0 \
  --camera_names global,wrist \
  --horizons 4,8,12 \
  --tac_mask_ratio 0.5 \
  --samples_per_episode 10 \
  --hidden_dim 512 \
  --num_layers 4 \
  --nheads 8 \
  --dropout 0.1 \
  --epochs 500 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --warmup_epochs 10 \
  --save_freq 50 \
  --log_freq 10 \
  --plot_freq 50 \
  --num_workers 4 \
  --seed 42
