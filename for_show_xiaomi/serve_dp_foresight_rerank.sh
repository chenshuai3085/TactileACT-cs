#!/bin/bash
# DP + Foresight Reranking Server
# Generates K candidates via batch DDPM/DDIM, scores with Foresight, selects gentlest contact.

cd /home/chenshuai/Project/TactileACT-cs

# Log directory for foresight predictions (set to skip logging)
LOG_DIR="/home/chenshuai/Project/output/foresight_logs/$(date +%Y%m%d_%H%M%S)"

python -m for_show_xiaomi.serve_dp_foresight_rerank \
    --ckpt_dir /home/chenshuai/Project/output/ckpt/dp_foresight_joint_0209 \
    --ckpt_name dp_topk_ep129_loss0.0029.pth  \
    --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0209 \
    --K 16 \
    --scheduler ddim \
    --num_inference_steps 50 \
    --action_skip 2 \
    --action_horizon 8 \
    --max_timesteps 300 \
    --port 8766 \
    --gpu 0 \
    --log_dir "$LOG_DIR"
