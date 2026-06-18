#!/usr/bin/env bash
set -euo pipefail

# This file is a copy-paste command sheet for board-wiping DP / tactile guidance
# tests.  Running this file itself will not start any server.  Open it, copy the
# block you need below, paste it into the terminal, and run that one command.

cat <<'EOF'
This file is now a command sheet, not a launcher.

Open it and copy only the command block you want:
  sed -n '1,260p' for_show_xiaomi/guide_forshow.sh

Main blocks:
  1. Current recommended baseline server, port 8765
  2. Current recommended PTG-guided server, port 8766
  3. Status check
  4. Robot client force logging
  5. Force-curve evaluation
  6. Historical commands
EOF

exit 0

###############################################################################
# 0. Common settings
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs

export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/guide_forshow

###############################################################################
# 1. Current recommended baseline: new DP, no guidance, port 8765
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_plus_peg0617_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  > /tmp/guide_forshow/new_plus_peg_ptg_baseline_8765.log 2>&1 &

tail -f /tmp/guide_forshow/new_plus_peg_ptg_baseline_8765.log

###############################################################################
# 2. Current recommended guided: same new DP + PTGProxy guidance, port 8766
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm default_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_plus_peg0617_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs.json \
  --host 0.0.0.0 \
  --port 8766 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --send_guidance_report \
  > /tmp/guide_forshow/new_plus_peg_ptg_guided_8766.log 2>&1 &

tail -f /tmp/guide_forshow/new_plus_peg_ptg_guided_8766.log

###############################################################################
# 3. Status check
###############################################################################

ss -ltnp | grep -E ':8765|:8766|:8775|:8776|:8785' || true
pgrep -af 'serve_dp_tac_quality_guided|serve_board_dp_foresight_guided|serve_dp_policy' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

###############################################################################
# 4. Robot client force logging, run on robot/client machine
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
TAG=board_newdp_ptgproxy_$(date +%Y%m%d_%H%M)
GPU_SERVER_IP=127.0.0.1

# Baseline trials, connect to port 8765.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8765 \
  --force_log_dir /home/chenshuai/Project/output/board_force_rollouts/${TAG}/baseline

# Guided trials, connect to port 8766.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8766 \
  --force_log_dir /home/chenshuai/Project/output/board_force_rollouts/${TAG}/guided

###############################################################################
# 5. Force-curve evaluation after real robot tests
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/${TAG} \
  --tag ${TAG}

###############################################################################
# 6. Historical commands from 2026-06-16 and 2026-06-17
###############################################################################

# Old full-data DP baseline, port 8766.
cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  for_show_xiaomi/serve_dp_policy.py \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_topk_ep504_loss0.0020.pth \
  --host 0.0.0.0 \
  --port 8766 \
  --gpu 0 \
  --num_inference_steps 20 \
  --action_skip 8 \
  --action_horizon 8 \
  --debug_dump_first_obs \
  --debug_dump_dir /home/chenshuai/Project/output/tactileact_dp_first_obs_full_ep504_skip8_steps20 \
  > /tmp/guide_forshow/old_full_baseline20_8766.log 2>&1 &

# Old full-data DP + BoardLatentEnergy guidance, port 8765.
cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  for_show_xiaomi/serve_board_dp_foresight_guided.py \
  --dp_ckpt /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000/dp_topk_ep504_loss0.0020.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --scorer_ckpt /home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/board_latent_energy_best.pt \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 20 \
  --guidance_steps 2 \
  --guidance_scale 0.001 \
  --guidance_path latent_only \
  --score_mode expert_margin \
  --alignment shift1 \
  --action_skip 8 \
  --action_horizon 8 \
  --send_guidance_report \
  > /tmp/guide_forshow/old_full_guided20_8765.log 2>&1 &

# Old positive-only DP baseline, port 8776.
cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  for_show_xiaomi/serve_dp_policy.py \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_positive_only_260609_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --host 0.0.0.0 \
  --port 8776 \
  --gpu 0 \
  --num_inference_steps 50 \
  --action_skip 8 \
  --action_horizon 8 \
  --debug_dump_first_obs \
  --debug_dump_dir /home/chenshuai/Project/output/tactileact_dp_first_obs_positive_best_skip8_steps50 \
  > /tmp/guide_forshow/old_pos_baseline50_8776.log 2>&1 &

# Old positive-only DP + BoardLatentEnergy guidance, port 8775.
cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  for_show_xiaomi/serve_board_dp_foresight_guided.py \
  --dp_ckpt /home/chenshuai/Project/output/dp_tac_concat_board_positive_only_260609_left_boardvae_rawimg200x266_ph16_oh2_e1000/dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --scorer_ckpt /home/chenshuai/Project/output/board_latent_energy/ce_margin_e10/board_latent_energy_best.pt \
  --host 0.0.0.0 \
  --port 8775 \
  --gpu 0 \
  --num_inference_steps 50 \
  --guidance_steps 2 \
  --guidance_scale 0.001 \
  --guidance_path latent_only \
  --score_mode expert_margin \
  --alignment shift1 \
  --action_skip 8 \
  --action_horizon 8 \
  --send_guidance_report \
  > /tmp/guide_forshow/old_pos_guided50_8775.log 2>&1 &

# Old full-data guided sweep commands:
#   port 8765: guidance_steps=10, guidance_scale=0.005
#   port 8766: guidance_steps=20, guidance_scale=0.005
#   port 8775: guidance_steps=10, guidance_scale=0.010
#   port 8776: guidance_steps=20, guidance_scale=0.010
# Use the old full-data guided command above and only change port,
# guidance_steps, guidance_scale, and log filename.
