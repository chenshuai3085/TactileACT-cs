#!/usr/bin/env bash

# 擦黑板真机测试命令表。
# 这个文件只用于复制命令，不要直接 bash 执行整个文件。
#
# 使用方式：
# 1. 先确认旧服务端是否还在运行；如端口被占用，先手动 kill 旧进程。
# 2. 选择一个 DP 版本。
# 3. 在一个终端复制该版本的“基线服务端”命令，端口 8765。
# 4. 在另一个终端复制同版本的“引导服务端”命令，端口 8769。
# 5. 在机器人 / client 机器上分别连接 8765 和 8769 做擦黑板测试。
# 6. 测试完成后复制该版本的“力曲线评估”命令。
#
# 当前主推测试：
# - 先测 A：260609+260610+260617 混合数据 DP。
# - 再测 D：action_offset=6 DP；这个版本还在训练，确认 best 合理后再真机。
# - E 是仅正样本 DP，适合和全量数据版本对比。
#
# 端口约定：
# - 8765：不加引导的基线服务端。
# - 8769：force-aware TacQuality 梯度引导服务端。
#
# 快速查看当前服务端：
#   pgrep -af 'serve_dp_tac_quality_guided|serve_board_dp_foresight_guided|serve_dp_policy'
#   ss -ltnp | grep -E ':8765|:8769' || true

###############################################################################
# 通用 client 命令：在机器人 / client 机器上运行
###############################################################################

# 连接基线服务端，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
python for_show_xiaomi/ws_client.py \
  --host 127.0.0.1 \
  --port 8765 \
  --disable_force_log

# 连接引导服务端，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
python for_show_xiaomi/ws_client.py \
  --host 127.0.0.1 \
  --port 8769 \
  --disable_force_log

###############################################################################
# A. 推荐版本：260609 + 260610 + 260617 混合数据 DP，dp_best.pth
###############################################################################

# A1. 基线服务端：不加引导，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_plus_peg0617_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_plus_peg0617_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_plus_peg0617_dpbest_force_aware_scorer \
  > /tmp/guide_forshow/board_plus_peg0617_dpbest_baseline_8765.log 2>&1 &
tail -f /tmp/guide_forshow/board_plus_peg0617_dpbest_baseline_8765.log

# A2. 引导服务端：force-aware TacQuality 梯度引导，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_plus_peg0617_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm force_aware_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_plus_peg0617_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.01 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_plus_peg0617_dpbest_force_aware_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/board_plus_peg0617_dpbest_force_aware_guided_8769.log 2>&1 &
tail -f /tmp/guide_forshow/board_plus_peg0617_dpbest_force_aware_guided_8769.log

# A3. 测试完成后的力曲线评估。
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/board_plus_peg0617_dpbest_force_aware_scorer \
  --tag board_plus_peg0617_dpbest_force_aware_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm force_aware_guided

###############################################################################
# B. 260617-only DP，dp_best.pth
###############################################################################

# B1. 基线服务端：不加引导，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260617_only_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260617_only_dpbest_force_aware_scorer \
  > /tmp/guide_forshow/board_260617_only_dpbest_baseline_8765.log 2>&1 &
tail -f /tmp/guide_forshow/board_260617_only_dpbest_baseline_8765.log

# B2. 引导服务端：force-aware TacQuality 梯度引导，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260617_only_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm force_aware_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.01 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260617_only_dpbest_force_aware_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/board_260617_only_dpbest_force_aware_guided_8769.log 2>&1 &
tail -f /tmp/guide_forshow/board_260617_only_dpbest_force_aware_guided_8769.log

# B3. 测试完成后的力曲线评估。
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/board_260617_only_dpbest_force_aware_scorer \
  --tag board_260617_only_dpbest_force_aware_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm force_aware_guided

###############################################################################
# C. 260609 + 260610 四类数据 DP，dp_best.pth
###############################################################################

# C1. 基线服务端：不加引导，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_e1000_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_e1000_dpbest_force_aware_scorer \
  > /tmp/guide_forshow/board_260609_260610_e1000_dpbest_baseline_8765.log 2>&1 &
tail -f /tmp/guide_forshow/board_260609_260610_e1000_dpbest_baseline_8765.log

# C2. 引导服务端：force-aware TacQuality 梯度引导，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_e1000_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm force_aware_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.01 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_e1000_dpbest_force_aware_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/board_260609_260610_e1000_dpbest_force_aware_guided_8769.log 2>&1 &
tail -f /tmp/guide_forshow/board_260609_260610_e1000_dpbest_force_aware_guided_8769.log

# C3. 测试完成后的力曲线评估。
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_e1000_dpbest_force_aware_scorer \
  --tag board_260609_260610_e1000_dpbest_force_aware_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm force_aware_guided

###############################################################################
# D. 260609 + 260610 action_offset=6 DP，dp_best.pth
###############################################################################

# D1. 基线服务端：不加引导，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_action_offset6_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_action_offset6_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_action_offset6_dpbest_force_aware_scorer \
  > /tmp/guide_forshow/board_260609_260610_action_offset6_dpbest_baseline_8765.log 2>&1 &
tail -f /tmp/guide_forshow/board_260609_260610_action_offset6_dpbest_baseline_8765.log

# D2. 引导服务端：force-aware TacQuality 梯度引导，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_action_offset6_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm force_aware_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2_action_offset6_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.01 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_action_offset6_dpbest_force_aware_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/board_260609_260610_action_offset6_dpbest_force_aware_guided_8769.log 2>&1 &
tail -f /tmp/guide_forshow/board_260609_260610_action_offset6_dpbest_force_aware_guided_8769.log

# D3. 测试完成后的力曲线评估。
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_action_offset6_dpbest_force_aware_scorer \
  --tag board_260609_260610_action_offset6_dpbest_force_aware_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm force_aware_guided

###############################################################################
# E. 仅正样本 DP：positive-only 260609，dp_best.pth
###############################################################################

# E1. 基线服务端：不加引导，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_positive_only_260609_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_positive_only_260609_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_positive_only_260609_dpbest_force_aware_scorer \
  > /tmp/guide_forshow/board_positive_only_260609_dpbest_baseline_8765.log 2>&1 &
tail -f /tmp/guide_forshow/board_positive_only_260609_dpbest_baseline_8765.log

# E2. 引导服务端：force-aware TacQuality 梯度引导，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_positive_only_260609_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm force_aware_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_positive_only_260609_left_boardvae_rawimg200x266_ph16_oh2_e1000 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.01 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_positive_only_260609_dpbest_force_aware_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/board_positive_only_260609_dpbest_force_aware_guided_8769.log 2>&1 &
tail -f /tmp/guide_forshow/board_positive_only_260609_dpbest_force_aware_guided_8769.log

# E3. 测试完成后的力曲线评估。
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/board_positive_only_260609_dpbest_force_aware_scorer \
  --tag board_positive_only_260609_dpbest_force_aware_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm force_aware_guided

###############################################################################
# F. 早期快速训练版本：260609 + 260610 quick，dp_best.pth，只建议调试
###############################################################################

# F1. 基线服务端：不加引导，端口 8765。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_quick_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_quick_dpbest_force_aware_scorer \
  > /tmp/guide_forshow/board_260609_260610_quick_dpbest_baseline_8765.log 2>&1 &
tail -f /tmp/guide_forshow/board_260609_260610_quick_dpbest_baseline_8765.log

# F2. 引导服务端：force-aware TacQuality 梯度引导，端口 8769。
cd /home/chenshuai/Project/TactileACT-cs
mkdir -p /tmp/guide_forshow /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_quick_dpbest_force_aware_scorer
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm force_aware_guided \
  --ckpt_dir /home/chenshuai/Project/output/dp_tac_concat_board_260609_260610_left_boardvae_rawimg200x266_ph16_oh2 \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/v2_board_action_conditioned_h16_e100_20260820/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_forceaware_board_20260621.json \
  --host 0.0.0.0 \
  --port 8769 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.01 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_quick_dpbest_force_aware_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/board_260609_260610_quick_dpbest_force_aware_guided_8769.log 2>&1 &
tail -f /tmp/guide_forshow/board_260609_260610_quick_dpbest_force_aware_guided_8769.log

# F3. 测试完成后的力曲线评估。
cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/board_260609_260610_quick_dpbest_force_aware_scorer \
  --tag board_260609_260610_quick_dpbest_force_aware_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm force_aware_guided
