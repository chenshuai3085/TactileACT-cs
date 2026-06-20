#!/usr/bin/env bash
set -euo pipefail

# This file is a copy-paste command sheet for board-wiping and insertion
# DP / tactile guidance tests. Running this file itself will not start any
# server. Open it, copy the block you need below, paste it into the terminal,
# and run that one command.

cat <<'EOF'
This file is now a command sheet, not a launcher.

Open it and copy only the command block you want:
  sed -n '1,260p' for_show_xiaomi/guide_forshow.sh

Main blocks:
  0. Common settings
  1. Board baseline server, port 8765
  2. Board s12 marker-joint guided server, port 8766
  3. Insertion baseline server, port 8785
  4. Insertion good-margin guided server, port 8786
  4b. Experimental board denoising-step guidance server, port 8768
  4c. Experimental insertion denoising-step guidance server, port 8788
  5. Preflight/status check
  6. Real rollout paired manifest
  7. Robot client commands
  8. Board force-curve evaluation
  9. Insertion server-side rollout evaluation
  10. Unified TacQuality real-rollout evaluation
  11. Historical commands

Current board scorer note:
  Use block 2 with --arm marker_joint_s12_guided.
  It is the current board candidate after semantic-direction, protected
  DDPM-step, and serving-smoke checks.  Baseline and guided logs are saved
  under the same BOARD_FORCE_ROOT so force-curve evaluation can compare them.

Current insertion scorer note:
  Use block 4 with --arm good_margin_guided.
  It avoids the saturated p_good probability score and is the current insertion
  candidate after score-mode ablation and serving-smoke checks.
EOF

exit 0

###############################################################################
# 0. Common settings
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs

export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/guide_forshow

export BOARD_DP_RUN=/media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr
export BOARD_FORESIGHT_DIR=/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload
export BOARD_FORESIGHT_CKPT=${BOARD_FORESIGHT_DIR}/foresight_best.ckpt
export TACQUALITY_ROLLOUT_CONFIG=/home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json
export BOARD_ROLLOUT_CONFIG=${TACQUALITY_ROLLOUT_CONFIG}
export BOARD_FORCE_ROOT=/home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer

export INSERTION_DP_RUN=/home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210
export INSERTION_VAE=/home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt
export INSERTION_FORESIGHT_DIR=/home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401
export INSERTION_FORESIGHT_CKPT=${INSERTION_FORESIGHT_DIR}/foresight_best.ckpt
export INSERTION_ROLLOUT_CONFIG=${TACQUALITY_ROLLOUT_CONFIG}
export INSERTION_ROLLOUT_ROOT=/home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer

###############################################################################
# 1. Current recommended baseline: 260617-only DP best, no guidance, port 8765
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8765 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer \
  > /tmp/guide_forshow/260617_best_baseline_8765.log 2>&1 &

tail -f /tmp/guide_forshow/260617_best_baseline_8765.log

###############################################################################
# 2. Current recommended guided: same 260617-only DP best + s12
#    marker_joint_action ForceBandTacQualityEnergy guidance, port 8766.
#    This is the current board recommendation: --arm marker_joint_s12_guided.
# Board contact gate is enabled by default:
#   marker magnitude <= 1.8: skip guidance
#   marker magnitude >= 2.3: full guidance
#   between them: linearly scaled guidance
# Add --disable_contact_gate only for ablation.
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm marker_joint_s12_guided \
  --ckpt_dir /media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8766 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/260617_best_marker_joint_s12_guided_8766.log 2>&1 &

tail -f /tmp/guide_forshow/260617_best_marker_joint_s12_guided_8766.log

###############################################################################
# 2b. Historical A/B guided arm: older marker_joint scorer, port 8767.
#     Use only when explicitly comparing older marker_joint against s12.
#     Logs intentionally go to a separate root so old-marker trajectories cannot
#     pollute the current s12 baseline/guided evaluation.
#     Smoke output:
#     /home/chenshuai/Project/output/tac_quality_guided_server_packet/board_260617_20260619_marker_joint_guided_smoke_20260619/guided_server_dry_run_smoke.json
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm marker_joint_guided \
  --ckpt_dir /media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8767 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --contact_gate_low 1.8 \
  --contact_gate_high 2.3 \
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_old_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/260617_best_marker_joint_old_guided_8767.log 2>&1 &

tail -f /tmp/guide_forshow/260617_best_marker_joint_old_guided_8767.log

###############################################################################
# 3. Current recommended insertion baseline: DP best/final, no guidance,
#    port 8785. The VAE override is required because the old DP config contains
#    an absolute TactileVAE path from another machine.  The Foresight path uses
#    matched latent_foresight_0401 evidence; do not use latent_foresight_full as
#    the default insertion real-test chain because it has a missing-key caveat.
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task insertion \
  --arm baseline \
  --disable_guidance \
  --ckpt_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210 \
  --ckpt_name dp_final.pth \
  --vae_checkpoint_override /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8785 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer \
  > /tmp/guide_forshow/insertion_baseline_8785.log 2>&1 &

tail -f /tmp/guide_forshow/insertion_baseline_8785.log

###############################################################################
# 4. Current recommended insertion guided: same DP + InsertionRiskScorerRuntime
#    good_margin final clean-action trust-region guidance, port 8786.  Uses
#    matched latent_foresight_0401 by default.
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task insertion \
  --arm good_margin_guided \
  --ckpt_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210 \
  --ckpt_name dp_final.pth \
  --vae_checkpoint_override /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8786 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --server_rollout_log_dir /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/insertion_good_margin_guided_8786.log 2>&1 &

tail -f /tmp/guide_forshow/insertion_good_margin_guided_8786.log

###############################################################################
# 4b. Experimental board denoising-step guidance: same board s12 scorer,
#     but TacQuality gradients are applied inside the last DDIM denoising step
#     on predicted clean action x0. Use only after the final-action baseline
#     and final-action guided commands above are working.
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task board \
  --arm marker_joint_s12_guided \
  --ckpt_dir /media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr \
  --ckpt_name dp_best.pth \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8768 \
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
  --server_rollout_log_dir /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_denoising_step_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/260617_best_marker_joint_s12_denoising_step_8768.log 2>&1 &

tail -f /tmp/guide_forshow/260617_best_marker_joint_s12_denoising_step_8768.log

###############################################################################
# 4c. Experimental insertion denoising-step guidance: same good_margin scorer,
#     but TacQuality gradients are applied inside the last DDIM denoising step
#     on predicted clean action x0.
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
CUDA_VISIBLE_DEVICES=0 nohup conda run --no-capture-output -n TactileACT python -u \
  -m for_show_xiaomi.serve_dp_tac_quality_guided \
  --task insertion \
  --arm good_margin_guided \
  --ckpt_dir /home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210 \
  --ckpt_name dp_final.pth \
  --vae_checkpoint_override /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt \
  --foresight_dir /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401 \
  --foresight_ckpt /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401/foresight_best.ckpt \
  --rollout_arm_config /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json \
  --host 0.0.0.0 \
  --port 8788 \
  --gpu 0 \
  --num_inference_steps 100 \
  --action_skip 0 \
  --action_horizon 8 \
  --guidance_location denoising_step \
  --ddpm_guidance_steps 1 \
  --ddpm_guidance_scale 0.001 \
  --ddpm_max_delta_norm 0.02 \
  --server_rollout_log_dir /home/chenshuai/Project/output/insertion_rollouts/good_margin_denoising_step_risk_scorer \
  --send_guidance_report \
  > /tmp/guide_forshow/insertion_good_margin_denoising_step_8788.log 2>&1 &

tail -f /tmp/guide_forshow/insertion_good_margin_denoising_step_8788.log

###############################################################################
# 5. Preflight/status check
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/preflight_tac_quality_deploy.py

test -s /media/chenshuai/EXTERNAL_USB/pih_output/dp_tac_concat_board_260617_only_left_boardvae_rawimg200x266_ph16_oh2_e2000_20260619_stable_fullwindow_slowlr/dp_best.pth
test -s /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_board_260609_260610_multistep16_boardvae_marker_only_e100_bs16_preload/foresight_best.ckpt
test -s /home/chenshuai/Project/output/tac_quality_rollout_arm_configs/tac_quality_rollout_arm_configs_current_s12_good_margin_20260619.json
test -s /home/chenshuai/Project/output/board_predicted_domain_force_band_energy_marker_joint_20260619_s12/force_band_tac_quality_energy_best.pt
test -s /home/chenshuai/Project/output/ckpt/dp_tac_concat_02090210/dp_final.pth
test -s /home/chenshuai/Project/output/tactile_vae_full/best_tactile_vae.pt
test -s /home/chenshuai/Project/output/foresight_ckpt/latent_foresight_0401/foresight_best.ckpt
test -s /home/chenshuai/Project/output/insertion_risk_scorer/insertion_risk_scorer_final.pt
mkdir -p /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer
mkdir -p /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer
ss -ltnp | grep -E ':8765|:8766|:8768|:8775|:8776|:8785|:8786|:8788' || true
pgrep -af 'serve_dp_tac_quality_guided|serve_board_dp_foresight_guided|serve_dp_policy' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

###############################################################################
# 6. Real rollout paired manifest
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/make_tac_quality_rollout_manifest.py \
  --output_dir /home/chenshuai/Project/output/tac_quality_real_rollout_manifest \
  --tag current_s12_good_margin_manifest \
  --tasks board,insertion \
  --board_pairs 3 \
  --insertion_pairs 3 \
  --order interleaved

sed -n '1,220p' /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.md

# Optional before all trials are complete: check how many planned rows already
# have matching server-side force_trace.csv. This should not be used as final
# evidence; it is only a bookkeeping dry run.
conda run --no-capture-output -n TactileACT python for_show_xiaomi/apply_rollout_manifest_metadata.py \
  --manifest_csv /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv \
  --dry_run \
  --allow_missing

###############################################################################
# 7. Robot client, run on robot/client machine
# The server saves one rollout directory for every wipe under:
#   /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer/baseline/
#   /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer/guided/
# and one rollout directory for every insertion episode under:
#   /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer/baseline/
#   /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer/guided/
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
GPU_SERVER_IP=127.0.0.1

# Baseline trials, connect to port 8765.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8765 \
  --disable_force_log

# Guided trials, connect to port 8766.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8766 \
  --disable_force_log

# Experimental board denoising-step guided trials, connect to port 8768.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8768 \
  --disable_force_log

# Insertion baseline trials, connect to port 8785.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8785 \
  --disable_force_log

# Insertion guided trials, connect to port 8786.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8786 \
  --disable_force_log

# Experimental insertion denoising-step guided trials, connect to port 8788.
python for_show_xiaomi/ws_client.py \
  --host ${GPU_SERVER_IP} \
  --port 8788 \
  --disable_force_log

###############################################################################
# 8. Board force-curve evaluation after real robot tests
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/apply_rollout_manifest_metadata.py \
  --manifest_csv /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv

conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_board_force_rollouts.py \
  --root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer \
  --tag board_260617_marker_joint_s12_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm marker_joint_s12_guided

###############################################################################
# 9. Insertion server-side rollout evaluation after robot tests
# This reads the force_trace.csv/metadata.json files saved by the GPU server.
# If metadata is missing, it still summarizes force/marker/action/guidance traces
# and writes a metadata_template.csv. Outcome claims require success/stopped
# labels to be filled for every trial.
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/apply_rollout_manifest_metadata.py \
  --manifest_csv /home/chenshuai/Project/output/tac_quality_real_rollout_manifest/current_s12_good_margin_manifest/tac_quality_rollout_manifest.csv

conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_insertion_rollouts.py \
  --root /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer \
  --output_dir /home/chenshuai/Project/output/insertion_rollout_eval \
  --tag insertion_good_margin_risk_scorer \
  --expected_baseline_arm baseline \
  --expected_guided_arm good_margin_guided

# After filling success/stopped_early/bounce_count/retry_count in the manifest
# CSV for insertion rows, re-run apply_rollout_manifest_metadata.py above and
# then rerun the evaluator. metadata_complete should become true.

###############################################################################
# 10. Unified TacQuality real-rollout evaluation after board + insertion tests
###############################################################################

cd /home/chenshuai/Project/TactileACT-cs
conda run --no-capture-output -n TactileACT python for_show_xiaomi/eval_tac_quality_real_rollouts.py \
  --board_root /home/chenshuai/Project/output/board_force_rollouts/260617_only_marker_joint_s12_scorer \
  --insertion_root /home/chenshuai/Project/output/insertion_rollouts/good_margin_risk_scorer \
  --output_dir /home/chenshuai/Project/output/tac_quality_real_rollout_eval \
  --tag current_s12_good_margin_tac_quality \
  --board_expected_baseline_arm baseline \
  --board_expected_guided_arm marker_joint_s12_guided \
  --insertion_expected_baseline_arm baseline \
  --insertion_expected_guided_arm good_margin_guided

###############################################################################
# 11. Historical commands from 2026-06-16 and 2026-06-17
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
