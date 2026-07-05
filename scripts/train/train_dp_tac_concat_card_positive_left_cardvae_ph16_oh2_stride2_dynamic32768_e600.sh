#!/bin/bash
set -euo pipefail

# Card tactile+vision DP using the task-local card marker TactileVAE.
# Policy behavior cloning uses the user-selected card replay/success data only.
# Bounce folders and other card data remain useful for unsupervised VAE
# coverage, but are excluded from policy training so the policy does not imitate
# failure/recovery behavior.
# temporal_stride=2 targets roughly 200-300 executed high-level steps for
# 400-600 frame demonstrations.

DATASET_DIR="/media/chenshuai/EXTERNAL_USB/pih_dataset/260626_replay_card/card_pos_keepsteps_20260626,/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/peg_in_hole_0629/success"
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
CARD_VAE="${OUTPUT_ROOT}/tactile_vae_card_260615_260626_260629_260701_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
SAVE_DIR="${OUTPUT_ROOT}/dp_tac_concat_card_positive_left_cardvae_rawimg200x266_ph16_oh2_stride2_dynamic32768_e600"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)
RESUME_ARGS=()
if [[ -f "${SAVE_DIR}/dp_latest.pth" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${SAVE_DIR}/dp_latest.pth")
fi

cd /home/chenshuai/Project/TactileACT-cs
mkdir -p "${SAVE_DIR}"

cat > "${SAVE_DIR}/README_task.txt" <<EOF
task=card_swipe
date=2026-07-05
policy=DP + frozen marker TactileVAE concat
data_used=${DATASET_DIR}
data_excluded=260701 card data, 260615 card data, 260629 bounce folders, and nested huaping data under 260629_v8j_card are excluded from policy training.
tactile_vae=${CARD_VAE}
output_root=${OUTPUT_ROOT}
temporal_stride=2
expected_step_scale=400-600 frame demos become about 200-300 stride-2 policy steps
image_loading=lazy_images because existing cache pathing would collide for multiple parent directories named success
notes=Use dp_best.pth selected by validation loss for deployment/evaluation.
EOF

{
    echo "=== train_dp_tac_concat_card_positive_left_cardvae_ph16_oh2_stride2_dynamic32768_e600 ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "card_vae=${CARD_VAE}"
    echo "save_dir=${SAVE_DIR}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "pred_horizon=16"
    echo "obs_horizon=2"
    echo "action_offset=0"
    echo "temporal_stride=2"
    echo "dynamic_train_windows=32768"
    echo "max_val_windows=4096"
    echo "epochs=600"
    echo "save_freq=50"
    echo "latest_freq=10"
    echo "topk_k=0"
    echo "image_loading=lazy_images"
    echo "resume_checkpoint=${RESUME_ARGS[*]:-none}"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${SAVE_DIR}/run_command.txt"

exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

if [[ ! -f "${CARD_VAE}" ]]; then
    echo "Missing card TactileVAE checkpoint: ${CARD_VAE}" >&2
    exit 1
fi

"${PYTHON_CMD[@]}" diffusion/train_dp_tac_concat.py \
    --dataset_dir "${DATASET_DIR}" \
    --save_dir "${SAVE_DIR}" \
    --camera_names global,wrist \
    --proprio_key proprio_joint \
    --action_key actions/joint_abs \
    --tac_side left \
    --tac_history 8 \
    --vae_checkpoint "${CARD_VAE}" \
    --vae_latent_dim 16 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --action_offset 0 \
    --temporal_stride 2 \
    "${RESUME_ARGS[@]}" \
    --resize_shape 200,266 \
    --crop_shape 200,266 \
    --epochs 600 \
    --batch_size 64 \
    --lr 1e-4 \
    --weight_decay 1e-6 \
    --warmup_steps 500 \
    --num_train_timesteps 100 \
    --num_inference_steps 100 \
    --diffusion_step_embed_dim 128 \
    --down_dims 512,1024,2048 \
    --lazy_images \
    --num_workers 4 \
    --dynamic_train_windows 32768 \
    --max_val_windows 4096 \
    --val_ratio 0.1 \
    --val_interval 5 \
    --log_interval 100 \
    --save_freq 50 \
    --latest_freq 10 \
    --topk_k 0 \
    --seed 7 \
    --gpu 0
