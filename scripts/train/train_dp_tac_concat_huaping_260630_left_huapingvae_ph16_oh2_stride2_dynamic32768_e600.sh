#!/bin/bash
set -euo pipefail

# Huaping tactile+vision DP using a task-local huaping marker TactileVAE.
# Uses only the 100 root episodes from the EXTERNAL_USB huaping copy; the
# "无变化" subdirectory is not read.
# temporal_stride=2 targets roughly 200-300 executed high-level steps for
# 400-600 frame demonstrations.

DATASET_DIR="/media/chenshuai/EXTERNAL_USB/pih_dataset/260629_v8j_card/260630_v8j_huaping/peg_in_hole_0630"
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
HUAPING_VAE="${OUTPUT_ROOT}/tactile_vae_huaping_260630_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
SAVE_DIR="${OUTPUT_ROOT}/dp_tac_concat_huaping_260630_external_left_huapingvae_rawimg200x266_ph16_oh2_stride2_dynamic32768_e600"
IMAGE_CACHE_DIR="${OUTPUT_ROOT}/cache/dp_tac_concat_huaping_260630_external_rawimg200x266_fp16"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)
RESUME_ARGS=()
if [[ -f "${SAVE_DIR}/dp_latest.pth" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${SAVE_DIR}/dp_latest.pth")
fi

cd /home/chenshuai/Project/TactileACT-cs
mkdir -p "${SAVE_DIR}"

cat > "${SAVE_DIR}/README_task.txt" <<EOF
task=huaping
date=2026-07-05
policy=DP + frozen marker TactileVAE concat
data_used=${DATASET_DIR}
data_excluded=${DATASET_DIR}/无变化
tactile_vae=${HUAPING_VAE}
output_root=${OUTPUT_ROOT}
temporal_stride=2
expected_step_scale=400-600 frame demos become about 200-300 stride-2 policy steps
notes=Use dp_best.pth selected by validation loss for deployment/evaluation.
EOF

{
    echo "=== train_dp_tac_concat_huaping_260630_left_huapingvae_ph16_oh2_stride2_dynamic32768_e600 ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "huaping_vae=${HUAPING_VAE}"
    echo "save_dir=${SAVE_DIR}"
    echo "image_cache_dir=${IMAGE_CACHE_DIR}"
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
    echo "resume_checkpoint=${RESUME_ARGS[*]:-none}"
    echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
} | tee -a "${SAVE_DIR}/run_command.txt"

exec > >(tee -a "${SAVE_DIR}/train.log") 2>&1

if [[ ! -f "${HUAPING_VAE}" ]]; then
    echo "Missing huaping TactileVAE checkpoint: ${HUAPING_VAE}" >&2
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
    --vae_checkpoint "${HUAPING_VAE}" \
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
    --image_cache_dir "${IMAGE_CACHE_DIR}" \
    --build_image_cache \
    --num_workers 4 \
    --dynamic_train_windows 32768 \
    --max_val_windows 4096 \
    --val_ratio 0.1 \
    --val_interval 5 \
    --log_interval 100 \
    --save_freq 50 \
    --latest_freq 10 \
    --topk_k 0 \
    --seed 5 \
    --gpu 0
