#!/usr/bin/env bash
set -euo pipefail

# Card tactile+vision DP for the 260707 v8j card dataset.
# User requested: no temporal skipping, tac_history=2, 600 epochs.

cd /home/chenshuai/Project/TactileACT-cs

DATA_ROOT="/media/chenshuai/czy_data22/pih_dataset/260707_v8j_card/peg_in_hole_0707"
DATASET_DIR="${DATA_ROOT}/success,${DATA_ROOT}/ercicharu"

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-600}
BATCH_SIZE=${BATCH_SIZE:-32}
VAE_EPOCHS=${VAE_EPOCHS:-150}
CARD_VAE=${CARD_VAE:-"${OUTPUT_ROOT}/tactile_vae_card_260707_left_tw2_ld16_s1_e${VAE_EPOCHS}/best_tactile_vae.pt"}
SAVE_DIR="${OUTPUT_ROOT}/dp_tac_concat_card_260707_left_cardvae_rawimg200x266_ph16_oh2_stride1_tachist2_cache_dynamic32768_e${EPOCHS}"
IMAGE_CACHE_DIR="${OUTPUT_ROOT}/cache/dp_tac_concat_card_260707_rawimg200x266_fp16"
PYTHON_CMD=(/home/chenshuai/miniconda3/envs/TactileACT/bin/python -u)

RESUME_ARGS=()
if [[ -f "${SAVE_DIR}/dp_latest.pth" ]]; then
    RESUME_ARGS=(--resume_checkpoint "${SAVE_DIR}/dp_latest.pth")
fi

mkdir -p "${SAVE_DIR}"

cat > "${SAVE_DIR}/README_task.txt" <<EOF
task=card_swipe
date=2026-07-07
policy=DP + frozen marker TactileVAE concat
data_used=${DATASET_DIR}
data_excluded=${DATA_ROOT}/pengzhuang has no direct episode_*.hdf5 at setup time; collision/failure data is not included in this positive policy run.
tactile_vae=${CARD_VAE}
output_root=${OUTPUT_ROOT}
temporal_stride=1
tac_history=2
pred_horizon=16
obs_horizon=2
n_action_steps=8
epochs=${EPOCHS}
batch_size=${BATCH_SIZE}
image_loading=internal image cache
image_cache_dir=${IMAGE_CACHE_DIR}
notes=Use dp_best.pth selected by validation loss for deployment/evaluation.
EOF

{
    echo "=== train_dp_tac_concat_card_260707_left_cardvae_ph16_oh2_stride1_tachist2_dynamic32768_e600 ==="
    echo "start_time=$(date '+%F %T')"
    echo "dataset_dir=${DATASET_DIR}"
    echo "card_vae=${CARD_VAE}"
    echo "save_dir=${SAVE_DIR}"
    echo "image_cache_dir=${IMAGE_CACHE_DIR}"
    echo "output_root=${OUTPUT_ROOT}"
    echo "pred_horizon=16"
    echo "obs_horizon=2"
    echo "n_action_steps=8"
    echo "action_offset=0"
    echo "temporal_stride=1"
    echo "tac_history=2"
    echo "dynamic_train_windows=32768"
    echo "max_val_windows=4096"
    echo "epochs=${EPOCHS}"
    echo "batch_size=${BATCH_SIZE}"
    echo "save_freq=50"
    echo "latest_freq=10"
    echo "topk_k=0"
    echo "image_loading=image_cache"
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
    --tac_history 2 \
    --vae_checkpoint "${CARD_VAE}" \
    --vae_latent_dim 16 \
    --pred_horizon 16 \
    --obs_horizon 2 \
    --n_action_steps 8 \
    --action_offset 0 \
    --temporal_stride 1 \
    "${RESUME_ARGS[@]}" \
    --resize_shape 200,266 \
    --crop_shape 200,266 \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
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
    --seed 7 \
    --gpu "${GPU}"
