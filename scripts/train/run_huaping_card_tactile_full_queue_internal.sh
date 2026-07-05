#!/usr/bin/env bash
set -euo pipefail

# Full single-GPU queue for the huaping/card tactile experiments.
# External drives are read-only data sources. All logs, VAE checkpoints, image
# caches, and DP checkpoints are written to the internal NVMe output root.

cd /home/chenshuai/Project/TactileACT-cs

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
QUEUE_LOG="${OUTPUT_ROOT}/huaping_card_tactile_full_queue_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "=== run_huaping_card_tactile_full_queue_internal ==="
echo "start_time=$(date '+%F %T')"
echo "output_root=${OUTPUT_ROOT}"
echo "queue_log=${QUEUE_LOG}"
echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
echo

echo "=== mount state ==="
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS /media/chenshuai/EXTERNAL_USB || true
findmnt -no TARGET,SOURCE,FSTYPE,OPTIONS /media/chenshuai/czy_data22 || true
df -h / "${OUTPUT_ROOT}" || true
echo

echo "$(date '+%F %T') starting huaping TactileVAE"
OUTPUT_ROOT="${OUTPUT_ROOT}" bash scripts/pretrain/run_tactile_vae_huaping_260630_left.sh

echo "$(date '+%F %T') starting card TactileVAE"
OUTPUT_ROOT="${OUTPUT_ROOT}" bash scripts/pretrain/run_tactile_vae_card_260615_260626_260629_260701_left.sh

HUAPING_VAE="${OUTPUT_ROOT}/tactile_vae_huaping_260630_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
CARD_VAE="${OUTPUT_ROOT}/tactile_vae_card_260615_260626_260629_260701_left_tw8_ld16_s2_e150/best_tactile_vae.pt"

if [[ ! -f "${HUAPING_VAE}" ]]; then
    echo "Missing huaping VAE after training: ${HUAPING_VAE}" >&2
    exit 1
fi
if [[ ! -f "${CARD_VAE}" ]]; then
    echo "Missing card VAE after training: ${CARD_VAE}" >&2
    exit 1
fi

echo "$(date '+%F %T') starting huaping tactile DP"
OUTPUT_ROOT="${OUTPUT_ROOT}" bash scripts/train/train_dp_tac_concat_huaping_260630_left_huapingvae_ph16_oh2_stride2_dynamic32768_e600.sh

echo "$(date '+%F %T') starting card tactile DP"
OUTPUT_ROOT="${OUTPUT_ROOT}" bash scripts/train/train_dp_tac_concat_card_positive_left_cardvae_ph16_oh2_stride2_dynamic32768_e600.sh

echo "$(date '+%F %T') queue complete"
