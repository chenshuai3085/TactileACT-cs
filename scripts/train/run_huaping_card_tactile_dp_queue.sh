#!/bin/bash
set -euo pipefail

# Queue tactile DP jobs on the single local GPU.
# It waits for the card TactileVAE tmux session to finish, verifies both VAE
# checkpoints, then trains huaping tactile DP followed by card tactile DP.

cd /home/chenshuai/Project/TactileACT-cs

OUTPUT_ROOT=${OUTPUT_ROOT:-/home/chenshuai/Project/output/pih_tactile}
QUEUE_LOG="${OUTPUT_ROOT}/huaping_card_tactile_dp_queue_$(date +%Y%m%d_%H%M%S).log"
HUAPING_VAE="${OUTPUT_ROOT}/tactile_vae_huaping_260630_left_tw8_ld16_s2_e150/best_tactile_vae.pt"
CARD_VAE="${OUTPUT_ROOT}/tactile_vae_card_260615_260626_260629_260701_left_tw8_ld16_s2_e150/best_tactile_vae.pt"

mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "=== run_huaping_card_tactile_dp_queue ==="
echo "start_time=$(date '+%F %T')"
echo "output_root=${OUTPUT_ROOT}"
echo "queue_log=${QUEUE_LOG}"

while tmux has-session -t vae_card_260615_260626_260629_260701_left 2>/dev/null; do
    echo "$(date '+%F %T') waiting for card TactileVAE tmux session to finish..."
    sleep 60
done

if [[ ! -f "${HUAPING_VAE}" ]]; then
    echo "Missing huaping VAE: ${HUAPING_VAE}" >&2
    exit 1
fi
if [[ ! -f "${CARD_VAE}" ]]; then
    echo "Missing card VAE: ${CARD_VAE}" >&2
    exit 1
fi

echo "$(date '+%F %T') starting huaping tactile DP"
bash scripts/train/train_dp_tac_concat_huaping_260630_left_huapingvae_ph16_oh2_stride2_dynamic32768_e600.sh

echo "$(date '+%F %T') starting card tactile DP"
bash scripts/train/train_dp_tac_concat_card_positive_left_cardvae_ph16_oh2_stride2_dynamic32768_e600.sh

echo "$(date '+%F %T') queue complete"
