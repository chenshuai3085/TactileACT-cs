#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

conda run -n TactileACT python TFAC_V5/pretrain_latent_foresight_multistep.py \
  --config TFAC_V5/config_pretrain_foresight_board_multistep16.json
