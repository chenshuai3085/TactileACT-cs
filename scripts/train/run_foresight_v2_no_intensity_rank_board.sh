#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/chenshuai/miniconda3/envs/TactileACT/bin/python
REPO=/home/chenshuai/Project/TactileACT-cs
RUNNER=/home/chenshuai/Project/TactileACT-cs/TFAC_V5/run_v2_action_foresight.py

cd "$REPO"
exec "$PYTHON" "$RUNNER" \
  --config "$REPO/configs/foresight_v2_no_intensity_rank_board.json"
