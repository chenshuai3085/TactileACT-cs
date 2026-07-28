#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONDA_ENV=${CONDA_ENV:-TactileACT}
GPU=${GPU:-0}
CKPT_DIR=${CKPT_DIR:-/home/chenshuai/Project/output/pi05_tactile_board}
PI0_WEIGHTS=${PI0_WEIGHTS:-}
SCORER_CKPT=${SCORER_CKPT:-}
TOKENIZER_BACKEND=${TOKENIZER_BACKEND:-ascii}

conda run --no-capture-output -n "${CONDA_ENV}" python -u -m pi0_tactile.serve \
  --ckpt_dir "${CKPT_DIR}" \
  --pi0_weights "${PI0_WEIGHTS}" \
  --pi05 \
  --action_dim 32 \
  --robot_action_dim 7 \
  --tokenizer_backend "${TOKENIZER_BACKEND}" \
  --gpu "${GPU}" \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8766}" \
  --num_flow_steps "${NUM_FLOW_STEPS:-10}" \
  --action_horizon_exec "${ACTION_HORIZON_EXEC:-8}" \
  --flow_guidance_steps "${FLOW_GUIDANCE_STEPS:-2}" \
  --flow_guidance_scale "${FLOW_GUIDANCE_SCALE:-0.02}" \
  --flow_guidance_max_total_delta "${FLOW_GUIDANCE_MAX_TOTAL_DELTA:-0.08}" \
  --flow_guidance_lambda_smooth "${FLOW_GUIDANCE_LAMBDA_SMOOTH:-0.0}" \
  --flow_guidance_scorer_ckpt "${SCORER_CKPT}" \
  --flow_guidance_scorer_runtime "${SCORER_RUNTIME:-force_band}" \
  --flow_guidance_score_mode "${SCORE_MODE:-energy_clipped}" \
  "$@"
