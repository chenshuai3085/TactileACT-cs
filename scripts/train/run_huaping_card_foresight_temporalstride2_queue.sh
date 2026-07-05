#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

GPU=${GPU:-0}
CONDA_ENV=${CONDA_ENV:-TactileACT}
LOG_DIR=${LOG_DIR:-/home/chenshuai/Project/output/foresight_logs}

echo "=== run_huaping_card_foresight_temporalstride2_queue ==="
echo "start_time=$(date '+%F %T')"
echo "gpu=${GPU}"
echo "scoring=not_trained_this_run; reuse_existing_scorer_temporarily"
echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || true)"

GPU="${GPU}" CONDA_ENV="${CONDA_ENV}" LOG_DIR="${LOG_DIR}" \
  bash scripts/train/train_foresight_huaping_temporalstride2.sh

GPU="${GPU}" CONDA_ENV="${CONDA_ENV}" LOG_DIR="${LOG_DIR}" \
  bash scripts/train/train_foresight_card_temporalstride2.sh
