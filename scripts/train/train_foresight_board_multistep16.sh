#!/usr/bin/env bash
set -euo pipefail

cd /home/chenshuai/Project/TactileACT-cs

CONFIG=${CONFIG:-TFAC_V5/config_pretrain_foresight_board_multistep16.json}
CONDA_ENV=${CONDA_ENV:-TactileACT}
GPU=${GPU:-0}
NAME_SUFFIX=${NAME_SUFFIX:-}
LOG_DIR=${LOG_DIR:-/home/chenshuai/Project/output/foresight_logs}

# Optional runtime overrides. Leave unset to use values from CONFIG.
NUM_EPOCHS=${NUM_EPOCHS:-}
BATCH_SIZE=${BATCH_SIZE:-}
LR=${LR:-}
PRELOAD=${PRELOAD:-}
NUM_WORKERS=${NUM_WORKERS:-}
DISK_NUM_WORKERS=${DISK_NUM_WORKERS:-}

mkdir -p "${LOG_DIR}"
RUN_ID=$(date +"%Y%m%d_%H%M%S")
TMP_CONFIG=$(mktemp "/tmp/foresight_board_multistep16_${RUN_ID}_XXXX.json")
LOG_FILE="${LOG_DIR}/foresight_board_multistep16_${RUN_ID}.log"

python - "${CONFIG}" "${TMP_CONFIG}" <<'PY'
import json
import os
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = json.load(f)

overrides = {
    "gpu": os.environ.get("GPU"),
    "num_epochs": os.environ.get("NUM_EPOCHS"),
    "batch_size": os.environ.get("BATCH_SIZE"),
    "lr": os.environ.get("LR"),
    "preload": os.environ.get("PRELOAD"),
    "num_workers": os.environ.get("NUM_WORKERS"),
    "disk_num_workers": os.environ.get("DISK_NUM_WORKERS"),
}

for key, value in overrides.items():
    if value in (None, ""):
        continue
    if key in ("gpu", "num_epochs", "batch_size", "disk_num_workers"):
        cfg[key] = int(value)
    elif key == "lr":
        cfg[key] = float(value)
    elif key == "preload":
        low = value.lower()
        if low in ("true", "1", "yes"):
            cfg[key] = True
        elif low in ("false", "0", "no"):
            cfg[key] = False
        else:
            cfg[key] = value
    else:
        cfg[key] = value

suffix = os.environ.get("NAME_SUFFIX", "")
if suffix:
    cfg["name"] = f'{cfg["name"]}_{suffix}'

with open(dst, "w") as f:
    json.dump(cfg, f, indent=4)

print(dst)
PY

echo "Config: ${TMP_CONFIG}"
echo "Log: ${LOG_FILE}"
echo "GPU: ${GPU}"

conda run -n "${CONDA_ENV}" python TFAC_V5/pretrain_latent_foresight_multistep.py \
  --config "${TMP_CONFIG}" 2>&1 | tee "${LOG_FILE}"
