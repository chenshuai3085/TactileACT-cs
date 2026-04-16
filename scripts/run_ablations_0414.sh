#!/bin/bash
# Run ablation experiments sequentially on 0414 dataset
# Priority order: film (most promising) → gate_film → bypass → no_foresight
# Each experiment trains for 1000 epochs (~4 hours)
# Usage: nohup bash scripts/run_ablations_0414.sh &

set -e

BASE_DIR="/home/chenshuai/Project/TactileACT-cs"
cd $BASE_DIR

echo "=== TFAC 0414 Ablation Experiments ==="
echo "Starting at $(date)"

# Experiment 1: pure FiLM (memory at full strength + per-layer FiLM)
echo ""
echo "=== [1/4] film: Pure FiLM conditioning (no gate) ==="
echo "Config: TFAC/config_xiaomi_0414_film.json"
conda run -n TactileACT python -u TFAC/train.py \
    --config TFAC/config_xiaomi_0414_film.json 2>&1 | tee /tmp/film_0414.log
echo "film completed at $(date)"

# Experiment 2: gate + FiLM (gate fusion + per-layer FiLM)
echo ""
echo "=== [2/4] gate_film: Gate + FiLM conditioning ==="
echo "Config: TFAC/config_xiaomi_0414_gate_film.json"
conda run -n TactileACT python -u TFAC/train.py \
    --config TFAC/config_xiaomi_0414_gate_film.json 2>&1 | tee /tmp/gate_film_0414.log
echo "gate_film completed at $(date)"

# Experiment 3: bypass (foresight trained but NOT fused)
echo ""
echo "=== [3/4] bypass: Foresight trained but no fusion ==="
echo "Config: TFAC/config_xiaomi_0414_bypass.json"
conda run -n TactileACT python -u TFAC/train.py \
    --config TFAC/config_xiaomi_0414_bypass.json 2>&1 | tee /tmp/bypass_0414.log
echo "bypass completed at $(date)"

# Experiment 4: no_foresight (pure dual-decoder + a1_refine, no foresight at all)
echo ""
echo "=== [4/4] no_foresight: Pure dual-decoder, no foresight ==="
echo "Config: TFAC/config_xiaomi_0414_no_foresight.json"
conda run -n TactileACT python -u TFAC/train.py \
    --config TFAC/config_xiaomi_0414_no_foresight.json 2>&1 | tee /tmp/no_foresight_0414.log
echo "no_foresight completed at $(date)"

echo ""
echo "=== All experiments completed at $(date) ==="
echo "Run 'python scripts/compare_experiments.py' to see comparison"
