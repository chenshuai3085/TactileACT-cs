#!/bin/bash
# Comprehensive diversity + quality test for ACT (kl=0.01, kl=1.0) and DP
# Run after ACT kl1 training finishes (frees GPU memory)

set -e
DATASET=/home/chenshuai/data/dataset/260309_0310
EPISODES="50,150,250"
K=16

echo "============================================"
echo "  Diversity & Quality Tests"
echo "  $(date)"
echo "============================================"

# ACT kl=0.01, T=1.0
echo ""
echo ">>> ACT kl=0.01, T=1.0"
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 1.0

# ACT kl=0.01, T=3.0
echo ""
echo ">>> ACT kl=0.01, T=3.0"
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 3.0

# ACT kl=0.01, T=10.0
echo ""
echo ">>> ACT kl=0.01, T=10.0"
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 10.0

# ACT kl=1.0, T=1.0
echo ""
echo ">>> ACT kl=1.0, T=1.0"
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl1_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 1.0

# ACT kl=1.0, T=3.0
echo ""
echo ">>> ACT kl=1.0, T=3.0"
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl1_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 3.0

# DP (if checkpoint exists)
if [ -f /home/chenshuai/Project/output/dp_joint7/dp_best.pth ]; then
  echo ""
  echo ">>> DP joint7 (DDPM 100 steps)"
  conda run -n TactileACT python scripts/test_dp_diversity.py \
    --dp_dir /home/chenshuai/Project/output/dp_joint7 \
    --dataset_dir $DATASET --K $K --episodes $EPISODES
fi

echo ""
echo "============================================"
echo "  All tests done at $(date)"
echo "============================================"
