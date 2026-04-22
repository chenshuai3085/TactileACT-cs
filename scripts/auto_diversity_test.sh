#!/bin/bash
# Auto diversity test: waits for ACT kl1 to finish, then runs all tests
# ACT kl1 PID=2857278, DP PID=3293916

DATASET=/home/chenshuai/data/dataset/260309_0310
EPISODES="50,150,250"
K=16
LOG=/home/chenshuai/Project/output/diversity_test_results.log

echo "Waiting for ACT kl1 training to finish (PID 2857278)..." | tee $LOG
while [ -d /proc/2857278 ]; do sleep 30; done
echo "ACT kl1 finished at $(date)" | tee -a $LOG

# Wait a moment for GPU memory to free
sleep 10

echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "  Starting Diversity Tests at $(date)" | tee -a $LOG
echo "============================================" | tee -a $LOG

# Test 1: ACT kl=0.01, T=1.0
echo "" | tee -a $LOG
echo ">>> Test 1: ACT kl=0.01, T=1.0" | tee -a $LOG
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 1.0 2>&1 | tee -a $LOG

# Test 2: ACT kl=0.01, T=3.0
echo "" | tee -a $LOG
echo ">>> Test 2: ACT kl=0.01, T=3.0" | tee -a $LOG
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 3.0 2>&1 | tee -a $LOG

# Test 3: ACT kl=0.01, T=10.0
echo "" | tee -a $LOG
echo ">>> Test 3: ACT kl=0.01, T=10.0" | tee -a $LOG
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl001_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 10.0 2>&1 | tee -a $LOG

# Test 4: ACT kl=1.0, T=1.0
echo "" | tee -a $LOG
echo ">>> Test 4: ACT kl=1.0, T=1.0" | tee -a $LOG
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl1_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 1.0 2>&1 | tee -a $LOG

# Test 5: ACT kl=1.0, T=3.0
echo "" | tee -a $LOG
echo ">>> Test 5: ACT kl=1.0, T=3.0" | tee -a $LOG
conda run -n TactileACT python scripts/test_dp_diversity.py \
  --act_dir /home/chenshuai/data/xiaomi_act/act_kl1_clip1099 \
  --dataset_dir $DATASET --K $K --episodes $EPISODES --T 3.0 2>&1 | tee -a $LOG

# Test 6: DP (uses dp_best.pth which should have good checkpoint by now)
echo "" | tee -a $LOG
echo ">>> Test 6: DP joint7 (DDPM 100 steps)" | tee -a $LOG
if [ -f /home/chenshuai/Project/output/dp_joint7/dp_best.pth ]; then
  conda run -n TactileACT python scripts/test_dp_diversity.py \
    --dp_dir /home/chenshuai/Project/output/dp_joint7 \
    --dataset_dir $DATASET --K $K --episodes $EPISODES 2>&1 | tee -a $LOG
else
  echo "DP checkpoint not found, skipping" | tee -a $LOG
fi

echo "" | tee -a $LOG
echo "============================================" | tee -a $LOG
echo "  All tests completed at $(date)" | tee -a $LOG
echo "============================================" | tee -a $LOG
