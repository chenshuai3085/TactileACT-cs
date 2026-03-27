#!/bin/bash
# TFAC training for Xiaomi Realman dataset
# Think → Dream → Act: ACT + Foresight Transformer + Contrastive Learning
#
# Prerequisites:
#   1. CLIP pretraining done (clip_pretrain_xiaomi.sh)
#   2. meta_data.json at save_dir/meta_data.json
#   3. data symlinked at save_dir/data/

python TFAC/train.py --config TFAC/config_xiaomi.json
