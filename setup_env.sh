#!/bin/bash
# TFAC 环境一键配置脚本
# 目标: A800 服务器, CUDA 12.1, Python 3.8
# 用法: bash setup_env.sh

set -e

ENV_NAME="TactileACT"

echo "====== 1. 创建 conda 环境 ======"
conda create -n $ENV_NAME python=3.8 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo "====== 2. 安装 PyTorch (CUDA 12.1) ======"
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo "====== 3. 安装核心依赖 ======"
pip install \
    numpy==1.24.4 \
    h5py==3.11.0 \
    matplotlib==3.7.5 \
    scipy==1.10.1 \
    opencv-python==4.13.0.92 \
    einops==0.8.1 \
    tqdm==4.67.3 \
    pillow==10.4.0 \
    scikit-learn==1.3.2 \
    PyYAML==6.0.3 \
    diffusers==0.36.0 \
    imageio==2.35.1 \
    ipython==8.12.3

echo "====== 4. 安装 detr (本地包) ======"
# detr 是项目内的本地包, 需要在项目目录下安装
cd "$(dirname "$0")"
if [ -d "detr" ]; then
    pip install -e detr/
    echo "detr installed from local"
elif [ -f "detr/setup.py" ] || [ -f "detr/pyproject.toml" ]; then
    pip install -e detr/
    echo "detr installed from local"
else
    echo "WARNING: detr/ directory not found. Install manually after cloning the repo."
fi

echo "====== 5. 验证安装 ======"
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'CUDA version: {torch.version.cuda}')

import numpy, h5py, matplotlib, scipy, cv2, einops, tqdm, PIL, sklearn, diffusers
print('All core packages OK')
"

echo ""
echo "====== 配置完成 ======"
echo "激活环境: conda activate $ENV_NAME"
echo ""
echo "注意事项:"
echo "  1. 数据目录需要手动迁移或挂载 (/home/chenshuai/data/xiaomi_act)"
echo "  2. config 中的路径需要根据新机器调整 (save_dir, vision_backbone_path 等)"
echo "  3. 如果 A800 的 CUDA 版本不是 12.1, 修改上面的 PyTorch 安装命令"
echo "  4. 多卡训练需要额外配置 (当前代码为单卡)"
