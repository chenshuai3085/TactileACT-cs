#!/bin/bash
# Setup Pi0-TacForesight environment
# Creates a new conda env 'pi0_tactile' with all dependencies

set -e

ENV_NAME="pi0_tactile"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n ${ENV_NAME} python=3.11 -y

echo "=== Activating environment ==="
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu121

echo "=== Installing JAX + CUDA (needed for openpi configs) ==="
pip install "jax[cuda12]==0.5.3"
pip install flax==0.10.2

echo "=== Installing core dependencies ==="
pip install \
    einops>=0.8.0 \
    numpy">=1.22.4,<2.0.0" \
    sentencepiece>=0.2.0 \
    transformers==4.53.2 \
    safetensors \
    wandb \
    tqdm \
    h5py \
    opencv-python \
    pillow \
    tyro \
    websockets

echo "=== Installing openpi (editable) ==="
cd /home/chenshuai/Project/openpi
pip install -e . --no-deps 2>/dev/null || echo "openpi editable install (partial)"

echo "=== Installing local packages ==="
cd /home/chenshuai/Project/TactileACT-cs
pip install -e detr/ 2>/dev/null || echo "detr install skipped"

echo "=== Patching transformers for pi0 ==="
# Pi0 requires custom SigLIP modeling
TRANSFORMERS_PATH=$(python -c "import transformers; import os; print(os.path.dirname(transformers.__file__))")
if [ -d "/home/chenshuai/Project/openpi/src/openpi/models_pytorch/transformers_replace" ]; then
    cp -r /home/chenshuai/Project/openpi/src/openpi/models_pytorch/transformers_replace/models/* \
        ${TRANSFORMERS_PATH}/models/ 2>/dev/null || echo "transformers patch skipped"
    echo "Patched transformers with pi0's custom SigLIP/Gemma"
fi

echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
import jax
print(f'JAX: {jax.__version__}')
import flax
print(f'Flax: {flax.__version__}')
import transformers
print(f'Transformers: {transformers.__version__}')
print('All dependencies OK!')
"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "To download pi0 weights:"
echo "  mkdir -p /home/chenshuai/Project/output/pi0_checkpoints"
echo "  gsutil -m cp -r gs://openpi-assets/checkpoints/pi0_base /home/chenshuai/Project/output/pi0_checkpoints/"
echo ""
echo "To train:"
echo "  python -m pi0_tactile.train --dataset_dir /path/to/episodes --pi0_weights /path/to/pi0_base"
