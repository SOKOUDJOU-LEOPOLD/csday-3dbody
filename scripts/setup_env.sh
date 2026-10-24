#!/bin/bash
# Builds the sam_3d_body conda env exactly as third_party/sam-3d-body/INSTALL.md describes.
# Run on Grace (login node is fine; needs internet). Only extra choice made here: the
# PyTorch build (cu121 wheels) and CUDA arch list for detectron2 (T4/RTX 6000 = 7.5, A100 = 8.0).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
module purge
module load Miniconda3/24.11.1 CUDA/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0"
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

pip install git+https://github.com/microsoft/MoGe.git

# SAM 3 (optional detector)
if [ ! -d "$ROOT/third_party/sam3" ]; then
  git clone https://github.com/facebookresearch/sam3.git "$ROOT/third_party/sam3"
fi
pip install -e "$ROOT/third_party/sam3"
pip install decord psutil

# demo server deps (not part of upstream)
pip install fastapi "uvicorn[standard]" python-multipart trimesh

hf download facebook/sam-3d-body-dinov3 --local-dir "$ROOT/checkpoints/sam-3d-body-dinov3"
echo "ENV READY"
