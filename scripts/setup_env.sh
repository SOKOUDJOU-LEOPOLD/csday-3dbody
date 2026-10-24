#!/bin/bash
# Builds the sam_3d_body conda env exactly as third_party/sam-3d-body/INSTALL.md describes.
# Needs internet. Only extra choices made here: the PyTorch build (cu121 wheels) and the
# toolchain to compile detectron2 against it (nvcc 12.1 + gcc 11; CUDA 12.1 allows gcc <= 12).
#   Grace (HPRC): Lmod modules provide both (system gcc 8.5 is too old).
#   Workstation (e.g. Ubuntu 24.04 with only CUDA 13 / gcc 13): installed into the env via conda.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if type module &>/dev/null; then
  module purge
  module load Miniconda3/24.11.1 GCC/11.3.0 CUDA/12.1.1
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
fi

conda env list | grep -q '^sam_3d_body ' || conda create -n sam_3d_body python=3.11 -y
conda activate sam_3d_body

if ! type module &>/dev/null; then
  conda install -y -c nvidia/label/cuda-12.1.1 -c conda-forge cuda-toolkit gxx_linux-64=11 gcc_linux-64=11
  export CUDA_HOME="$CONDA_PREFIX" CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
fi

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub

# detectron2 imports pkg_resources, which setuptools >= 81 no longer ships
pip install "setuptools<81"

# T4 / RTX 6000 = 7.5, A100 = 8.0, H100 = 9.0
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.0;9.0"
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

test -f "$ROOT/checkpoints/sam-3d-body-dinov3/model.ckpt" ||
  hf download facebook/sam-3d-body-dinov3 --local-dir "$ROOT/checkpoints/sam-3d-body-dinov3"
echo "ENV READY"
