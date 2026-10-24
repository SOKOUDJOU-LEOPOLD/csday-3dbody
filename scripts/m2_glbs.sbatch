#!/bin/bash
#SBATCH --job-name=csday-m2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/m2_%j.log
# Milestone 2: samples -> GLBs with server/model.py + server/mesh.py, once per detector.
set -euo pipefail
cd /scratch/user/lsokoudj/CS-Day/csday-3dbody
module purge
module load Miniconda3/24.11.1 GCC/11.3.0 CUDA/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_3d_body
nvidia-smi --query-gpu=name --format=csv,noheader

for DET in vitdet sam3; do
  echo "=== detector: $DET"
  DETECTOR=$DET python -m scripts.make_glbs --out output/m2_$DET || echo "FAILED: $DET"
done
