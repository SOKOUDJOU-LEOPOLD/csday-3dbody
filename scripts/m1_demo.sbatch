#!/bin/bash
#SBATCH --job-name=csday-m1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/m1_%j.log
# Milestone 1: upstream demo.py on a sample photo, once per detector, fully offline.
set -euo pipefail
ROOT=/scratch/user/lsokoudj/CS-Day/csday-3dbody
CK=$ROOT/checkpoints
module purge
module load Miniconda3/24.11.1 GCC/11.3.0 CUDA/12.1.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sam_3d_body
# compute nodes have no internet: HF weights and the DINOv3 torch.hub code are pre-cached on scratch
export HF_HUB_CACHE=$CK/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=$CK/torch_home
nvidia-smi --query-gpu=name --format=csv,noheader

cd $ROOT/third_party/sam-3d-body
for DET in vitdet sam3; do
  echo "=== detector: $DET"
  /usr/bin/time -f "wall %e s" python demo.py \
    --image_folder notebook/images \
    --output_folder $ROOT/output/m1_$DET \
    --checkpoint_path $CK/sam-3d-body-dinov3/model.ckpt \
    --mhr_path $CK/sam-3d-body-dinov3/assets/mhr_model.pt \
    --detector_name $DET --detector_path $CK/vitdet \
    --fov_path $CK/moge-2-vitl-normal/model.pt
done
ls -la $ROOT/output/m1_*
