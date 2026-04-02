#!/bin/bash
#SBATCH --job-name=oneshot_fast
#SBATCH --output=experiments/vehicle/slurm/oneshot_fast_%j.out
#SBATCH --error=experiments/vehicle/slurm/oneshot_fast_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=mb-l40s
#SBATCH -A vtrajectory

# Environment Setup
cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm
source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

# 1. Training (LR 0.01, No Mirroring)
python3 main.py --config configs/archive/ngsim/train_ngsim_oneshot_bezier_20ep.yaml

# 2. Inference
python3 main.py --config configs/archive/ngsim/inference_ngsim_oneshot_bezier_20ep.yaml
