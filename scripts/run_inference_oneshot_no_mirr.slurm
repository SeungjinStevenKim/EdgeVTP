#!/bin/bash
#SBATCH --job-name=oneshot_inf
#SBATCH --output=experiments/vehicle/slurm/oneshot_inf_%j.out
#SBATCH --error=experiments/vehicle/slurm/oneshot_inf_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=mb-l40s
#SBATCH -A vtrajectory

# Environment Setup
cd /project/vtrajectory/models/VT_Former-main
source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

# Inference - One-shot Bezier (30M filter + Aug)
python3 main.py --config configs/ngsim/inference_oneshot_fast.yaml
