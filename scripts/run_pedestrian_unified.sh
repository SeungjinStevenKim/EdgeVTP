#!/bin/bash
#SBATCH --job-name=pedestrian_unified
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=experiments/pedestrian/slurm/%x_%j.out

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/pedestrian/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

python main.py --config configs/archive/pedestrian/pedestrian_unified.yaml
