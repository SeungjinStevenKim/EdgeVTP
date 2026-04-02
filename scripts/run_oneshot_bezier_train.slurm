#!/bin/bash
#SBATCH --job-name=oneshot_tr
#SBATCH --output=experiments/vehicle/slurm/oneshot_train_%j.out
#SBATCH --error=experiments/vehicle/slurm/oneshot_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=mb-l40s
#SBATCH -A vtrajectory

# Usage: sbatch --export=TRAIN_CONFIG=configs/archive/ngsim/train_ngsim_oneshot_bezier_20ep.yaml run_oneshot_bezier_train.slurm

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

TRAIN_CONFIG=${TRAIN_CONFIG:-configs/archive/ngsim/train_ngsim_oneshot_bezier_20ep.yaml}
echo "=== Training: $TRAIN_CONFIG ==="
python3 main.py --config "$TRAIN_CONFIG"
echo "=== Done ==="
