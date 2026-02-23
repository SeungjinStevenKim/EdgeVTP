#!/bin/bash
#SBATCH --job-name=ngsim25m
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-h100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=experiments/vehicle/slurm/ngsim25m_%j.out

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "=== NGSIM Train (25m radius, data augmentation) ==="
python main.py --config configs/ngsim/train_ngsim_25m.yaml

echo ""
echo "=== NGSIM Inference ==="
python main.py --config configs/ngsim/inference_ngsim_25m.yaml

echo ""
echo "=== Done ==="
