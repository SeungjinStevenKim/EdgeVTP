#!/bin/bash
#SBATCH --job-name=vtf_mirror
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/vehicle/slurm/ngsim_30m_aug_mirror_20ep_%j.out
#SBATCH --error=experiments/vehicle/slurm/ngsim_30m_aug_mirror_20ep_%j.err

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "=== Training: Original VT-Former (30M + Scale Aug + Mirroring) ==="
python main.py --config configs/ngsim/train_ngsim_30m_aug_mirror_20ep.yaml

echo ""
echo "=== Inference: Original VT-Former (30M + Scale Aug + Mirroring) ==="
python main.py --config configs/ngsim/inference_ngsim_30m_aug_mirror_20ep.yaml

echo "Done"
