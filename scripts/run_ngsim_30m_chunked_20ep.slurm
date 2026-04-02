#!/bin/bash
#SBATCH --job-name=ngsim_chunked_20
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=experiments/vehicle/slurm/ngsim_chunked_20ep_%j.out

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "=== NGSIM Train 30m chunked 20ep ==="
python main.py --config configs/archive/ngsim/train_ngsim_30m_chunked_20ep.yaml

echo ""
echo "=== NGSIM Inference 30m chunked 20ep ==="
python main.py --config configs/archive/ngsim/inference_ngsim_30m_chunked_20ep.yaml

echo ""
echo "=== Done ==="
