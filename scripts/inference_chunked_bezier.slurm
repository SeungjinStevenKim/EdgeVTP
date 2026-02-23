#!/bin/bash
#SBATCH --job-name=inf_bez_chk
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/vehicle/slurm/inference_chunked_bezier_%j.out
#SBATCH --error=experiments/vehicle/slurm/inference_chunked_bezier_%j.err

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "=== Inference: Chunked Bezier (No Aug) ==="
python main.py --config configs/ngsim/inference_ngsim_30m_chunked_bezier_20ep.yaml

echo ""
echo "=== Inference: Chunked Bezier (Aug) ==="
python main.py --config configs/ngsim/inference_ngsim_30m_aug_chunked_bezier_20ep.yaml

echo "Done"
