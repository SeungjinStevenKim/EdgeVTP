#!/bin/bash
#SBATCH --job-name=oneshot_inf
#SBATCH --output=experiments/vehicle/slurm/oneshot_inference_%j.out
#SBATCH --error=experiments/vehicle/slurm/oneshot_inference_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=mb-l40s
#SBATCH -A vtrajectory

# Usage: sbatch --export=INFERENCE_CONFIG=configs/archive/ngsim/inference_ngsim_oneshot_bezier_20ep.yaml run_oneshot_bezier_inference.slurm

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

INFERENCE_CONFIG=${INFERENCE_CONFIG:-configs/archive/ngsim/inference_ngsim_oneshot_bezier_20ep.yaml}
echo "=== Inference: $INFERENCE_CONFIG ==="
python3 main.py --config "$INFERENCE_CONFIG"
echo "=== Done ==="
