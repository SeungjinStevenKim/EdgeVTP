#!/bin/bash
#SBATCH --job-name=bez_chk_20
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/vehicle/slurm/ngsim_chunked_bezier_20ep_%j.out
#SBATCH --error=experiments/vehicle/slurm/ngsim_chunked_bezier_20ep_%j.err

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "Starting training: Chunked Bezier (5-waypoint stages) 20ep"
python main.py --config configs/archive/ngsim/train_ngsim_30m_aug_chunked_bezier_20ep.yaml

echo "Training complete. Starting inference..."

# Create inference config dynamically
cp configs/archive/ngsim/train_ngsim_30m_aug_chunked_bezier_20ep.yaml configs/archive/ngsim/inference_ngsim_30m_aug_chunked_bezier_20ep.yaml
sed -i 's/train: true/train: false/g' configs/archive/ngsim/inference_ngsim_30m_aug_chunked_bezier_20ep.yaml
sed -i 's/run_name: .*/run_name: "inference_ngsim_30m_aug_chunked_bezier_20ep"/g' configs/archive/ngsim/inference_ngsim_30m_aug_chunked_bezier_20ep.yaml

python main.py --config configs/archive/ngsim/inference_ngsim_30m_aug_chunked_bezier_20ep.yaml

echo "Job complete"
