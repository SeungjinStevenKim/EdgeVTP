#!/bin/bash
#SBATCH --job-name=rot5_40
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/vehicle/slurm/ngsim_rot5_40ep_%j.out
#SBATCH --error=experiments/vehicle/slurm/ngsim_rot5_40ep_%j.err

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "Starting training: 30m filter + scale_only + 5deg rotation (40ep)"
python main.py --config configs/archive/ngsim/train_ngsim_30m_aug_rot5_40ep.yaml

echo "Training complete. Starting inference..."

# Create inference config dynamically
cp configs/archive/ngsim/train_ngsim_30m_aug_rot5_40ep.yaml configs/archive/ngsim/inference_ngsim_30m_aug_rot5_40ep.yaml
sed -i 's/train: true/train: false/g' configs/archive/ngsim/inference_ngsim_30m_aug_rot5_40ep.yaml
sed -i 's/run_name: .*/run_name: "inference_ngsim_30m_aug_rot5_40ep"/g' configs/archive/ngsim/inference_ngsim_30m_aug_rot5_40ep.yaml

python main.py --config configs/archive/ngsim/inference_ngsim_30m_aug_rot5_40ep.yaml

echo "Job complete"
