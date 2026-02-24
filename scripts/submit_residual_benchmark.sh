#!/bin/bash
# Residual separation (Cx/Delta) benchmark: train 20ep, 40ep, 80ep then inference
# Usage: ./scripts/submit_residual_benchmark.sh

cd /project/vtrajectory/models/VT_Former-main
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p experiments/vehicle/slurm

echo "=== Submitting Residual (Cx/Delta) jobs ==="

# 1. Training jobs (parallel)
J20=$(sbatch --parsable --job-name=res_20ep --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=08:00:00 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/residual_20ep_%j.out \
  --error=experiments/vehicle/slurm/residual_20ep_%j.err \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config configs/ngsim/train_ngsim_30m_oneshot_bezier_20ep_residual.yaml")
echo "  Train 20ep: job $J20"

J40=$(sbatch --parsable --job-name=res_40ep --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=12:00:00 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/residual_40ep_%j.out \
  --error=experiments/vehicle/slurm/residual_40ep_%j.err \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config configs/ngsim/train_ngsim_30m_oneshot_bezier_40ep_residual.yaml")
echo "  Train 40ep: job $J40"

J80=$(sbatch --parsable --job-name=res_80ep --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=24:00:00 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/residual_80ep_%j.out \
  --error=experiments/vehicle/slurm/residual_80ep_%j.err \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config configs/ngsim/train_ngsim_30m_oneshot_bezier_80ep_residual.yaml")
echo "  Train 80ep: job $J80"

# 2. Inference jobs (after corresponding training completes)
I20=$(sbatch --parsable --dependency=afterok:$J20 --job-name=res_inf20 --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=02:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/vehicle/slurm/residual_inf20_%j.out \
  --error=experiments/vehicle/slurm/residual_inf20_%j.err \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config configs/ngsim/inference_ngsim_30m_oneshot_bezier_20ep_residual.yaml")
echo "  Inference 20ep (after train): job $I20"

I40=$(sbatch --parsable --dependency=afterok:$J40 --job-name=res_inf40 --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=02:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/vehicle/slurm/residual_inf40_%j.out \
  --error=experiments/vehicle/slurm/residual_inf40_%j.err \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config configs/ngsim/inference_ngsim_30m_oneshot_bezier_40ep_residual.yaml")
echo "  Inference 40ep (after train): job $I40"

I80=$(sbatch --parsable --dependency=afterok:$J80 --job-name=res_inf80 --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=02:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/vehicle/slurm/residual_inf80_%j.out \
  --error=experiments/vehicle/slurm/residual_inf80_%j.err \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config configs/ngsim/inference_ngsim_30m_oneshot_bezier_80ep_residual.yaml")
echo "  Inference 80ep (after train): job $I80"

echo ""
echo "All 6 jobs submitted (3 train + 3 inference with dependencies)."
echo "Check: squeue -u \$USER"
echo "Logs: experiments/vehicle/slurm/residual_*.out"
