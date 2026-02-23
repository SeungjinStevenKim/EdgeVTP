#!/bin/bash
# Submit 4 benchmark jobs in parallel (each uses 1 GPU)
# Total wall time: ~2-3 hours instead of 8-12 hours
#
# Usage: ./scripts/run_all_benchmark_parallel.slurm

cd /project/vtrajectory/models/VT_Former-main
CONFIGS="configs/pedestrian/local"

echo "Submitting 4 parallel jobs..."

# Job 1: Original 10m
J1=$(sbatch --parsable --job-name=orig10m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/orig10m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_original_10m.yaml && python main.py --config $CONFIGS/inference_original_10m.yaml")
echo "  Original 10m: job $J1"

# Job 2: KAN 10m
J2=$(sbatch --parsable --job-name=kan10m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/kan10m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_kan_10m.yaml && python main.py --config $CONFIGS/inference_kan_10m.yaml")
echo "  KAN 10m: job $J2"

# Job 3: KAN 15m
J3=$(sbatch --parsable --job-name=kan15m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/kan15m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_kan_15m.yaml && python main.py --config $CONFIGS/inference_kan_15m.yaml")
echo "  KAN 15m: job $J3"

# Job 4: Mamba 10m
J4=$(sbatch --parsable --job-name=mamba10m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/mamba10m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_mamba_10m.yaml && python main.py --config $CONFIGS/inference_mamba_10m.yaml")
echo "  Mamba 10m: job $J4"

echo ""
echo "All 4 jobs submitted. Run in parallel on 4 GPUs."
echo "Check: squeue -u \$USER"
echo "Logs: experiments/pedestrian/slurm/{orig10m,kan10m,kan15m,mamba10m}_<jobid>.out"
