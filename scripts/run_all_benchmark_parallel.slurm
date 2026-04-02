#!/bin/bash
# Submit 4 benchmark jobs in parallel (each uses 1 GPU)
# Total wall time: ~2-3 hours instead of 8-12 hours
#
# Usage: ./scripts/run_all_benchmark_parallel.slurm

cd /project/vtrajectory/models/VT_Former-main
CONFIGS="configs/archive/pedestrian"

echo "Submitting 4 parallel jobs..."

# Job 1: Original 10m (Aug)
J1=$(sbatch --parsable --job-name=ped_orig10m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/ped_orig10m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_pedestrian_original_10m_aug.yaml && python main.py --config $CONFIGS/inference_pedestrian_original_10m_aug.yaml")
echo "  Pedestrian Original 10m (Aug): job $J1"

# Job 2: KAN 10m (Aug)
J2=$(sbatch --parsable --job-name=ped_kan10m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/ped_kan10m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_pedestrian_kan_10m_aug.yaml && python main.py --config $CONFIGS/inference_pedestrian_kan_10m_aug.yaml")
echo "  Pedestrian KAN 10m (Aug): job $J2"

# Job 3: KAN 15m (Aug)
J3=$(sbatch --parsable --job-name=ped_kan15m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/ped_kan15m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_pedestrian_kan_15m_aug.yaml && python main.py --config $CONFIGS/inference_pedestrian_kan_15m_aug.yaml")
echo "  Pedestrian KAN 15m (Aug): job $J3"

# Job 4: Mamba 10m (Aug)
J4=$(sbatch --parsable --job-name=ped_mamba10m --account=vtrajectory --partition=mb-l40s --gres=gpu:1 --time=04:00:00 --mem=32G --cpus-per-task=8 \
  --output=experiments/pedestrian/slurm/ped_mamba10m_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_pedestrian_mamba_10m_aug.yaml && python main.py --config $CONFIGS/inference_pedestrian_mamba_10m_aug.yaml")
echo "  Mamba 10m: job $J4"

echo ""
echo "All 4 jobs submitted. Run in parallel on 4 GPUs."
echo "Check: squeue -u \$USER"
echo "Logs: experiments/pedestrian/slurm/{ped_orig10m,ped_kan10m,ped_kan15m,ped_mamba10m}_<jobid>.out"
