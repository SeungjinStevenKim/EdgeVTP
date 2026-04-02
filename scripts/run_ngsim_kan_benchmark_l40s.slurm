#!/bin/bash
# NGSIM KAN Benchmark: 4 variants x 3 epochs = 12 jobs, mb-l40s
# 1. KAN only - 20/40/80ep
# 2. KAN + 30m filter - 20/40/80ep
# 3. KAN + 30m filter + aug - 20/40/80ep
# 4. KAN + aug - 20/40/80ep
#
# Usage: bash scripts/run_ngsim_kan_benchmark_l40s.slurm

cd /project/vtrajectory/models/VT_Former-main
CONFIGS="configs/archive/ngsim"
mkdir -p experiments/vehicle/slurm
PARTITION="mb-l40s"

# L40S: KAN may be slower than MLP. 20ep~5h, 40ep~9h, 80ep~18h
TIME_20=05:00:00
TIME_40=09:00:00
TIME_80=18:00:00

echo "Submitting 12 NGSIM KAN benchmark jobs ($PARTITION)..."

# KAN only: 20, 40, 80ep
J1=$(sbatch --parsable --job-name=ngsim_kan20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_kan20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_20ep.yaml")
echo "  KAN 20ep: $J1"

J2=$(sbatch --parsable --job-name=ngsim_kan40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_kan40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_40ep.yaml")
echo "  KAN 40ep: $J2"

J3=$(sbatch --parsable --job-name=ngsim_kan80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_kan80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_80ep.yaml")
echo "  KAN 80ep: $J3"

# KAN + 30m: 20, 40, 80ep
J4=$(sbatch --parsable --job-name=ngsim_k30m20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_k30m20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_30m_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_30m_20ep.yaml")
echo "  KAN+30m 20ep: $J4"

J5=$(sbatch --parsable --job-name=ngsim_k30m40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_k30m40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_30m_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_30m_40ep.yaml")
echo "  KAN+30m 40ep: $J5"

J6=$(sbatch --parsable --job-name=ngsim_k30m80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_k30m80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_30m_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_30m_80ep.yaml")
echo "  KAN+30m 80ep: $J6"

# KAN + 30m + aug: 20, 40, 80ep
J7=$(sbatch --parsable --job-name=ngsim_k30a20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_k30a20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_30m_aug_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_30m_aug_20ep.yaml")
echo "  KAN+30m+aug 20ep: $J7"

J8=$(sbatch --parsable --job-name=ngsim_k30a40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_k30a40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_30m_aug_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_30m_aug_40ep.yaml")
echo "  KAN+30m+aug 40ep: $J8"

J9=$(sbatch --parsable --job-name=ngsim_k30a80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_k30a80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_30m_aug_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_30m_aug_80ep.yaml")
echo "  KAN+30m+aug 80ep: $J9"

# KAN + aug: 20, 40, 80ep
J10=$(sbatch --parsable --job-name=ngsim_kaug20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_kaug20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_aug_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_aug_20ep.yaml")
echo "  KAN+aug 20ep: $J10"

J11=$(sbatch --parsable --job-name=ngsim_kaug40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_kaug40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_aug_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_aug_40ep.yaml")
echo "  KAN+aug 40ep: $J11"

J12=$(sbatch --parsable --job-name=ngsim_kaug80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_kaug80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_kan_aug_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_kan_aug_80ep.yaml")
echo "  KAN+aug 80ep: $J12"

echo ""
echo "All 12 KAN jobs submitted. squeue -u \$USER"
echo "Results: experiments/vehicle/inference/inference_ngsim_kan_*/inference_results.txt"
