#!/bin/bash
# NGSIM Benchmark: 4 variants x 3 epochs = 12 jobs, mb-l40s (15obs)
# 1. Original (no filter, no aug) - 20/40/80ep
# 2. 30m filter - 20/40/80ep
# 3. 30m filter + aug - 20/40/80ep
# 4. Aug only - 20/40/80ep
#
# Usage: bash scripts/run_ngsim_benchmark_l40s.slurm

cd /project/vtrajectory/models/VT_Former-main
CONFIGS="configs/ngsim"
mkdir -p experiments/vehicle/slurm
PARTITION="mb-l40s"

# L40S: H100보다 느림. 20ep~3-4h, 40ep~6-8h, 80ep~12-16h
TIME_20=05:00:00
TIME_40=09:00:00
TIME_80=18:00:00

echo "Submitting 12 NGSIM benchmark jobs ($PARTITION, 15obs)..."

# Original: 20, 40, 80ep
J1=$(sbatch --parsable --job-name=ngsim_orig20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_orig20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_original_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_original_20ep.yaml")
echo "  Original 20ep: $J1"

J2=$(sbatch --parsable --job-name=ngsim_orig40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_orig40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_original_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_original_40ep.yaml")
echo "  Original 40ep: $J2"

J3=$(sbatch --parsable --job-name=ngsim_orig80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_orig80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_original_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_original_80ep.yaml")
echo "  Original 80ep: $J3"

# 30m filter: 20, 40, 80ep
J4=$(sbatch --parsable --job-name=ngsim_30m20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_30m20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_20ep.yaml")
echo "  30m 20ep: $J4"

J5=$(sbatch --parsable --job-name=ngsim_30m40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_30m40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_40ep.yaml")
echo "  30m 40ep: $J5"

J6=$(sbatch --parsable --job-name=ngsim_30m80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_30m80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_80ep.yaml")
echo "  30m 80ep: $J6"

# 30m + aug: 20, 40, 80ep
J7=$(sbatch --parsable --job-name=ngsim_30a20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_30a20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_aug_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_aug_20ep.yaml")
echo "  30m+aug 20ep: $J7"

J8=$(sbatch --parsable --job-name=ngsim_30a40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_30a40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_aug_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_aug_40ep.yaml")
echo "  30m+aug 40ep: $J8"

J9=$(sbatch --parsable --job-name=ngsim_30a80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_30a80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_aug_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_aug_80ep.yaml")
echo "  30m+aug 80ep: $J9"

# Aug only: 20, 40, 80ep
J10=$(sbatch --parsable --job-name=ngsim_aug20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_aug20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_aug_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_aug_20ep.yaml")
echo "  Aug 20ep: $J10"

J11=$(sbatch --parsable --job-name=ngsim_aug40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_aug40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_aug_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_aug_40ep.yaml")
echo "  Aug 40ep: $J11"

J12=$(sbatch --parsable --job-name=ngsim_aug80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/ngsim_aug80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_aug_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_aug_80ep.yaml")
echo "  Aug 80ep: $J12"

echo ""
echo "All 12 jobs submitted. squeue -u \$USER"
echo "Results: experiments/vehicle/inference/inference_ngsim_*/*/inference_results.txt"
echo "Summary: python scripts/collect_ngsim_results.py"
