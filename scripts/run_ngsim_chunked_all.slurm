#!/bin/bash
# NGSIM Chunked: all variants (chunk only, 30m+chunk, aug+chunk, 30m+aug+chunk)
# Chunk only: 20, 40, 60, 80 ep
# 30m+chunk: 20, 40, 80 ep
# Aug+chunk: 20, 40, 80 ep
# 30m+aug+chunk: 20, 40, 80 ep
#
# Usage: bash scripts/run_ngsim_chunked_all.slurm

PARTITION=mb-l40s
CONFIGS=configs/ngsim
TIME_20=02:00:00
TIME_40=04:00:00
TIME_60=06:00:00
TIME_80=08:00:00

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/vehicle/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "Submitting NGSIM Chunked jobs ($PARTITION)..."

# 1. Chunk only (no filter, no aug): 20, 40, 60, 80 ep
J1=$(sbatch --parsable --job-name=chk_20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_chunked_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_chunked_20ep.yaml")
echo "  Chunk only 20ep: $J1"

J2=$(sbatch --parsable --job-name=chk_40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_chunked_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_chunked_40ep.yaml")
echo "  Chunk only 40ep: $J2"

J3=$(sbatch --parsable --job-name=chk_60 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_60 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_60_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_chunked_60ep.yaml && python main.py --config $CONFIGS/inference_ngsim_chunked_60ep.yaml")
echo "  Chunk only 60ep: $J3"

J4=$(sbatch --parsable --job-name=chk_80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_chunked_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_chunked_80ep.yaml")
echo "  Chunk only 80ep: $J4"

# 2. 30m filter + chunk: 20, 40, 80 ep
J5=$(sbatch --parsable --job-name=chk_30m20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_30m20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_chunked_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_chunked_20ep.yaml")
echo "  30m+Chunk 20ep: $J5"

J6=$(sbatch --parsable --job-name=chk_30m40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_30m40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_chunked_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_chunked_40ep.yaml")
echo "  30m+Chunk 40ep: $J6"

J7=$(sbatch --parsable --job-name=chk_30m80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_30m80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_chunked_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_chunked_80ep.yaml")
echo "  30m+Chunk 80ep: $J7"

# 3. Data aug + chunk: 20, 40, 80 ep
J8=$(sbatch --parsable --job-name=chk_aug20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_aug20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_aug_chunked_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_aug_chunked_20ep.yaml")
echo "  Aug+Chunk 20ep: $J8"

J9=$(sbatch --parsable --job-name=chk_aug40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_aug40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_aug_chunked_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_aug_chunked_40ep.yaml")
echo "  Aug+Chunk 40ep: $J9"

J10=$(sbatch --parsable --job-name=chk_aug80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_aug80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_aug_chunked_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_aug_chunked_80ep.yaml")
echo "  Aug+Chunk 80ep: $J10"

# 4. 30m filter + data aug + chunk: 20, 40, 80 ep
J11=$(sbatch --parsable --job-name=chk_30a20 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_20 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_30a20_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_aug_chunked_20ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_aug_chunked_20ep.yaml")
echo "  30m+Aug+Chunk 20ep: $J11"

J12=$(sbatch --parsable --job-name=chk_30a40 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_40 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_30a40_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_aug_chunked_40ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_aug_chunked_40ep.yaml")
echo "  30m+Aug+Chunk 40ep: $J12"

J13=$(sbatch --parsable --job-name=chk_30a80 --account=vtrajectory --partition=$PARTITION --gres=gpu:1 --time=$TIME_80 --mem=64G --cpus-per-task=16 \
  --output=experiments/vehicle/slurm/chk_30a80_%j.out \
  --wrap="cd /project/vtrajectory/models/VT_Former-main && source ~/.bashrc && conda activate /project/vtrajectory/conda_envs/vt_former && export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:\$LD_LIBRARY_PATH && python main.py --config $CONFIGS/train_ngsim_30m_aug_chunked_80ep.yaml && python main.py --config $CONFIGS/inference_ngsim_30m_aug_chunked_80ep.yaml")
echo "  30m+Aug+Chunk 80ep: $J13"

echo ""
echo "All 13 Chunked jobs submitted. squeue -u \$USER"
