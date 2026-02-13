#!/bin/bash
#SBATCH --job-name=vt_former_pedestrian_with_KAN_training
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-h100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out

cd /project/vtrajectory/models/VT_Former-main

source ~/.bashrc
# 용량이 넉넉한 새로운 경로의 환경 활성화
conda activate /project/vtrajectory/conda_envs/vt_former

# 라이브러리 경로 설정 (libcusparse 에러 방지용)
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

echo "Starting vt_former NGSIM inference at $(date)"

# NGSIM 설정 파일로 실행
python main.py --config configs/pedestrian/birds_eye.yaml

echo "Job finished at $(date)"
