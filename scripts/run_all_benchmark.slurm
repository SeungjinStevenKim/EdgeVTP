#!/bin/bash
#SBATCH --job-name=vt_benchmark
#SBATCH --account=vtrajectory
#SBATCH --partition=mb-l40s
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=experiments/pedestrian/slurm/benchmark_%j.out

# Full benchmark: train + inference for all 4 model variants
# 1. Original VT-Former 10m
# 2. KAN VT-Former 10m
# 3. KAN VT-Former 15m
# 4. Mamba VT-Former 10m

cd /project/vtrajectory/models/VT_Former-main
mkdir -p experiments/pedestrian/slurm

source ~/.bashrc
conda activate /project/vtrajectory/conda_envs/vt_former
export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:$LD_LIBRARY_PATH

CONFIGS="configs/pedestrian/local"

echo "=============================================="
echo "VT-Former Full Benchmark - $(date)"
echo "=============================================="

# --- TRAIN ---
echo ""
echo "========== PHASE 1: TRAINING =========="

echo ""
echo ">>> 1/4 Train Original VT-Former 10m"
python main.py --config "$CONFIGS/train_original_10m.yaml"

echo ""
echo ">>> 2/4 Train KAN VT-Former 10m"
python main.py --config "$CONFIGS/train_kan_10m.yaml"

echo ""
echo ">>> 3/4 Train KAN VT-Former 15m"
python main.py --config "$CONFIGS/train_kan_15m.yaml"

echo ""
echo ">>> 4/4 Train Mamba VT-Former 10m"
python main.py --config "$CONFIGS/train_mamba_10m.yaml"

# --- INFERENCE ---
echo ""
echo "========== PHASE 2: INFERENCE =========="

echo ""
echo ">>> 1/4 Inference Original VT-Former 10m"
python main.py --config "$CONFIGS/inference_original_10m.yaml"

echo ""
echo ">>> 2/4 Inference KAN VT-Former 10m"
python main.py --config "$CONFIGS/inference_kan_10m.yaml"

echo ""
echo ">>> 3/4 Inference KAN VT-Former 15m"
python main.py --config "$CONFIGS/inference_kan_15m.yaml"

echo ""
echo ">>> 4/4 Inference Mamba VT-Former 10m"
python main.py --config "$CONFIGS/inference_mamba_10m.yaml"

echo ""
echo "=============================================="
echo "Benchmark complete. $(date)"
echo "Results: experiments/pedestrian/inference/"
echo "=============================================="
