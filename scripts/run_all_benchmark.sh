#!/bin/bash
# Run full benchmark: train + inference for all model variants
# 1. Original VT-Former 10m
# 2. KAN VT-Former 10m
# 3. KAN VT-Former 15m
# 4. Mamba VT-Former 10m
#
# Usage: ./scripts/run_all_benchmark.sh
# Output: experiments/pedestrian/slurm/local_<timestamp>.out

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIGS_DIR="configs/archive/pedestrian"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiments/pedestrian/slurm/benchmark_all_${TIMESTAMP}.out"
mkdir -p experiments/pedestrian/slurm

# Env setup
if [ -d "/project/vtrajectory/conda_envs/vt_former" ]; then
    source ~/.bashrc 2>/dev/null || true
    conda activate /project/vtrajectory/conda_envs/vt_former
    export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:${LD_LIBRARY_PATH:-}
elif command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate vt_former 2>/dev/null || conda activate base
fi

exec 1> >(tee -a "$LOG_FILE") 2>&1

echo "=============================================="
echo "VT-Former Full Benchmark - $TIMESTAMP"
echo "=============================================="

# --- TRAIN ---
echo ""
echo "========== PHASE 1: TRAINING =========="

echo ""
echo ">>> 1/4 Train Pedestrian Original VT-Former 10m (Aug)"
python main.py --config "$CONFIGS_DIR/train_pedestrian_original_10m_aug.yaml"

echo ""
echo ">>> 2/4 Train Pedestrian KAN VT-Former 10m (Aug)"
python main.py --config "$CONFIGS_DIR/train_pedestrian_kan_10m_aug.yaml"

echo ""
echo ">>> 3/4 Train Pedestrian KAN VT-Former 15m (Aug)"
python main.py --config "$CONFIGS_DIR/train_pedestrian_kan_15m_aug.yaml"

echo ""
echo ">>> 4/4 Train Pedestrian Mamba VT-Former 10m (Aug)"
python main.py --config "$CONFIGS_DIR/train_pedestrian_mamba_10m_aug.yaml"

# --- INFERENCE ---
echo ""
echo "========== PHASE 2: INFERENCE =========="

echo ""
echo ">>> 1/4 Inference Pedestrian Original VT-Former 10m (Aug)"
python main.py --config "$CONFIGS_DIR/inference_pedestrian_original_10m_aug.yaml"

echo ""
echo ">>> 2/4 Inference Pedestrian KAN VT-Former 10m (Aug)"
python main.py --config "$CONFIGS_DIR/inference_pedestrian_kan_10m_aug.yaml"

echo ""
echo ">>> 3/4 Inference Pedestrian KAN VT-Former 15m (Aug)"
python main.py --config "$CONFIGS_DIR/inference_pedestrian_kan_15m_aug.yaml"

echo ""
echo ">>> 4/4 Inference Pedestrian Mamba VT-Former 10m (Aug)"
python main.py --config "$CONFIGS_DIR/inference_pedestrian_mamba_10m_aug.yaml"

echo ""
echo "=============================================="
echo "Benchmark complete. Log: $LOG_FILE"
echo "Results: experiments/pedestrian/inference/"
echo "=============================================="
