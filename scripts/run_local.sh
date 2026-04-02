#!/bin/bash
# Local run script - same as SLURM job but runs on current machine
# Usage: ./scripts/run_local.sh [config_path]
# Example: ./scripts/run_local.sh configs/archive/pedestrian/pedestrian_unified.yaml
#
# Output: experiments/pedestrian/slurm/local_<timestamp>.out (same format as cluster jobs)
# Results: experiments/pedestrian/train/<run_name>/ or inference/<run_name>/ + inference_results.txt

set -e
CONFIG="${1:-configs/archive/pedestrian/pedestrian_unified.yaml}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p experiments/pedestrian/slurm
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="experiments/pedestrian/slurm/local_${TIMESTAMP}.out"

echo "Running locally: config=$CONFIG"
echo "Log file: $LOG_FILE"
echo ""

# Env setup (cluster path or local conda)
if [ -d "/project/vtrajectory/conda_envs/vt_former" ]; then
    source ~/.bashrc 2>/dev/null || true
    conda activate /project/vtrajectory/conda_envs/vt_former
    export LD_LIBRARY_PATH=/project/vtrajectory/conda_envs/vt_former/lib:${LD_LIBRARY_PATH:-}
elif command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate vt_former 2>/dev/null || conda activate base
fi

# Run and tee to log file (stdout + file, same format as SLURM)
python main.py --config "$CONFIG" 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
