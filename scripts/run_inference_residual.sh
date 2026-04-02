#!/bin/bash
# Run residual inference (20ep) - conda vt_former 활성화 후 실행
# Usage: conda activate vt_former && ./scripts/run_inference_residual.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== Run: inference_ngsim_30m_oneshot_bezier_20ep_residual ==="
python main.py --config configs/archive/ngsim/inference_ngsim_30m_oneshot_bezier_20ep_residual.yaml
