#!/bin/bash
# Local run - vt_former env 활성화 후 실행
# Usage: conda activate vt_former && ./scripts/setup_local.sh

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== Run: train_ngsim_30m_oneshot_bezier_20ep_residual ==="
python main.py --config configs/ngsim/train_ngsim_30m_oneshot_bezier_20ep_residual.yaml
