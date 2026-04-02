#!/bin/bash
# Run residual training locally (20ep)
# Usage: conda activate vt_former && ./scripts/run_local_residual.sh

set -e
cd "$(dirname "$0")/.."

python main.py --config configs/archive/ngsim/train_ngsim_30m_oneshot_bezier_20ep_residual.yaml
