#!/usr/bin/env bash
# Run NGSIM inference for the three main paper operating points (K=16, 5 Hz).
# Requires datasets/ngsim/{train,val,test} and the matching ngsim.pt under each model_dir.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
MAIN="ngsim_share/main.py"
CONFIGS=(
  experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_k16_5hz/config.yaml
  experiments/vehicle/inference/inference_ngsim_20m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml
  experiments/vehicle/inference/inference_ngsim_30m_oneshot_bezier_80ep_residual_v2_tcn_5hz_k16/config.yaml
)
for cfg in "${CONFIGS[@]}"; do
  echo "=== $cfg ==="
  "$PYTHON" "$MAIN" --config "$cfg"
done
echo "Done. Metrics: experiments/vehicle/inference/<run_name>/inference_results.txt"
