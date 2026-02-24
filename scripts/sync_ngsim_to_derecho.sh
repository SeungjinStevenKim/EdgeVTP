#!/bin/bash
# NGSIM 데이터셋을 Derecho 서버로 전송
# Usage: ./scripts/sync_ngsim_to_derecho.sh
# Note: 2FA 입력 필요할 수 있음 (ncar-two-factor)

set -e
cd "$(dirname "$0")/.."

REMOTE="sekim@derecho.hpc.ucar.edu"
REMOTE_PATH="~/VT-Former-Varient/datasets/"

echo "=== NGSIM dataset → Derecho ==="
echo "Target: $REMOTE:$REMOTE_PATH"
echo "Size: ~410MB"
echo ""

rsync -avz -e ssh \
  datasets/ngsim \
  "$REMOTE:$REMOTE_PATH"

echo ""
echo "Done. NGSIM synced to $REMOTE:$REMOTE_PATH"
