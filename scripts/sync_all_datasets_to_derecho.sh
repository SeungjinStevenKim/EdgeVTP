#!/bin/bash
# Pedestrian + NGSIM 데이터셋을 Derecho 서버로 전송
# Usage: ./scripts/sync_all_datasets_to_derecho.sh
# Note: 2FA 입력 필요할 수 있음 (ncar-two-factor)

set -e
cd "$(dirname "$0")/.."

REMOTE="sekim@derecho.hpc.ucar.edu"
REMOTE_PATH="~/VT-Former-Varient/datasets/"

echo "=== All datasets → Derecho ==="
echo "Target: $REMOTE:$REMOTE_PATH"
echo ""

# Pedestrian: eth, hotel, univ, zara1, zara2
echo ">>> Syncing pedestrian (eth, hotel, univ, zara1, zara2)..."
rsync -avz -e ssh \
  datasets/eth datasets/hotel datasets/univ datasets/zara1 datasets/zara2 \
  "$REMOTE:$REMOTE_PATH"

# NGSIM
echo ""
echo ">>> Syncing NGSIM..."
rsync -avz -e ssh \
  datasets/ngsim \
  "$REMOTE:$REMOTE_PATH"

echo ""
echo "Done. All datasets synced to $REMOTE:$REMOTE_PATH"
