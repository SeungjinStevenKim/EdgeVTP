#!/bin/bash
# datasets 전체 + data-set(high_D, tar) → Glade campaign
# Usage: ./scripts/sync_datasets_to_campaign.sh
# Note: 2FA 입력 필요할 수 있음 (ncar-two-factor)

set -e
cd "$(dirname "$0")/.."

REMOTE="sekim@derecho.hpc.ucar.edu"
REMOTE_PATH="/glade/campaign/uwyo/wyom0239/VT-Former-Varient/datasets/"

echo "=== All datasets → Glade campaign ==="
echo "Target: $REMOTE:$REMOTE_PATH"
echo ""

# 1. VT_Former-main/datasets (ngsim, eth, hotel, univ, zara1, zara2, raw, Carolinas_*)
echo ">>> Syncing datasets (ngsim, pedestrian, Carolinas)..."
rsync -avz --progress -e ssh \
  datasets/ \
  "$REMOTE:$REMOTE_PATH"

# 2. data-set (high_D, test.tar, train.tar, val.tar)
echo ""
echo ">>> Syncing HighD (high_D, test.tar, train.tar, val.tar)..."
rsync -avz --progress -e ssh \
  ../../data-set/high_D \
  ../../data-set/test.tar \
  ../../data-set/train.tar \
  ../../data-set/val.tar \
  "$REMOTE:$REMOTE_PATH"

echo ""
echo "Done. All datasets synced to $REMOTE:$REMOTE_PATH"
