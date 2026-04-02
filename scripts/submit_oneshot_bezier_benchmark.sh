#!/bin/bash
# One-shot Bezier benchmark: submit all train and inference jobs
# Run train jobs first. After all trains complete, run inference jobs.

cd /project/vtrajectory/models/VT_Former-main
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Train configs (30m_aug_20ep already has results, optional to skip)
TRAIN_CONFIGS=(
  "configs/archive/ngsim/train_ngsim_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/train_ngsim_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/train_ngsim_oneshot_bezier_80ep.yaml"
  "configs/archive/ngsim/train_ngsim_30m_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/train_ngsim_30m_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/train_ngsim_30m_oneshot_bezier_80ep.yaml"
  "configs/archive/ngsim/train_ngsim_aug_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/train_ngsim_aug_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/train_ngsim_aug_oneshot_bezier_80ep.yaml"
  # "configs/archive/ngsim/train_ngsim_30m_aug_oneshot_bezier_20ep.yaml"  # DONE - skip
  "configs/archive/ngsim/train_ngsim_30m_aug_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/train_ngsim_30m_aug_oneshot_bezier_80ep.yaml"
)

# Inference configs (all 12)
INFERENCE_CONFIGS=(
  "configs/archive/ngsim/inference_ngsim_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/inference_ngsim_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/inference_ngsim_oneshot_bezier_80ep.yaml"
  "configs/archive/ngsim/inference_ngsim_30m_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/inference_ngsim_30m_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/inference_ngsim_30m_oneshot_bezier_80ep.yaml"
  "configs/archive/ngsim/inference_ngsim_aug_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/inference_ngsim_aug_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/inference_ngsim_aug_oneshot_bezier_80ep.yaml"
  "configs/archive/ngsim/inference_ngsim_30m_aug_oneshot_bezier_20ep.yaml"
  "configs/archive/ngsim/inference_ngsim_30m_aug_oneshot_bezier_40ep.yaml"
  "configs/archive/ngsim/inference_ngsim_30m_aug_oneshot_bezier_80ep.yaml"
)

MODE="${1:-both}"  # train | inference | both

submit_train() {
  echo "Submitting train: $1"
  sbatch --export=ALL,TRAIN_CONFIG="$1" "$SCRIPT_DIR/run_oneshot_bezier_train.slurm"
}

submit_inference() {
  echo "Submitting inference: $1"
  sbatch --export=ALL,INFERENCE_CONFIG="$1" "$SCRIPT_DIR/run_oneshot_bezier_inference.slurm"
}

case "$MODE" in
  train)
    for c in "${TRAIN_CONFIGS[@]}"; do submit_train "$c"; done
    ;;
  inference)
    for c in "${INFERENCE_CONFIGS[@]}"; do submit_inference "$c"; done
    ;;
  both)
    for c in "${TRAIN_CONFIGS[@]}"; do submit_train "$c"; done
    echo ""
    echo "Train jobs submitted. Run inference after trains complete:"
    echo "  ./scripts/submit_oneshot_bezier_benchmark.sh inference"
    ;;
  *)
    echo "Usage: $0 {train|inference|both}"
    exit 1
    ;;
esac

echo "Done."
