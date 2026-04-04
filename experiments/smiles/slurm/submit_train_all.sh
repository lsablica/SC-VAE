#!/bin/bash

set -euo pipefail

NODE_NAME="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=(
  "spcauchy-128"
  "gaussian-64"
  "gaussian-128"
)
SEEDS=(0 1 2)

JOB_IDS=()

for MODEL_NAME in "${MODELS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    CMD=(sbatch --parsable)
    if [[ -n "$NODE_NAME" ]]; then
      CMD+=(-w "$NODE_NAME")
    fi
    CMD+=("$SCRIPT_DIR/train_single_zinc250k.sbatch" "$MODEL_NAME" "$SEED")

    JOB_ID="$("${CMD[@]}")"
    JOB_IDS+=("$JOB_ID")

    echo "submitted job: $JOB_ID"
    echo "model=$MODEL_NAME seed=$SEED amp=0"
    if [[ -n "$NODE_NAME" ]]; then
      echo "node=$NODE_NAME"
    fi
  done
done

echo "submitted ${#JOB_IDS[@]} training jobs total"
echo "job_ids=${JOB_IDS[*]}"
