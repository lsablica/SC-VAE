#!/bin/bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash experiments/smiles/slurm/submit_train_single.sh <model-name> <seed> [node-name]"
  echo "Example: bash experiments/smiles/slurm/submit_train_single.sh spcauchy-128 0"
  echo "Example: bash experiments/smiles/slurm/submit_train_single.sh spcauchy-128 0 clustergpu03"
  exit 1
fi

MODEL_NAME="$1"
SEED="$2"
NODE_NAME="${3:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=(sbatch --parsable)
if [[ -n "$NODE_NAME" ]]; then
  CMD+=(-w "$NODE_NAME")
fi
CMD+=("$SCRIPT_DIR/train_single_zinc250k.sbatch" "$MODEL_NAME" "$SEED")

JOB_ID="$("${CMD[@]}")"

echo "submitted job: $JOB_ID"
echo "model=$MODEL_NAME seed=$SEED amp=0"
if [[ -n "$NODE_NAME" ]]; then
  echo "node=$NODE_NAME"
fi
