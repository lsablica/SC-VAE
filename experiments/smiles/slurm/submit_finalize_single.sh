#!/bin/bash

set -euo pipefail

NODE_NAME="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=(sbatch --parsable)
if [[ -n "$NODE_NAME" ]]; then
  CMD+=(-w "$NODE_NAME")
fi
CMD+=("$SCRIPT_DIR/finalize_zinc250k.sbatch")

JOB_ID="$("${CMD[@]}")"

echo "submitted job: $JOB_ID"
echo "job=finalize amp=0"
if [[ -n "$NODE_NAME" ]]; then
  echo "node=$NODE_NAME"
fi
