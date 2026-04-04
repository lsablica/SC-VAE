#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD=(sbatch --parsable "$SCRIPT_DIR/preprocess_zinc250k.sbatch")

JOB_ID="$("${CMD[@]}")"

echo "submitted job: $JOB_ID"
echo "job=preprocess"
