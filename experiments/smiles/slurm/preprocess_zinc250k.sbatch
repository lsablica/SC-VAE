#!/bin/bash
#SBATCH --job-name=smiles-zinc-prep
#SBATCH --partition=medium
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="$HOME/projects/SC-VAE"
ENV_NAME="scvae-smiles"

DATA_ROOT="$HOME/projects/SC-VAE-data/zinc250k/raw"
PROCESSED_ROOT="$HOME/projects/SC-VAE-data/zinc250k/processed"

VALIDATION_FRACTION=0.1
TEST_FRACTION=0.1
SPLIT_SEED=13
MAX_SMILES_LENGTH=68

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
PYTHON_BIN="$(which python)"

cd "$REPO_ROOT"

echo "[slurm preprocess] dataset=zinc250k"
echo "[slurm preprocess] env_name=$ENV_NAME python=$PYTHON_BIN"
echo "[slurm preprocess] data_root=$DATA_ROOT"
echo "[slurm preprocess] processed_root=$PROCESSED_ROOT"
echo "[slurm preprocess] max_smiles_length=$MAX_SMILES_LENGTH"

"$PYTHON_BIN" -m experiments.smiles.preprocess \
  --data-root "$DATA_ROOT" \
  --processed-root "$PROCESSED_ROOT" \
  --validation-fraction "$VALIDATION_FRACTION" \
  --test-fraction "$TEST_FRACTION" \
  --split-seed "$SPLIT_SEED" \
  --max-smiles-length "$MAX_SMILES_LENGTH"
