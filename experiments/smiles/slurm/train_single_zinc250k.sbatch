#!/bin/bash
#SBATCH --job-name=smiles-zinc-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch train_single_zinc250k.sbatch <model-name> <seed>"
  echo "Example: sbatch train_single_zinc250k.sbatch spcauchy-128 0"
  exit 1
fi

MODEL_NAME="$1"
SEED="$2"

case "$MODEL_NAME" in
  spcauchy-128|gaussian-64|gaussian-128) ;;
  *)
    echo "Unsupported model name: $MODEL_NAME"
    exit 1
    ;;
esac

REPO_ROOT="$HOME/projects/SC-VAE"
ENV_NAME="scvae-smiles"

RUN_ID="section54_zinc250k_main"
DATA_ROOT="$HOME/projects/SC-VAE-data/zinc250k/raw"
PROCESSED_ROOT="$HOME/projects/SC-VAE-data/zinc250k/processed"
RUNS_ROOT="$HOME/projects/SC-VAE-runs-zinc"

EPOCHS=120
BATCH_SIZE=1024
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
NUM_WORKERS=8

AMP=0
BETA_START=0.0
BETA_TARGET=0.015
BETA_ZERO_EPOCHS=1
BETA_WARMUP_EPOCHS=20
SPCAUCHY_RHO_BIAS_INIT=0.0
GRAD_CLIP_NORM=1.0

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

echo "[slurm train] dataset=zinc250k model=$MODEL_NAME seed=$SEED run_id=$RUN_ID"
echo "[slurm train] env_name=$ENV_NAME python=$PYTHON_BIN"
echo "[slurm train] data_root=$DATA_ROOT"
echo "[slurm train] processed_root=$PROCESSED_ROOT"
echo "[slurm train] runs_root=$RUNS_ROOT"
echo "[slurm train] beta_start=$BETA_START beta_target=$BETA_TARGET beta_zero_epochs=$BETA_ZERO_EPOCHS beta_warmup_epochs=$BETA_WARMUP_EPOCHS rho_bias_init=$SPCAUCHY_RHO_BIAS_INIT grad_clip_norm=$GRAD_CLIP_NORM amp=$AMP"
echo "[slurm train] max_smiles_length=$MAX_SMILES_LENGTH"

CMD=(
  "$PYTHON_BIN" -m experiments.smiles.train
  --dataset-name zinc250k
  --model-name "$MODEL_NAME"
  --seed "$SEED"
  --device cuda
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --beta-start "$BETA_START"
  --beta-target "$BETA_TARGET"
  --beta-zero-epochs "$BETA_ZERO_EPOCHS"
  --beta-warmup-epochs "$BETA_WARMUP_EPOCHS"
  --spcauchy-rho-bias-init "$SPCAUCHY_RHO_BIAS_INIT"
  --grad-clip-norm "$GRAD_CLIP_NORM"
  --validation-fraction "$VALIDATION_FRACTION"
  --test-fraction "$TEST_FRACTION"
  --split-seed "$SPLIT_SEED"
  --max-smiles-length "$MAX_SMILES_LENGTH"
  --num-workers "$NUM_WORKERS"
  --data-root "$DATA_ROOT"
  --processed-root "$PROCESSED_ROOT"
  --runs-root "$RUNS_ROOT"
  --run-id "$RUN_ID"
)

if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp)
fi

"${CMD[@]}"
