#!/bin/bash
#SBATCH --job-name=smiles-zinc-finalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="$HOME/projects/SC-VAE"
ENV_NAME="scvae-smiles"

RUN_ID="section54_zinc250k_main"
DATA_ROOT="$HOME/projects/SC-VAE-data/zinc250k/raw"
PROCESSED_ROOT="$HOME/projects/SC-VAE-data/zinc250k/processed"
RUNS_ROOT="$HOME/projects/SC-VAE-runs-zinc"
AGGREGATED_ROOT="$HOME/projects/SC-VAE-aggregated-zinc"

EPOCHS=120
BATCH_SIZE=1024
NUM_WORKERS=8
NUM_PRIOR_SAMPLES=10000
INTERPOLATION_POOL_SIZE=2000
INTERPOLATION_PAIRS_PER_BIN=30
INTERPOLATION_STEPS=11

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

echo "[slurm finalize] dataset=zinc250k run_id=$RUN_ID"
echo "[slurm finalize] env_name=$ENV_NAME python=$PYTHON_BIN"
echo "[slurm finalize] runs_root=$RUNS_ROOT"
echo "[slurm finalize] aggregated_root=$AGGREGATED_ROOT"
echo "[slurm finalize] beta_start=$BETA_START beta_target=$BETA_TARGET beta_zero_epochs=$BETA_ZERO_EPOCHS beta_warmup_epochs=$BETA_WARMUP_EPOCHS rho_bias_init=$SPCAUCHY_RHO_BIAS_INIT grad_clip_norm=$GRAD_CLIP_NORM amp=$AMP"
echo "[slurm finalize] max_smiles_length=$MAX_SMILES_LENGTH"

CMD=(
  "$PYTHON_BIN" -m experiments.smiles.run_all
  --device cuda
  --seeds 0 1 2
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
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
  --aggregated-root "$AGGREGATED_ROOT"
  --run-id "$RUN_ID"
  --num-prior-samples "$NUM_PRIOR_SAMPLES"
  --interpolation-pool-size "$INTERPOLATION_POOL_SIZE"
  --interpolation-pairs-per-bin "$INTERPOLATION_PAIRS_PER_BIN"
  --interpolation-steps "$INTERPOLATION_STEPS"
  --skip-existing
)

if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp)
fi

"${CMD[@]}"
