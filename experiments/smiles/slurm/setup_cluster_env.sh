#!/bin/bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-scvae-smiles}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTORCH_CUDA="${PYTORCH_CUDA:-12.1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH."
  echo "If your cluster uses modules, load Conda first, for example:"
  echo "  module load anaconda"
  return 1 2>/dev/null || exit 1
fi

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "[setup env] creating env $ENV_NAME with python=$PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" pip
else
  echo "[setup env] env $ENV_NAME already exists"
fi

conda activate "$ENV_NAME"

echo "[setup env] installing PyTorch with pytorch-cuda=$PYTORCH_CUDA"
conda install -y -c pytorch -c nvidia pytorch pytorch-cuda="$PYTORCH_CUDA"

echo "[setup env] installing chemistry/data stack"
conda install -y -c conda-forge rdkit pandas numpy scipy scikit-learn matplotlib tqdm

echo "[setup env] sanity check"
python - <<'PY'
import sys
import torch
import rdkit
import pandas
import numpy
import scipy
import sklearn
import matplotlib
import tqdm

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("rdkit:", rdkit.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("sklearn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("tqdm:", tqdm.__version__)
PY

echo
echo "[setup env] done"
echo "Current env: $ENV_NAME"
echo "Python: $(which python)"
echo
echo "If you sourced this script, the env is already active."
echo "If you executed it normally, activate later with:"
echo "  source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate $ENV_NAME"
