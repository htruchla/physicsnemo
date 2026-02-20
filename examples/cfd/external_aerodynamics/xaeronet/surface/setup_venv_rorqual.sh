#!/bin/bash

# Usage:
#   bash setup_venv.sh

set -euo pipefail

# 1 CONFIGURATION

PYTHON_MODULE="python/3.11.5"
CUDA_MODULE="cuda/12.6"
VENV_DIR="$PWD/venv"

# cu126 matches CUDA 12.6; torch 2.6.0 is the latest stable release for cu126
TORCH_CUDA_TAG="cu126"
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"

# PyG extension wheels must match torch + CUDA exactly
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"
TORCH_GEOMETRIC_VERSION="2.6.1"
TORCH_SCATTER_VERSION="2.1.4"
TORCH_SPARSE_VERSION="0.6.18"
TORCH_CLUSTER_VERSION="1.6.3"
TORCH_SPLINE_VERSION="1.2.2"

# 2 Load modules

echo ">>> Loading system modules..."
module purge
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"
echo "    Python : $(python --version)"
echo "    CUDA   : $CUDA_MODULE"

# 3 Create virtual environment

if [ -d "$VENV_DIR" ]; then
    echo ""
    echo ">>> Virtual environment already exists at $VENV_DIR."
    echo "    To fully rebuild: rm -rf $VENV_DIR && bash setup_venv.sh"
    read -rp "    Continue installing into existing venv? [y/N] " confirm
    [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
else
    echo ">>> Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo ">>> Activated: $(which python)"
pip install --upgrade pip setuptools wheel

# 4 Install PyTorch (must come before everything else)


echo ""
echo ">>> Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA_TAG}..."
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible — check your cuda module!'
print(f'    torch {torch.__version__} | CUDA {torch.version.cuda} | GPUs visible: {torch.cuda.device_count()}')
"

# 5 Install PyTorch Geometric + extensions (must match torch + CUDA exactly)

echo ""
echo ">>> Installing PyTorch Geometric ${TORCH_GEOMETRIC_VERSION}..."
pip install "torch-geometric==${TORCH_GEOMETRIC_VERSION}"

echo ">>> Installing PyG extensions (scatter, sparse, cluster, spline-conv)..."
pip install \
    "torch-scatter==${TORCH_SCATTER_VERSION}" \
    "torch-sparse==${TORCH_SPARSE_VERSION}" \
    "torch-cluster==${TORCH_CLUSTER_VERSION}" \
    "torch-spline-conv==${TORCH_SPLINE_VERSION}" \
    --find-links "$PYG_WHEEL_URL"


# 6 Install remaining requirements

echo ""
echo ">>> Installing remaining requirements from requirements_rorqual.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_rorqual.txt"


# 7 Smoke test — verify every key import across all three scripts


echo ""
echo ">>> Running import smoke test..."
python - <<'EOF'
import sys

checks = [
    ("torch",                           "torch"),
    ("torch_geometric",                 "torch_geometric"),
    ("vtk",                             "vtk"),
    ("pyvista",                         "pyvista"),
    ("numpy",                           "numpy"),
    ("sklearn",                         "sklearn"),
    ("hydra",                           "hydra"),
    ("omegaconf",                       "omegaconf"),
    ("tqdm",                            "tqdm"),
    ("tensorboard",                     "tensorboard"),
    ("wandb",                           "wandb"),
    ("physicsnemo.distributed",         "physicsnemo.distributed"),
    ("physicsnemo.models.meshgraphnet", "physicsnemo.models.meshgraphnet"),
    ("physicsnemo.datapipes.cae",       "physicsnemo.datapipes.cae"),
    ("physicsnemo.launch.logging",      "physicsnemo.launch.logging"),
    ("physicsnemo.sym.geometry",        "physicsnemo.sym.geometry"),
]

failed = []
for label, mod in checks:
    try:
        __import__(mod)
        print(f"  [OK]   {label}")
    except ImportError as e:
        print(f"  [FAIL] {label}: {e}")
        failed.append(label)

if failed:
    print(f"\n  {len(failed)} import(s) failed: {failed}")
    sys.exit(1)
else:
    print(f"\n  All {len(checks)} imports succeeded.")
EOF

# 8 Done

echo ""
echo "================================================================="
echo " Environment ready: $VENV_DIR"
echo " Activate manually with:"
echo "   source $VENV_DIR/bin/activate"
echo "================================================================="
