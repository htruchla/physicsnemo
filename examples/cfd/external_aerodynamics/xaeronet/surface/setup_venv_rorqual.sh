#!/bin/bash
# =============================================================================
# setup_venv.sh — Create the Python virtual environment for the DrivAer pipeline
# Cluster: Rorqual (DRAC)  |  Python 3.11.5  |  CUDA 12.6
#
# Usage:
#   bash setup_venv.sh
#
# Run this ONCE from your project directory on a Rorqual login node.
# The resulting ./venv is referenced by all sbatch scripts automatically.
# =============================================================================

set -euo pipefail

# --------------------------------------------------------------------------- #
# 1. CONFIGURATION
# --------------------------------------------------------------------------- #

PYTHON_MODULE="python/3.11.5"
CUDA_MODULE="cuda/12.6"
VENV_DIR="$PWD/xaeronet"

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

# --------------------------------------------------------------------------- #
# 2. Load modules
# --------------------------------------------------------------------------- #

echo ">>> Loading system modules..."
module --force purge
module load StdEnv/2023       # Reload Compute Canada standard env first
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"
echo "    Python : $(python --version)"
echo "    CUDA   : $CUDA_MODULE"

# --------------------------------------------------------------------------- #
# 3. Create virtual environment
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# 4. Install PyTorch (must come before everything else)
# --------------------------------------------------------------------------- #

echo ""
echo ">>> Installing PyTorch ${TORCH_VERSION}+${TORCH_CUDA_TAG}..."
# Pin the full local version tag (e.g. torch==2.6.0+cu126) so pip cannot
# match the Compute Canada wheelhouse build (torch==2.6.0+computecanada),
# which is compiled without CUDA support.
pip install \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
    --extra-index-url "https://pypi.org/simple"

# NOTE: Login nodes have no GPUs, so cuda.is_available() is always False here.
# CUDA availability is only meaningful inside a GPU job. We check the build
# version string instead, which is set at compile time regardless of hardware.
python -c "
import torch
cuda_build = torch.version.cuda
if cuda_build is None:
    print('  [FAIL] torch installed WITHOUT CUDA support.')
    print(f'         Version: {torch.__version__}')
    print('         Expected +cu126 build — CC wheelhouse may have intercepted.')
    raise SystemExit(1)
else:
    print(f'  [OK]   torch {torch.__version__} | built for CUDA {cuda_build}')
    print('         (GPU count will be >0 only inside a GPU job, not on login node)')
"

# --------------------------------------------------------------------------- #
# 5. Install PyTorch Geometric + extensions (must match torch + CUDA exactly)
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# 6. Install remaining requirements
# --------------------------------------------------------------------------- #

echo ""
echo ">>> Installing remaining requirements from requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements.txt"

# --------------------------------------------------------------------------- #
# 7. Smoke test — verify every key import across all three scripts
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# 8. Done
# --------------------------------------------------------------------------- #

echo ""
echo "================================================================="
echo " Environment ready: $VENV_DIR"
echo " Activate manually with:"
echo "   source $VENV_DIR/bin/activate"
echo "================================================================="