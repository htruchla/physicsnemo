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

TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"

# PyG wheels must reference the CUDA version; cu126 matches CUDA 12.6
TORCH_CUDA_TAG="cu126"
PYG_WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA_TAG}.html"
TORCH_GEOMETRIC_VERSION="2.6.1"
# PyG extension versions are resolved automatically — CC wheelhouse
# serves +computecanada builds that do not match strict version pins.

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
echo ">>> Installing PyTorch ${TORCH_VERSION}..."
# On Compute Canada / Rorqual, pip always resolves torch from the CC wheelhouse
# (/cvmfs/), which serves +computecanada builds compiled WITH CUDA support.
# Pinning +cu126 causes a "no matching distribution" error because that local
# version tag does not exist in the CC index. We let CC serve their build and
# verify CUDA was compiled in via torch.version.cuda (not cuda.is_available(),
# which is always False on login nodes that have no GPUs).
pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}"

python -c "
import torch
cuda_build = torch.version.cuda
if cuda_build is None:
    print('  [FAIL] torch installed WITHOUT CUDA support.')
    print(f'         Version: {torch.__version__}')
    raise SystemExit(1)
else:
    print(f'  [OK]   torch {torch.__version__} | compiled for CUDA {cuda_build}')
    print('         (cuda.is_available() will be True only inside a GPU job)')
"

# --------------------------------------------------------------------------- #
# 5. Install PyTorch Geometric + extensions (must match torch + CUDA exactly)
# --------------------------------------------------------------------------- #

echo ""
echo ">>> Installing PyTorch Geometric ${TORCH_GEOMETRIC_VERSION}..."
pip install "torch-geometric==${TORCH_GEOMETRIC_VERSION}"

echo ">>> Installing PyG extensions (scatter, sparse, cluster, spline-conv)..."
# No strict version pins — CC wheelhouse has these as +computecanada or +pt26cu126
# builds that are incompatible with exact version matching. We search both the
# PyG wheel page and PyPI and let pip pick the best compatible version.
pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    --find-links "$PYG_WHEEL_URL"

# --------------------------------------------------------------------------- #
# 6. Install remaining requirements
# --------------------------------------------------------------------------- #

echo ""
echo ">>> Installing remaining requirements from requirements.txt..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_rorqual.txt"

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
echo ""
echo " To activate in your current shell:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo " To deactivate:"
echo "   deactivate"
echo ""
echo " The sbatch pipeline scripts activate it automatically."
echo " When activated, your prompt will show: (xaeronet)"
echo "================================================================="