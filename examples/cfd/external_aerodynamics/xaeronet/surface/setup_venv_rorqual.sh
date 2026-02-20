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

# Target torch 2.9.1 — the CC wheelhouse PyG extensions (+torch29.computecanada)
# require torch~=2.9.0. Using 2.6.0 causes pip to upgrade torch anyway and
# break torchvision, so we target 2.9.1 from the start for consistency.
TORCH_VERSION="2.9.1"
TORCH_GEOMETRIC_VERSION="2.6.1"
# PyG extension versions resolved automatically from CC wheelhouse.

# --------------------------------------------------------------------------- #
# 2. Load modules
# --------------------------------------------------------------------------- #

echo ">>> Loading system modules..."
module --force purge
module load StdEnv/2023       # Reload Compute Canada standard env first
module load "$PYTHON_MODULE"
module load "$CUDA_MODULE"
module load vtk/9.3.0         # VTK provided as system module — no pip wheel on CC
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
# Let CC wheelhouse serve torch and a compatible torchvision.
# Pinning torchvision causes version conflicts because CC's torchvision build
# tags (+computecanada) do not match standard version strings.
pip install \
    "torch==${TORCH_VERSION}" \
    torchvision

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
# CC wheelhouse has +torch29.computecanada builds for all extensions that are
# compatible with torch 2.9.1. No --find-links needed — CC resolves these natively.
pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv

# --------------------------------------------------------------------------- #
# 6. Install remaining requirements
# --------------------------------------------------------------------------- #

echo ""
echo ">>> Linking system VTK into venv..."
# VTK has no pip wheel on Compute Canada. It is provided as a system module.
# We load the vtk module and add its Python bindings to the venv via a .pth
# file so all subsequent pip installs and Python processes can see it.
module load vtk/9.3.0

# After loading the module, the system Python can already import vtk.
# We ask it directly for the path and write it to a .pth file in the venv.
# A .pth file in site-packages is read at Python startup and its paths
# are added to sys.path — making "import vtk" work inside the venv.
VTK_SITE=$(python -c "import vtk, os; print(os.path.dirname(vtk.__file__))")

if [ -n "$VTK_SITE" ]; then
    echo "    VTK bindings at: $VTK_SITE"
    echo "$VTK_SITE" > "$VENV_DIR/lib/python3.11/site-packages/vtk_system.pth"
    echo "    Linked via vtk_system.pth"
else
    echo "    ERROR: Could not locate VTK after module load vtk/9.3.0"
    exit 1
fi

echo ""
echo ">>> Installing nvidia-physicsnemo-sym (--no-deps --no-build-isolation)..."
# --no-deps: skip the vtk>=9.2.6 pip dependency since we provide it via the
#            system module above. All other deps are already installed.
# --no-build-isolation: setup.py imports torch at build time; this lets it
#                       find torch from the current venv.
pip install --no-build-isolation --no-deps nvidia-physicsnemo-sym

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