#!/bin/bash
# Usage:
#   bash pipeline_rorqual 8.sh
#
# This submits three jobs in sequence. Each stage only starts if the previous
# one completed successfully (--dependency=afterok). If any stage fails, the
# downstream jobs are automatically cancelled by SLURM.
#
# Before submitting:
#   1. Run setup_venv.sh 
#   2. Verify partition names in 01_preprocess.sh, 02_compute_stats.sh,
#      and 03_train.sh match your Rorqual allocation (use: sinfo).
#   3. Verify GPU counts in 03_train.sh match your allocation.
#   4. Ensure conf/config.yaml exists and all paths inside it are correct.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify venv exists before submitting anything
if [ ! -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/venv"
    echo "       Run setup_venv.sh first."
    exit 1
fi

# Verify Hydra config exists
if [ ! -f "$SCRIPT_DIR/conf/config.yaml" ]; then
    echo "ERROR: Hydra config not found at $SCRIPT_DIR/conf/config.yaml"
    echo "       All three Python scripts require this file."
    exit 1
fi

mkdir -p "$SCRIPT_DIR/logs"

echo "=================================================="
echo " Submitting DrivAer pipeline"
echo " Working directory: $SCRIPT_DIR"
echo "=================================================="

# Stage 1 — Preprocessing (CPU)
JOB1=$(sbatch --parsable "$SCRIPT_DIR/01_preprocess.sh")
echo " Stage 1  preprocess     → Job $JOB1"

# Stage 2 — Compute statistics (CPU), depends on Stage 1
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$SCRIPT_DIR/02_compute_stats.sh")
echo " Stage 2  compute_stats  → Job $JOB2  (after $JOB1)"

# Stage 3 — Training (GPU), depends on Stage 2
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$SCRIPT_DIR/03_train.sh")
echo " Stage 3  train          → Job $JOB3  (after $JOB2)"

echo "=================================================="
echo " All stages submitted. Useful commands:"
echo "   squeue -u \$USER                   # view job queue"
echo "   tail -f logs/preprocess_${JOB1}.out"
echo "   tail -f logs/stats_${JOB2}.out"
echo "   tail -f logs/train_${JOB3}.out"
echo "   scancel $JOB1 $JOB2 $JOB3         # cancel all stages"
echo "=================================================="
