#!/bin/bash
#SBATCH --job-name=xaeronet_full_pipeline
#SBATCH --output=logs/xaeronet_full_pipeline.out
#SBATCH --error=logs/xaeronet_full_pipeline.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32        # Should match cfg.num_preprocess_workers in conf/config.yaml

#SBATCH -- account=rrg-nadaraja-ac
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=cpu           # Replace with Rorqual's CPU partition name (check: sinfo)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Verify venv exists before submitting anything
if [ ! -f "$SCRIPT_DIR/xaeronet/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at $SCRIPT_DIR/xaeronet"
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
echo " Stage 1  preprocess     -> Job $JOB1"

#Stage 2 - splitting the test and validation data
JOB2=$(sbatch --parsable --dependency_afterrok:$JOB1 "$SCRIPT_DIR/02_val_test_split.sh")
echo " Stage 2 data split  -> Job $JOB2"

# Stage 3 — Compute statistics (CPU), depends on Stage 1
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$SCRIPT_DIR/03_compute_stats.sh")
echo " Stage 2  compute_stats  -> Job $JOB3  (after $JOB2)"

# Stage 4 — Training (GPU), depends on Stage 2
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 "$SCRIPT_DIR/04_train.sh")
echo " Stage 3  train          -> Job $JOB4  (after $JOB3)"

echo "=================================================="
echo " All stages submitted. Useful commands:"
echo "   squeue -u \$USER                   # view job queue"
echo "   tail -f logs/preprocess_${JOB1}.out"
echo "   tail -f logs/stats_${JOB2}.out"
echo "   tail -f logs/train_${JOB3}.out"
echo "   scancel $JOB1 $JOB2 $JOB3         # cancel all stages"
echo "=================================================="
