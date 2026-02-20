#!/bin/bash
#SBATCH --job-name=xaeronet_computestats
#SBATCH --output=logs/xaeronet_computestats.out
#SBATCH --error=logs/xaeronet_computestats.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16        # Should match cfg.num_preprocess_workers in conf/config.yaml

#SBATCH -- account=rrg-nadaraja-ac
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---------- Environment ----------
module --force purge
module load python/3.11.5
module load cuda/12.6
module load vtk/9.3.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
VENV_PATH="$HOME/envs/xaeronet"
source "$VENV_PATH/bin/activate"

# ---------- Setup ----------
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# ---------- Run ----------
echo "=== [$(date)] Stats computation started on $(hostname) ==="
python compute_stats_rorqual.py
EXIT_CODE=$?
echo "=== [$(date)] Stats computation finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
