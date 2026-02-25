#!/bin/bash
#SBATCH --job-name=02_xaeronet_val_test_split
#SBATCH --output=logs/02_xaeronet_val_test_split.out
#SBATCH --error=logs/02_xaeronet_val_test_split.err
#SBATCH --time=0:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8        # Should match cfg.num_preprocess_workers in conf/config.yaml

#SBATCH --account=rrg-nadaraja-ac
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks=1

# ---------- Environment ----------
module --force purge
module load StdEnv/2023
module load cuda/12.6
module load vtk/9.3.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PYTHONPATH=/home/htruchla/envs/xaeronet/lib/python3.10/site-packages:${PYTHONPATH}

VENV_PATH="$HOME/envs/xaeronet"
source "$VENV_PATH/bin/activate"

# ---------- Setup ----------
cd $SLURM_SUBMIT_DIR
mkdir -p logs
mkdir -p /home/htruchla/links/scratch/XAERONET/partitions_val
mkdir -p /home/htruchla/links/scratch/XAERONET/partitions_test

# ---------- Run ----------
echo "=== [$(date)] Stats computation started on $(hostname) ==="
python val_test_split_rorqual.py
EXIT_CODE=$?
echo "=== [$(date)] Stats computation finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
