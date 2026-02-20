#!/bin/bash
#SBATCH --job-name=xaeronet_val_test_split
#SBATCH --output=logs/xaeronet_val_test_split.out
#SBATCH --error=logs/xaeronet_val_test_split.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8        # Should match cfg.num_preprocess_workers in conf/config.yaml

#SBATCH -- account=rrg-nadaraja-ac
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --partition=cpu           # Replace with Rorqual's CPU partition name (check: sinfo)

# ---------- Environment ----------
module --force purge
module load python/3.11.5
module load cuda/12.6
module load vtk/9.3.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
source $SLURM_SUBMIT_DIR/xaeronet/bin/activate

# ---------- Setup ----------
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# ---------- Run ----------
echo "=== [$(date)] Stats computation started on $(hostname) ==="
python val_test_split_rorqual.py
EXIT_CODE=$?
echo "=== [$(date)] Stats computation finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
