#!/bin/bash
#SBATCH --job-name=xaeronet_preprocess
#SBATCH --output=logs/xaeronet_preprocess.out
#SBATCH --error=logs/xaeronet_preprocess.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32        # Should match cfg.num_preprocess_workers in conf/config.yaml

#SBATCH --account=rrg-nadaraja-ac
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=1
#SBATCH --ntasks=1

echo " ***********Loading modules************** "

# ---------- Environment ----------
module --force purge

module load StdEnv/2023          
module load python/3.11.5
module load cuda/12.6
module load vtk/9.3.0

echo " ********Modules loaded exporting cud a directory  ********"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

echo " **************Setting path for env ***************"
VENV_PATH="$HOME/envs/xaeronet"
source "$VENV_PATH/bin/activate"

echo "****************Installing pyvista *********************"
pip install --upgrade --force-reinstall pyvista

# ---------- Setup ----------
cd $SLURM_SUBMIT_DIR
mkdir -p logs partitions point_clouds

# ---------- Run ----------
echo "=== [$(date)] Preprocessing started on $(hostname) ==="
python preprocessor_rorqual.py
EXIT_CODE=$?
echo "=== [$(date)] Preprocessing finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
