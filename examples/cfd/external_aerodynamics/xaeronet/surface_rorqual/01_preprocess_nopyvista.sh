#!/bin/bash
#SBATCH --job-name=xaeronet_preprocess_nopyvista
#SBATCH --output=logs/xaeronet_preprocess_nopyvista.out
#SBATCH --error=logs/xaeronet_preprocess_nopyvista.err
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
echo " ***********Loading StdEnv/2023 at time $(date)************** "
module load StdEnv/2023
echo " ***********Loading python/3.11.5 at time $(date)************** "
module load python/3.11.5
echo " ***********Loading cuda/12.6 at time $(date)************** "
module load cuda/12.6
echo " ***********Loading vtk/9.3.0 at time $(date)************** "
module load vtk/9.3.0
echo " ********Modules loaded exporting cuda directory  ********"
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo " ********torch loading  ********"
module load torch/2.5.1
echo " ********torch geometric loading  ********"
module load torch_geometric/2.7.0
echo " ********pyvista loading  ********"
module load pyvista/0.47.1
echo " ********hydra loading  ********"
module load hydra_core/1.3.2

echo " **************Setting path for env ***************"
VENV_PATH="$HOME/envs/xaeronet"
source "$VENV_PATH/bin/activate"


# ---------- Setup ----------
cd $SLURM_SUBMIT_DIR
mkdir -p logs partitions point_clouds

# ---------- Run ----------
echo "=== [$(date)] Preprocessing started on $(hostname) ==="
python preprocessor_rorqual.py
EXIT_CODE=$?
echo "=== [$(date)] Preprocessing finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
