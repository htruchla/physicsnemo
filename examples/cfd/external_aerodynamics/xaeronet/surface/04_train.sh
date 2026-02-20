#!/bin/bash
#SBATCH --job-name=xaeronet_train
#SBATCH --output=logs/xaeronet_train.out
#SBATCH --error=logs/xaeronet_train.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8         # CPU threads per GPU worker for data loading

#SBATCH -- account=rrg-nadaraja-ac
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=2                 # Number of GPU nodes — adjust to your allocation
#SBATCH --ntasks-per-node=4       # One task per GPU; must equal --gres=gpu:N below

#SBATCH --gres=gpu:4              # GPUs per node — adjust to match ntasks-per-node
# ---------- Environment ----------
module --force purge
module load python/3.11.5
module load cuda/12.6
module load vtk//9.3.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
source $SLURM_SUBMIT_DIR/xaeronet/bin/activate

# ---------- Setup ----------
cd $SLURM_SUBMIT_DIR
mkdir -p logs tensorboard checkpoints

# ---------- Distributed setup ----------
# physicsnemo's DistributedManager reads these standard environment variables.
# MASTER_ADDR: hostname of rank-0 node, resolved from SLURM's node list.
# WORLD_SIZE:  total number of GPU processes across all nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS   # nodes × ntasks-per-node = total GPUs

# ---------- Run ----------
# srun spawns one Python process per GPU task.
# Do NOT use torchrun here — srun + DistributedManager handles process launch.
echo "=== [$(date)] Training started ==="
echo "=== Nodes: $SLURM_NODELIST | Total GPUs (world size): $WORLD_SIZE ==="

srun python train_rorqual.py

EXIT_CODE=$?
echo "=== [$(date)] Training finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
