#!/bin/bash
#SBATCH --job-name=04_xaeronet_train_smallerjob
#SBATCH --output=logs/04_xaeronet_train_smallerjob.out
#SBATCH --error=logs/04_xaeronet_train_smallerjob.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8         # CPU threads per GPU worker for data loading

#SBATCH --account=def-nadaraja
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=2                 # Number of GPU nodes — adjust to your allocation
#SBATCH --ntasks-per-node=4       # One task per GPU; must equal --gres=gpu:N below

#SBATCH --gres=gpu:4              # increase when going to larger run GPUs per node — adjust to match ntasks-per-node
#SBATCH --exclude=rg32601         #excluding suspected faulty node
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

# Force NCCL to use InfiniBand for inter-node communication
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_DISABLE=0
# export NCCL_DEBUG=INFO   # temporarily — shows NCCL init details in the .err log

echo "=== MASTER_ADDR: $MASTER_ADDR ==="

#MAKE SURE wandb IS OFFLINE WHEN RUNNING ON RORQUAL/NARVAL/TamIA, IT WILL CRASH OTHERWISE BC IT IS TRYING TO SYNC 
wandb offline 

echo "=== Checking data paths from compute node ==="
ls /home/htruchla/links/scratch/XAERONET/partitions_training_DEBUGSET/ || echo "TRAINING DIR NOT ACCESSIBLE"
ls /home/htruchla/links/scratch/XAERONET/partitions_val_DEBUGSET/ || echo "VAL DIR NOT ACCESSIBLE"
ls /home/htruchla/links/scratch/XAERONET/ || echo "BASE DIR NOT ACCESSIBLE"

echo "=== stats_file check ==="
ls $SLURM_SUBMIT_DIR/global_stats.json || echo "global_stats.json NOT FOUND in submit dir"

srun python -u train_rorqual.py

EXIT_CODE=$?
echo "=== [$(date)] Training finished with exit code $EXIT_CODE ==="
exit $EXIT_CODE
