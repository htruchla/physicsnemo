#!/bin/bash
#SBATCH --job-name=04_xaeronet_train
#SBATCH --output=logs/full_runs/2026_03_12/04_xaeronet_train_%j.out
#SBATCH --error=logs/full_runs/2026_03_12/04_xaeronet_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8         # CPU threads per GPU worker for data loading

#SBATCH --account=def-nadaraja
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=2                 # Number of GPU nodes — adjust to your allocation
#SBATCH --ntasks-per-node=4       # One task per GPU; must equal --gres=gpu:N below

#SBATCH --gres=gpu:4              # increase when going to larger run GPUs per node — adjust to match ntasks-per-node
#SBATCH --exclude=rg32601,rg32503,rg31602,rg31605,rg32403,rg32602

# ---------- Resubmission parameters ----------
MAX_RESUBMISSIONS=10
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}  #inherit from env when resubmitting 

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
#make sure you set up with the current date
mkdir -p logs/debugging/2026_03_12 tensorboard checkpoints

# ---------- Distributed setup ----------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS   # nodes × ntasks-per-node = total GPUs

# ---------- Run ----------
# Force NCCL to use InfiniBand for inter-node communication

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0          # <-- KEY FIX: was missing, Gloo was using ethernet
export NCCL_IB_DISABLE=0              # explicitly keep IB enabled for NCCL
export NCCL_TIMEOUT=1800000
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export TORCH_NCCL_ENABLE_MONITORING=1
export GLOO_TIMEOUT_SECONDS=3600 


#DEBUGGING
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
ulimit -c unlimited

echo "=== [$(date)] Job started ==="
echo "=== Nodes: $SLURM_NODELIST ==="
echo "=== Master: $MASTER_ADDR | World size: $WORLD_SIZE ==="
echo "=== Resubmit count: $RESUBMIT_COUNT / $MAX_RESUBMISSIONS ==="

#quick cinnectivity test before training
echo "=== Testing inter-node GPU connectivity ==="
srun --ntasks=$WORLD_SIZE bash -c 'echo "Rank $SLURM_PROCID on $(hostname) — GPU $SLURM_LOCALID ready"'


wandb offline #MAKE SURE wandb IS OFFLINE WHEN RUNNING ON RORQUAL/NARVAL/TamIA, IT WILL CRASH OTHERWISE BC IT IS TRYING TO SYNC ONLINE

echo " === RUNNING train_rorqual.py ==="
srun python -u train_rorqual.py --config-name=config

EXIT_CODE=$?
echo "=== [$(date)] Job finished with exit code $EXIT_CODE ==="

# ---------- Resubmit if needed ----------
if [ ! -f "training_complete.flag" ]; then
    if [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMISSIONS" ]; then
        NEW_COUNT=$((RESUBMIT_COUNT + 1))
        echo "=== Resubmitting job (attempt $NEW_COUNT / $MAX_RESUBMISSIONS) ==="
        sbatch --export=ALL,RESUBMIT_COUNT=$NEW_COUNT "$0"
    else
        echo "=== Reached MAX_RESUBMISSIONS ($MAX_RESUBMISSIONS). Not resubmitting. ==="
    fi
else
    echo "=== Training is fully complete. Not resubmitting. ==="
fi

exit $EXIT_CODE

