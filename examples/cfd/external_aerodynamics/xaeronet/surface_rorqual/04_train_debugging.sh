#!/bin/bash
#SBATCH --job-name=04_xaeronet_train_debugging
#SBATCH --output=logs/debugging/04_xaeronet_train_debugging_%j.out
#SBATCH --error=logs/debugging/04_xaeronet_train_debugging_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=8         # CPU threads per GPU worker for data loading

#SBATCH --account=def-nadaraja
#SBATCH --mail-user=hana.truchla@mail.mcgill.ca
#SBATCH --mail-type=ALL

#SBATCH --nodes=2                 # Number of GPU nodes — adjust to your allocation
#SBATCH --ntasks-per-node=4       # One task per GPU; must equal --gres=gpu:N below

#SBATCH --gres=gpu:4              # increase when going to larger run GPUs per node — adjust to match ntasks-per-node
#SBATCH --exclude=rg32601         #excluding suspected faulty node

# ---------- Resubmission guard ----------
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
mkdir -p logs tensorboard checkpoints 
mkdir -p logs/full_run logs/debugging

# ---------- Distributed setup ----------
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS   # nodes × ntasks-per-node = total GPUs

# ---------- Run ----------
# Force NCCL to use InfiniBand for inter-node communication
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_DISABLE=0

echo "=== [$(date)] Training started (resubmission #${RESUBMIT_COUNT})==="
echo "=== Nodes: $SLURM_NODELIST | Total GPUs (world size): $WORLD_SIZE ==="
echo "===  : $MASTER_ADDR ==="

wandb offline #MAKE SURE wandb IS OFFLINE WHEN RUNNING ON RORQUAL/NARVAL/TamIA, IT WILL CRASH OTHERWISE BC IT IS TRYING TO SYNC ONLINE

echo " === RUNNING train_rorqual.py ==="
srun python -u train_rorqual.py --config-name=config_debug

EXIT_CODE=$?
echo "=== [$(date)] Job finished with exit code $EXIT_CODE ==="

# ---------- Resubmit if needed ----------
if [ $EXIT_CODE -eq 0 ]; then
    if [ -f "training_complete.flag" ]; then
        echo "=== Training is fully complete. Not resubmitting. ==="
    elif [ "$RESUBMIT_COUNT" -lt "$MAX_RESUBMISSIONS" ]; then
        NEW_COUNT=$((RESUBMIT_COUNT + 1))
        echo "=== Resubmitting job (attempt $NEW_COUNT / $MAX_RESUBMISSIONS) ==="
        sbatch --export=ALL,RESUBMIT_COUNT=$NEW_COUNT "$0"
    else
        echo "=== Reached MAX_RESUBMISSIONS ($MAX_RESUBMISSIONS). Not resubmitting. ==="
    fi

exit $EXIT_CODE
