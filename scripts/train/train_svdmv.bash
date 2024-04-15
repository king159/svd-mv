#!/bin/bash
#SBATCH --job-name=svd360
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=26
#SBATCH --mem-per-cpu=60G
#SBATCH --partition=Video-aigc-general-32
#SBATCH --gres=gpu:8
#SBATCH --output=/mnt/petrelfs/wangjinghao.p/svd-360/slurm_output/%j_%x_out.log
#SBATCH --err=/mnt/petrelfs/wangjinghao.p/svd-360/slurm_output/%j_%x_err.log
#SBTACH --priority=4294967295

nvidia-smi
cd /mnt/petrelfs/wangjinghao.p/svd-360
# export NCCL_PROTO=simple
# export RDMAV_FORK_SAFE=1
# export FI_EFA_FORK_SAFE=1
# export FI_EFA_USE_DEVICE_RDMA=1
# export FI_PROVIDER=efa
# export FI_LOG_LEVEL=1
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=ib0

# offline
export TRANSFORMERS_OFFLINE="1"
export WANDB_MODE="offline" 
export http_proxy=
export https_proxy=
# # # cpu
# # export OMP_NUM_THREADS=4

# # start
echo "START TIME: $(date)"

GPUS_PER_NODE=8
NNODES=$SLURM_NNODES
NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=10001

# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="accelerate launch \
    --config_file src/train/accelerate_config/deepspeed_zero_3_multi_node.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    "
export PROGRAM="\
main.py train src/train_config/train_slurm_svd_mv.yaml
"

export CMD="$LAUNCHER $PROGRAM"

srun --jobid $SLURM_JOBID bash -c "$CMD"

echo "END TIME: $(date)"