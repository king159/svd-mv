#!/bin/bash
#SBATCH --job-name=data_model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=60G
#SBATCH --partition=Video-aigc-general
#SBATCH --gres=gpu:1
#SBATCH --output=/mnt/petrelfs/wangjinghao.p/svd-360/slurm_output/%j_%x_out.log
#SBATCH --err=/mnt/petrelfs/wangjinghao.p/svd-360/slurm_output/%j_%x_err.log
#SBTACH --priority=4294967295

cd /mnt/petrelfs/wangjinghao.p/svd-360

# export TRANSFORMERS_OFFLINE="1"
# export WANDB_MODE="offline" 
export HF_ENDPOINT=https://hf-mirror.com
export http_proxy=http://wangjinghao.p:VC%403L%21gMz2@10.1.8.50:33128/
export https_proxy=http://wangjinghao.p:VC%403L%21gMz2@10.1.8.50:33128/

srun python /mnt/petrelfs/wangjinghao.p/get_model.py