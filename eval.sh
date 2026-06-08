#!/bin/sh
#SBATCH -J qwen_blimp_eval
#SBATCH -A MLMI-ae581-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:30:00
#SBATCH --mail-type=ALL
#SBATCH -p ampere
#SBATCH --output=logs/%x.out   						    # submit script's standard-out
#SBATCH --error=logs/%x.err  
#SBATCH -D /rds/user/ae581/hpc-work/diss # set working directory 


module purge                           
module load python/3.11.0-icl    

# source ./eval-venv/bin/activate
source ./trl/trl-venv/bin/activate

cd /rds/user/ae581/hpc-work/diss

echo Evaluating base Qwen2.5-3B-Instruct on BLiMP...

lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-3B-Instruct,trust_remote_code=True \
    --tasks blimp \
    --device cuda:0 \
    --batch_size 8 \
    --output_path ./results/qwen3b_base_blimp

echo Job complete






