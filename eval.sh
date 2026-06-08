#!/bin/sh
#SBATCH -J kto_gsm8k_eval
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

echo Evaluating model...

NAME="pythia-12b-deduped-v0"

# lm_eval --model hf \
#     --model_args pretrained=EleutherAI/$NAME \
#     --tasks piqa \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path results

# Base model
# lm_eval --model hf \
#   --model_args pretrained=EleutherAI/pythia-12b-deduped \
#   --tasks gsm8k \
#   --output_path ./results/base

# # Fine-tuned
# lm_eval --model hf \
#   --model_args pretrained=EleutherAI/pythia-12b-deduped,peft=./outputs/checkpoint-<N> \
#   --tasks gsm8k \
#   --output_path ./results/finetuned


# arc 36
# gsm8k 117
# piqa 7000

python eval.py \
    --base EleutherAI/pythia-12b-deduped \
    --finetuned ./outputs/kto_gsm8k/checkpoint-117 \
    --tasks gsm8k \
    --n_samples 999999

echo Job complete






