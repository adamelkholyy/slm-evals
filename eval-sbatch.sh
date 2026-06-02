#!/bin/sh
#SBATCH -J slm-eval
#SBATCH -A MLMI-ae581-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mail-type=ALL
#SBATCH -p ampere
#SBATCH --output=logs/log.out   						    # submit script's standard-out
#SBATCH --error=logs/log.err  
#SBATCH -D /rds/user/ae581/hpc-work/diss # set working directory 


module purge                           
module load python/3.11.0-icl    

source ./eval-venv/bin/activate

cd /rds/user/ae581/hpc-work/diss

echo Evaluating model...

NAME="pythia-12b-deduped-v0"

lm_eval --model hf \
    --model_args pretrained=EleutherAI/$NAME \
    --tasks piqa \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results


echo Job complete






