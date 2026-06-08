#!/bin/sh
#SBATCH -J dpo_hellaswag
#SBATCH -A MLMI-ae581-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mail-type=ALL
#SBATCH -p ampere
#SBATCH --output=logs/%x.out   						    # submit script's standard-out
#SBATCH --error=logs/%x.err  
#SBATCH -D /rds/user/ae581/hpc-work/diss # set working directory 


module purge                           
module load python/3.11.0-icl    


cd /rds/user/ae581/hpc-work/diss

source ./trl/trl-venv/bin/activate


echo Training model

python3 trainer.py --method dpo --task hellaswag



echo Job complete






