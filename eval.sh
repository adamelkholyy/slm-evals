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



DS=gsm8k
echo Evaluating $DS...

# # # baseline benchmark
# CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
#     --model_args pretrained=Qwen/Qwen2.5-3B,trust_remote_code=True \
#     --tasks $DS \
#     --device cuda:0 \
#     --batch_size 8 \
#     --output_path ./benchmarks/qwen3b_base_$DS

# # # kto benchmark
CUDA_VISIBLE_DEVICES=0 lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-3B,peft=./outputs/kto_run_qwen3b/adapter-kto \
  --tasks $DS \
  --device cuda:0 \
  --output_path ./benchmarks/qwen3b_kto_$DS \
  --batch_size auto

# # # sft benchmark
CUDA_VISIBLE_DEVICES=0 lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-3B,peft=./outputs/sft_run_qwen3b/adapter-sft \
  --tasks $DS \
  --device cuda:0 \
  --output_path ./benchmarks/qwen3b_sft_$DS \
  --batch_size auto


# # # grpo benchmark
lm_eval \
  --model hf \
  --model_args pretrained=outputs/grpo_run_qwen3b/checkpoint-grpo,dtype="bfloat16" \
  --tasks $DS \
  --output_path ./benchmarks/qwen3b_grpo_$DS \
  --device cuda:0 \
  --batch_size auto



echo Job complete






