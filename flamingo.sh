#!/bin/sh

cd /local/scratch/ae581

source ./trl/trl-venv/bin/activate


echo Training model

CUDA_VISIBLE_DEVICES=0 python3 trainer.py --method grpo --run_name grpo_new_qwen3b



echo Job complete

