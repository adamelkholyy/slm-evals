#!/bin/sh

cd /local/scratch/ae581

source ./trl-venv/bin/activate


echo Training model

CUDA_VISIBLE_DEVICES=0 python3 trainer.py --method kto --run_name kto_qwen3b



echo Job complete

