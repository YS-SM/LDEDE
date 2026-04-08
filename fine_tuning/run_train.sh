#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONWARNINGS="ignore"

source /root/autodl-tmp/venv01/bin/activate

echo "========================================"
echo "Current available GPU information:"
echo "========================================"
nvidia-smi
echo "========================================"

PYTHONUNBUFFERED=1 python /root/autodl-tmp/LRP_SPIN/fine_tuning/accelerate_cli.py

echo "========================================"
echo "Training script execution completed!"
echo "Results are saved in: /root/autodl-tmp/LRP_SPIN/WT_model/2025_11_23_gpt2_xl"
echo "========================================"