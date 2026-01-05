#!/bin/bash
GPU_IDS=0
SEED=846514

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "Using GPUs: $GPU_IDS"

python3 infer.py \
    --meta_data_file ./data/meta.json \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results \
    --seed 42 \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \
    --in_the_wild