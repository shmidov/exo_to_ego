#!/bin/bash
GPU_IDS=0,1
SEED=846514

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "Using GPUs: $GPU_IDS"

#! In-the-wild inference
python3 EgoX-main/infer.py \
    --meta_data_file ./EgoX-main/example/in_the_wild/meta.json \
    --depth_root ./EgoX-main/example/in_the_wild/depth_maps/ \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./EgoX-mainoutputs \
    --seed $SEED \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \
    --in_the_wild \
    --idx 1
