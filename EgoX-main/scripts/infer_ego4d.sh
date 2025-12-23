#!/bin/bash
GPU_IDS=6
SEED=42

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "Using GPUs: $GPU_IDS"

#! Ego-Exo4D inference
python3 infer.py \
    --prompt ./example/egoexo4D/caption.txt \
    --exo_video_path ./example/egoexo4D/exo_path.txt \
    --ego_prior_video_path ./example/egoexo4D/ego_prior_path.txt \
    --meta_data_file ./example/egoexo4D/camera_params.json \
    --depth_root ./example/egoexo4D/depth_maps/ \
    --model_path ./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers \
    --lora_path ./checkpoints/EgoX/pytorch_lora_weights.safetensors \
    --lora_rank 256 \
    --out ./results \
    --seed $SEED \
    --use_GGA \
    --cos_sim_scaling_factor 3.0 \

