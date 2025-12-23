#!/bin/bash

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Video path
VIDEO_PATH="../example/in_the_wild/videos/joker/exo.mp4"

# Inference settings
START_FRAME=0
END_FRAME=48
PIPELINE="lyra"

echo "=========================================="
echo "ViPE Inference"
echo "=========================================="
echo "Video: $VIDEO_PATH"
echo "Frame range: $START_FRAME to $END_FRAME"
echo "Pipeline: $PIPELINE"
echo "=========================================="

# Check if video file exists
if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "ERROR: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Run ViPE inference
vipe infer "$VIDEO_PATH" \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --assume_fixed_camera_pose \
    --pipeline $PIPELINE

echo "Inference completed!"