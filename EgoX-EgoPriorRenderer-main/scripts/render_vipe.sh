#!/bin/bash

# Configuration
CUDA_VISIBLE_DEVICES=0

# Paths
INPUT_DIR="vipe_results/joker"
META_JSON_PATH="../example/in_the_wild/meta.json"
OUT_DIR="../example/in_the_wild/videos"

# Rendering parameters
POINT_SIZE="5.0"
START_FRAME="0"
END_FRAME="48"

echo "=========================================="
echo "Running ViPE Point Cloud Rendering"
echo "=========================================="
echo "Input: $INPUT_DIR"
echo "Meta JSON: $META_JSON_PATH"
echo "Output: $OUT_DIR"
echo "Frames: $START_FRAME - $END_FRAME"
echo "=========================================="

python scripts/render_vipe_pointcloud.py \
    --input_dir $INPUT_DIR \
    --out_dir $OUT_DIR \
    --meta_json_path $META_JSON_PATH \
    --point_size $POINT_SIZE \
    --start_frame $START_FRAME \
    --end_frame $END_FRAME \
    --fish_eye_rendering \
    --use_mean_bg

echo "Completed!"
