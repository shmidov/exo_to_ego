#!/usr/bin/env bash

wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
mv ./Anaconda3-2022.10-Linux-x86_64.sh ../
bash ../Anaconda3-2022.10-Linux-x86_64.sh -b -p $HOME/PROGS/anaconda3
source ~/PROGS/anaconda3/bin/activate
conda init

conda create -n exo_to_ego python=3.10 -y

conda activate exo_to_ego

# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # for A100 on DGX
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r ./requirements.txt

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='DAVIAN-Robotics/EgoX', local_dir='./checkpoints/EgoX', allow_patterns='*.safetensors')"


