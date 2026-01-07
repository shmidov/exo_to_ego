#!/bin/bash

echo "🚀 [START] Enroot automated container customization script"

# === CONFIG ===
# BASE_IMAGE="nvidia/cuda:12.1.1-devel-ubuntu22.04"
BASE_IMAGE="nvcr.io#nvidia/pytorch:25.10-py3"
BASE_SQSH="$PWD/$(echo $BASE_IMAGE | tr ':/' '+').sqsh"
CONTAINER_NAME="exo_to_ego.sqsh"
RUN_STEP2=0

# Step 1: Import base image if needed
if [ ! -f "$BASE_SQSH" ]; then
    echo "📦 Importing base image..."
    enroot import -o "$BASE_SQSH" docker://$BASE_IMAGE || { echo "❌ Failed to import base image"; exit 1; }
else
    echo "📦 Found existing image."
fi

# Step 2: Customize container with Python, tmux and other system packages
if [ "$RUN_STEP2" -eq 1 ]; then
  echo "🔧 [STEP 2] Installing Python 3.10, tmux and system packages inside container..."
  srun -p mig,work -G 1 \
    --container-image="$BASE_SQSH" \
    --container-save="$PWD/$CONTAINER_NAME" \
    bash <<'EOF'
set -e

echo "🔧 [STEP 2] Installing Python 3.10 prerequisites"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common curl gnupg ca-certificates wget

echo "🔧 [STEP 2] Adding deadsnakes PPA"
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update

echo "🔧 [STEP 2] Installing Python 3.10, tmux and system packages"
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    build-essential \
    curl \
    tmux

echo "🔧 [STEP 2] Configuring python3 alternative"
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

echo "🔧 [STEP 2] Skipping system pip upgrades (handled in conda env)"

# quick verification that tmux was installed
if command -v tmux >/dev/null 2>&1; then
  echo "✔️ tmux installed: $(tmux -V)"
else
  echo "⚠️ tmux not found after install (this is unexpected)"
fi

# Clean up apt caches to reduce image size
apt-get clean
rm -rf /var/lib/apt/lists/*

EOF

  if [ $? -ne 0 ]; then
      echo "❌ Failed in STEP 2: Python/tmux setup"
      exit 1
  fi
  echo "✅ [STEP 2] Finished Python 3.10 & tmux setup in container"
else
  echo "⏭️ [STEP 2] Skipping Python/tmux system packages (already installed)"
fi


# Step 3: Mount your code, set up conda env, install packages, and download models
echo "🌀 [STEP 3] Mounting your project, creating conda env, installing packages, and downloading models..."
srun -p mig,work -G 1 \
  --container-image="$PWD/$CONTAINER_NAME" \
  --container-save="$PWD/$CONTAINER_NAME" \
  --container-mounts="$PWD:/workspace" \
  bash <<'EOF'
set -e
echo "🌀 [STEP 3] Entering /workspace"
cd /workspace

CONDA_ROOT="$HOME/PROGS/anaconda3"
CONDA_INSTALLER="/tmp/Anaconda3-2022.10-Linux-x86_64.sh"
REQ_MARKER="/workspace/.exo_to_ego_requirements_installed"
FORCE_REINSTALL=0
FORCE_REQUIREMENTS=1
INSTALL_TORCH=0
RUN_CHECKPOINTS=0

if [ "$FORCE_REINSTALL" -eq 1 ]; then
  echo "🌀 [STEP 3] Force reinstall enabled, resetting conda install and requirements marker"
  rm -rf "$CONDA_ROOT"
  rm -f "$REQ_MARKER"
fi

if [ -d "$CONDA_ROOT" ]; then
  echo "🌀 [STEP 3] Anaconda already installed at $CONDA_ROOT"
else
  echo "🌀 [STEP 3] Downloading and installing Anaconda"
  if [ ! -f "$CONDA_INSTALLER" ]; then
    wget -O "$CONDA_INSTALLER" https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
  else
    echo "🌀 [STEP 3] Found existing Anaconda installer at $CONDA_INSTALLER"
  fi
  bash "$CONDA_INSTALLER" -b -p "$CONDA_ROOT"
fi

if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
else
  source "$CONDA_ROOT/bin/activate"
fi

echo "🌀 [STEP 3] Creating and activating conda env (python=3.10)"
if conda env list | awk '{print $1}' | grep -qx "exo_to_ego"; then
  if [ "$FORCE_REINSTALL" -eq 1 ]; then
    echo "🌀 [STEP 3] Removing existing conda env exo_to_ego"
    conda remove -n exo_to_ego --all -y
  else
    echo "🌀 [STEP 3] Conda env exo_to_ego already exists"
  fi
else
  conda create -n exo_to_ego python=3.10 -y
fi
conda activate exo_to_ego

if [ "$INSTALL_TORCH" -eq 1 ]; then
  echo "🌀 [STEP 3] Installing PyTorch (CUDA 11.8)"
  if [ "$FORCE_REINSTALL" -eq 1 ]; then
    pip3 install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu118
  else
    if python -c "import torch" >/dev/null 2>&1; then
      echo "🌀 [STEP 3] Torch already installed, skipping"
    else
      pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    fi
  fi
else
  echo "⏭️ [STEP 3] Skipping PyTorch install (already installed)"
fi

echo "🌀 [STEP 3] Installing project requirements"
if [ -f requirements.txt ]; then
  if [ "$FORCE_REQUIREMENTS" -eq 1 ]; then
    pip install -r ./requirements.txt
    touch "$REQ_MARKER"
  else
    if [ -f "$REQ_MARKER" ]; then
      echo "🌀 [STEP 3] Requirements already installed (marker found)"
    else
      pip install -r ./requirements.txt
      touch "$REQ_MARKER"
    fi
  fi
else
  echo "⚠️ requirements.txt not found, skipping"
fi

echo "🛠 Adding src/ to PYTHONPATH for this session"
export PYTHONPATH="/workspace/src:$PYTHONPATH"

echo "🫠 [STEP 3a] Verifying torch install using the conda python"
python -c "import torch; print('Torch version:', getattr(torch,'__version__','<not installed>')); print('CUDA available:', torch.cuda.is_available())"

# Verify tmux is available in the final container
if command -v tmux >/dev/null 2>&1; then
  echo "✔️ tmux available in container: $(tmux -V)"
else
  echo "⚠️ tmux not found in the container (installation step may have failed)"
fi

if [ "$RUN_CHECKPOINTS" -eq 1 ]; then
  echo "🌀 [STEP 3b] Downloading Hugging Face models"
  MODEL_DIR_1="./checkpoints/pretrained_model/Wan2.1-I2V-14B-480P-Diffusers"
  MODEL_DIR_2="./checkpoints/EgoX"

  if [ -d "$MODEL_DIR_1" ] && [ "$(ls -A "$MODEL_DIR_1" 2>/dev/null)" ]; then
    echo "🌀 [STEP 3b] Found existing model at $MODEL_DIR_1"
  else
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Wan-AI/Wan2.1-I2V-14B-480P-Diffusers', local_dir='$MODEL_DIR_1')"
  fi

  if [ -d "$MODEL_DIR_2" ] && [ "$(ls -A "$MODEL_DIR_2" 2>/dev/null)" ]; then
    echo "🌀 [STEP 3b] Found existing model at $MODEL_DIR_2"
  else
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='DAVIAN-Robotics/EgoX', local_dir='$MODEL_DIR_2', allow_patterns='*.safetensors')"
  fi
else
  echo "⏭️ [STEP 3b] Skipping model downloads (already present)"
fi

echo "✅ Python environment setup done"
exit 2>/dev/null
EOF

if [ $? -ne 0 ]; then
    echo "❌ Failed in STEP 3: venv and requirements setup"
    exit 1
fi

echo "🎉 DONE! Customized container is at $PWD/$CONTAINER_NAME"
