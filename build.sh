#!/bin/bash
  
set -e
PROJECT_DIR_1="/home/sh57680/"
PROJECT_DIR_2="/mnt/thorai/"
PROJECT_DIR_3="/scratch/"
IMAGE_NAME="tangerine"

echo "📦 Step 1: Building container image..."
podman build -t $IMAGE_NAME .

echo "🚀 Step 2: Running container with GPU access..."
podman run -it --rm \
  --name tangerine_cont \
  --hooks-dir=/usr/share/containers/oci/hooks.d \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -e HF_HOME=/scratch/sh57680/hf_cache \
  -e HF_HUB_CACHE=/scratch/sh57680/hf_cache/hub \
  -e HF_XET_CACHE=/scratch/sh57680/hf_cache/xet \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$PROJECT_DIR_1":"$PROJECT_DIR_1" \
  -v "$PROJECT_DIR_2":"$PROJECT_DIR_2" \
  -v "$PROJECT_DIR_3":"$PROJECT_DIR_3" \
  -w /workspace/Med3DVLM \
  "$IMAGE_NAME"