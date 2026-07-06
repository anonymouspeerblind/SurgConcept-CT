#!/bin/bash

set -e
PROJECT_DIR_1="working code dirctory"
PROJECT_DIR_2="data directory"
PROJECT_DIR_3="scratch directory"
IMAGE_NAME="tangerine"

echo "📦 Step 1: Building container image..."
podman build -t $IMAGE_NAME .

echo "🚀 Step 2: Running container with GPU access..."
podman run -it --rm \
  --name tangerine_cont \
  --hooks-dir=hooks dir path here \
  --device nvidia.com/gpu=all \
  --security-opt=label=disable \
  -e HF_HOME= huggingface cache path here \
  -e HF_HUB_CACHE= huggingface hub path here \
  -e HF_XET_CACHE= XET path here \
  -e HF_HUB_DISABLE_XET=1 \
  -e HF_HUB_DOWNLOAD_TIMEOUT=60 \
  -v "$PROJECT_DIR_1":"$PROJECT_DIR_1" \
  -v "$PROJECT_DIR_2":"$PROJECT_DIR_2" \
  -v "$PROJECT_DIR_3":"$PROJECT_DIR_3" \
  -w workspace path here \
  "$IMAGE_NAME"