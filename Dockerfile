FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VENV_PATH=/opt/venv
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONUNBUFFERED=1

# Hugging Face settings
ENV HF_HOME=huggingface cache path here
ENV HF_HUB_CACHE=huggingface hub path here
ENV HF_XET_CACHE=XET path here
ENV HF_HUB_DISABLE_XET=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=60

# H100 = sm_90. A100 = sm_80.
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"

# ------------------------------------------------------------
# System packages + Python 3.12
# Ubuntu 24.04 provides Python 3.12 by default.
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    wget \
    ca-certificates \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    unzip \
    openssh-client \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Create Python virtual environment
# ------------------------------------------------------------
RUN python3.12 -m venv ${VENV_PATH} && \
    ${VENV_PATH}/bin/python -m pip install --upgrade pip setuptools wheel packaging ninja

WORKDIR /workspace

WORKDIR workspace path here
ENV PYTHONPATH=workspace path here:$PYTHONPATH

RUN python -m pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.5.0.post8

RUN python -m pip install --no-cache-dir \
    SimpleITK==2.4.1 \
    Unidecode==1.3.8 \
    accelerate==1.3.0 \
    bert-score==0.3.13 \
    deepspeed==0.16.3 \
    einops==0.8.1 \
    evaluate==0.4.3 \
    gradio==5.20.1 \
    monai==1.4.0 \
    nltk==3.9.1 \
    peft==0.14.0 \
    pyngrok==7.2.3 \
    rouge_score==0.1.2 \
    sentencepiece==0.2.0 \
    timm==1.0.14 \
    transformers==4.48.3 \
    wandb==0.19.6 \
    huggingface_hub[cli]

CMD ["/bin/bash"]