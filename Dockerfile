FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Configurar Python 3.10 como predeterminado
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Instalar dependencias de Python con soporte CUDA
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install transformers==4.38.2 peft==0.9.0 datasets accelerate==0.27.2 sentencepiece protobuf && \
    pip3 install bitsandbytes>=0.41.1 scipy

WORKDIR /data

CMD ["python3", "/data/scripts/train_lora.py"]