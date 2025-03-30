FROM python:3.10

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
RUN pip install transformers==4.38.2 peft==0.9.0 datasets accelerate==0.27.2 sentencepiece protobuf
RUN pip install -i https://pypi.org/simple/ bitsandbytes

# Configurar el directorio de trabajo
WORKDIR /data

# Comando por defecto (mantiene el contenedor activo)
CMD ["tail", "-f", "/dev/null"]