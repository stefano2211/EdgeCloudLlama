# Fine-Tuning de Llama3 con LoRA (Soporte GPU)

Proyecto completo para fine-tuning de Llama3-8B con soporte GPU y carga automática en Ollama.

## Requisitos Previos

1. **Hardware**:
   - GPU NVIDIA con al menos 16GB VRAM
   - Drivers NVIDIA >= 525.60.13

2. **Software**:
   ```bash
   # Instalar NVIDIA Container Toolkit
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/$(arch) /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker