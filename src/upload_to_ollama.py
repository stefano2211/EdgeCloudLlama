import subprocess
import time
import os
from datetime import datetime

def check_ollama_ready():
    max_retries = 10
    retry_delay = 10
    
    for i in range(max_retries):
        try:
            result = subprocess.run(
                ["docker", "exec", "ollama", "ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            print(f"🔄 Ollama no está listo, reintentando ({i+1}/{max_retries})...")
            time.sleep(retry_delay)
    
    return False

def upload_custom_model():
    try:
        start_time = datetime.now()
        print(f"⏰ Inicio de carga del modelo: {start_time}")
        
        # 1. Verificar que Ollama esté listo
        if not check_ollama_ready():
            raise RuntimeError("Ollama no está disponible después de múltiples intentos")

        # 2. Verificar/descargar modelo base
        base_model = "llama3:8b"
        print(f"🔍 Verificando modelo base {base_model}...")
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "list"],
            capture_output=True,
            text=True
        )
        
        if base_model not in result.stdout:
            print(f"⬇️ Descargando modelo base {base_model}...")
            subprocess.run(
                ["docker", "exec", "ollama", "ollama", "pull", base_model],
                check=True
            )

        # 3. Copiar template Modelfile
        print("📄 Preparando Modelfile...")
        subprocess.run(["cp", "/data/scripts/template_modelfile", "/models/Modelfile"], check=True)

        # 4. Crear modelo personalizado
        print("🛠️ Creando modelo personalizado...")
        subprocess.run(
            ["docker", "exec", "ollama", "ollama", "create", "llama3-custom", "-f", "/models/Modelfile"],
            check=True
        )

        # 5. Verificar creación
        print("🔎 Verificando modelo creado...")
        result = subprocess.run(
            ["docker", "exec", "ollama", "ollama", "list"],
            capture_output=True,
            text=True
        )
        
        if "llama3-custom" in result.stdout:
            end_time = datetime.now()
            duration = end_time - start_time
            print(f"✅ Modelo 'llama3-custom' creado exitosamente en {duration}!")
            print("🌐 Accede a la interfaz web en: http://localhost:3001")
        else:
            raise RuntimeError("El modelo no se creó correctamente")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error en el proceso: {str(e)}")
        print(e.stderr)
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")

if __name__ == "__main__":
    upload_custom_model()