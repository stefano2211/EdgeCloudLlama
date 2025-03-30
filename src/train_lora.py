import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    logging,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import subprocess
from datetime import datetime

# Configuración
os.environ["PYTHONUNBUFFERED"] = "1"
logging.set_verbosity_info()

def print_gpu_utilization():
    if torch.cuda.is_available():
        print(f"🖥️ GPU Memory Usage: Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB, Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")

def load_model_4bit(model_name):
    # Configuración para carga en 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print("🔍 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🚀 Cargando modelo en 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model, tokenizer

def main():
    try:
        start_time = datetime.now()
        print(f"⏰ Inicio del entrenamiento: {start_time}")
        
        # 1. Cargar modelo y tokenizer
        model_name = "meta-llama/Llama-3-8B"
        model, tokenizer = load_model_4bit(model_name)
        print_gpu_utilization()

        # 2. Configurar LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print_gpu_utilization()

        # 3. Cargar y preparar datos
        print("📂 Cargando dataset...")
        dataset = load_dataset("json", data_files="/data/mes_data_v1.json")

        def tokenize_function(examples):
            tokenized = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256,
                return_tensors="pt"
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        print("🔢 Tokenizando dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # 4. Configurar entrenamiento
        training_args = TrainingArguments(
            output_dir="/models/lora_llama",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=5,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            learning_rate=1e-4,
            fp16=True,
            optim="paged_adamw_8bit",
            report_to="none",
            gradient_checkpointing=True,
            save_total_limit=2,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
        )

        # 5. Entrenar y guardar
        print("🏋️ Comenzando entrenamiento...")
        trainer.train()
        
        print("💾 Guardando modelo...")
        model.save_pretrained("/models/lora_llama")
        tokenizer.save_pretrained("/models/lora_llama")

        # 6. Subir a Ollama automáticamente
        print("⬆️ Subiendo modelo a Ollama...")
        subprocess.run(["python3", "/data/scripts/upload_to_ollama.py"], check=True)

        end_time = datetime.now()
        duration = end_time - start_time
        print(f"✅ Entrenamiento completado en {duration}")
        print_gpu_utilization()

    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    main()