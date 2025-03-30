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

# Configuración
os.environ["PYTHONUNBUFFERED"] = "1"
logging.set_verbosity_info()

def load_model_4bit(model_name):
    # Configuración para carga en 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.float16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return model, tokenizer

def main():
    # 1. Cargar modelo y tokenizer
    model_name = "meta-llama/Llama-3.1-8B"
    model, tokenizer = load_model_4bit(model_name)

    # 2. Configurar LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Cargar y preparar datos
    dataset = load_dataset("json", data_files="/data/mes_data_v1.json")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128  # Aumentamos el contexto
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 4. Configurar entrenamiento
    training_args = TrainingArguments(
        output_dir="/models",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],

    )

    # 5. Entrenar y guardar
    trainer.train()
    model.save_pretrained("/models/lora_llama")
    tokenizer.save_pretrained("/models/lora_llama")

if __name__ == "__main__":
    main()