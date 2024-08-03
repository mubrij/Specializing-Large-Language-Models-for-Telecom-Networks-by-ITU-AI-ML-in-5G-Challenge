import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import datasets
from datasets import load_dataset
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from utils import remove_release_number, encode_answer, generate_prompt, llm_inference, get_results_with_labels

MODEL_PATH = 'microsoft/phi-2'
TUNED_MODEL_PATH = 'peft_phi_2'
USE_RAG = True

# Config to load model with a 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Read PHI-2 model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             device_map="auto",
                                             trust_remote_code=True,
                                             quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                          device_map="auto",
                                          trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def generate_prompt(data_point):
    return f"""
<human>: {data_point["question"]}
<assistant>: {data_point["answer"]}
""".strip()

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, max_length=512, padding="max_length", truncation=True)
    return tokenized_full_prompt

data = load_dataset("json", data_files="second_data.json")
dataset = data["train"].train_test_split(
                test_size=0.2, shuffle=True, seed=42) # ! Seed = 42 (?)
train_dataset, val_dataset = dataset["train"], dataset["test"]

data = train_dataset.shuffle().map(generate_and_tokenize_prompt)
val_data = val_dataset.shuffle().map(generate_and_tokenize_prompt)


model.gradient_checkpointing_enable()
# Prepare model for QLoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
peft_config = LoraConfig(task_type="CAUSAL_LM",
                         r=16,  # reduce if running into out-of-memory issues
                         lora_alpha=32,
                         target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
                         lora_dropout=0.05)
peft_model = get_peft_model(model, peft_config)
# Set training arguments, data collator for LLMs and Trainer
training_args = TrainingArguments(output_dir=TUNED_MODEL_PATH,
                                  learning_rate=1e-3,
                                  per_device_train_batch_size=10,  # reduce if running into out-of-memory issues
                                  num_train_epochs=5,  # reduce if running into out-of-memory issues
                                  weight_decay=0.01,
                                  eval_strategy='epoch',
                                  logging_steps=20,
                                  fp16=True,
                                  save_strategy='no')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(model=peft_model,
                  args=training_args,
                  train_dataset=data,
                  eval_dataset=val_data,
                  tokenizer=tokenizer,
                  data_collator=data_collator)
# Fine-tune the model
trainer.train()
model_final = trainer.model
# Save the fine-tuned model
model_final.save_pretrained(TUNED_MODEL_PATH)
