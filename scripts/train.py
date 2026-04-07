import sys
sys.path.append("/work/MLShare/rek21muv/mistral")

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import mlflow
from app.config import settings

def load_finqa(split):
    path = settings.data_path.replace("dev.json", f"{split}.json")
    with open(path) as f:
        data = json.load(f)
    samples = []
    for item in data:
        context = " ".join([v for _, v in item["qa"]["model_input"]])
        question = item["qa"]["question"]
        answer = item["qa"]["answer"]
        text = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        samples.append({"text": text})
    return Dataset.from_list(samples)

train_ds = load_finqa("train")
eval_ds = load_finqa("dev")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    settings.model_path,
    quantization_config=bnb_config
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

mlflow.set_tracking_uri(settings.mlflow_uri)
mlflow.set_experiment("finqa-mistral-lora")

training_args = TrainingArguments(
    output_dir="/work/MLShare/rek21muv/mistral/models/mistral-7b-finqa-v2",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=settings.learning_rate,
    fp16=True,
    logging_steps=390,
    evaluation_strategy="steps",
    eval_steps=390,
    save_steps=390,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="mlflow"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=settings.max_seq_length,
)

with mlflow.start_run():
    mlflow.log_params({
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": settings.learning_rate,
        "epochs": 20,
        "batch_size": 4,
        "resumed_from": "checkpoint-1000"
    })
    trainer.train(resume_from_checkpoint="/work/MLShare/rek21muv/mistral/models/mistral-7b-finqa/checkpoint-1000")
    if trainer.state.best_metric is not None:
        mlflow.log_metric("final_eval_loss", trainer.state.best_metric)