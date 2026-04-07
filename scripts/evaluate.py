import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import mlflow
from app.config import settings

def normalize_number(text):
    text = text.lower().strip()
    text = text.replace(",", "").replace("%", "").replace("$", "").replace("billion", "e9").replace("million", "e6").replace("thousand", "e3")
    numbers = re.findall(r"-?\d+\.?\d*(?:e[+-]?\d+)?", text)
    if numbers:
        try:
            return round(float(numbers[0]), 4)
        except:
            return None
    return None

def is_correct(pred, gold):
    # Try exact string match first
    if str(gold).lower() in pred.lower():
        return True
    # Try numeric match
    pred_num = normalize_number(pred)
    gold_num = normalize_number(str(gold))
    if pred_num is not None and gold_num is not None:
        if gold_num == 0:
            return pred_num == 0
        return abs(pred_num - gold_num) / abs(gold_num) < 0.01  # 1% tolerance
    return False

with open(settings.data_path) as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(settings.model_path, quantization_config=bnb_config)
model = PeftModel.from_pretrained(model, settings.finetuned_path)
model.eval()

def get_prediction(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=settings.max_seq_length).to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=settings.max_new_tokens, temperature=settings.temperature)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip().split("\n")[0]
    return answer

correct = 0
total = len(data)

for i, item in enumerate(data):
    context = " ".join([v for _, v in item["qa"]["model_input"]])
    question = item["qa"]["question"]
    gold = item["qa"]["exe_ans"]
    pred = get_prediction(context, question)
    if is_correct(pred, gold):
        correct += 1
    if i % 50 == 0:
        print(f"Step {i}/{total} — Running accuracy: {correct/(i+1)*100:.2f}%")

accuracy = correct / total * 100
print(f"\nFinal Execution Accuracy: {accuracy:.2f}%")

mlflow.set_tracking_uri(settings.mlflow_uri)
mlflow.set_experiment("finqa-mistral-lora")
with mlflow.start_run():
    mlflow.log_metric("execution_accuracy", accuracy)