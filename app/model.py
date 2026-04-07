import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from app.config import settings

def load_model():
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
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, settings.finetuned_path)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, context: str, question: str) -> str:
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.max_seq_length
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip().split("\n")[0]
    return answer