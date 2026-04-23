# chat_lora.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B-Instruct"
LORA = "qwen25-15b-lora"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype="auto")
model = PeftModel.from_pretrained(base, LORA)
model.eval()

messages = [{"role": "system", "content": "Ты полезный ассистент. Отвечай по-русски."}]

while True:
    q = input("User: ").strip()
    if q.lower() in {"exit", "quit"}:
        break

    messages.append({"role": "user", "content": q})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, temperature=0.7, top_p=0.9, do_sample=True)

    ans = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    print("AI:", ans, "\n")
    messages.append({"role": "assistant", "content": ans})