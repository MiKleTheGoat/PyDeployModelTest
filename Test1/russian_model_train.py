from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "MLNavigator/russian-retrieval"
SPLIT = "train"
OUT_DIR = "qwen25-15b-lora"

SYSTEM = (
    "Ты полезный ассистент. Отвечай по-русски кратко и по делу. " 
    "Также можешь подробнее разбирать вопрос, если пользователь попросил."
)

def to_str(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(map(str, x))
    return str(x)

def format_example(ex):
    user_text = to_str(ex.get("q", "")).strip()
    assistant_text = to_str(ex.get("a", "")).strip()
    ctx = to_str(ex.get("context", "")).strip()

    if ctx:
        assistant_text = f"{assistant_text}\n\nПояснение: {ctx}"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return text

# 1) dataset
ds = load_dataset(DATASET_NAME, split=SPLIT)
ds = ds.filter(lambda ex: len(to_str(ex.get("a", "")).strip()) > 0)

# 2) tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    dtype="auto",
)

# 3) LoRA
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 4) train args
args = SFTConfig(
    output_dir=OUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_length=512,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,          # если CUDA
    bf16=False,
    gradient_checkpointing=True,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    formatting_func=format_example,
    peft_config=peft_config,
    args=args,
)
try:
    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Done. Saved to:", OUT_DIR)
except Exception as e:
    print(f"Error accured: {e}")