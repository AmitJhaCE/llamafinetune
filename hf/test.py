import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "/workspace/llmfinetuning/basemodel"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

messages = ["On this wonderful day", "Another day in the battlefield,"]
for msg in messages:
    tokens = tokenizer(msg, return_tensors="pt").to("cuda")
    generated_ids = model.generate(**tokens, max_new_tokens=30)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(output)




