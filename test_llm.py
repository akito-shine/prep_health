from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_name = "HuggingFaceH4/zephyr-7b-alpha"


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

prompt = "Hello! Summarize: I am building a PrEP intake chatbot."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # or "cpu"
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))