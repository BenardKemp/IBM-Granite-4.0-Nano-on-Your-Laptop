from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "ibm-granite/granite-4.0-micro"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",
)

inputs = tokenizer("Explain how edge AI works:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))