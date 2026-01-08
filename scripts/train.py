from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bralynn/pydevmini1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="float16",
    device_map="auto",
)


