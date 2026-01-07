"""
src/load_model.py
"""



from transformers import pipeline

model_id = "../models/jinx-gpt-oss-20b-Q4_K_M.gguf"  # https://huggingface.co/papers/2508.08243 - Jinx: Unlimited LLMs for Probing Alignment Failures

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])