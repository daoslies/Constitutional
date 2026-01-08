#from llama_cpp import Llama, llama_cpp
import torch

"""
src/load_model.py
"""
#jinx-gpt-oss-20b-Q4_K_M.gguf  | pydevmini1-q8_0.gguf | Qwen3-4B-Instruct-2507-Q8_0.gguf | pydevmini_full

model_path = "models/pydevmini_full"  # https://huggingface.co/papers/2508.08243 - Jinx: Unlimited LLMs for Probing Alignment Failures


# print(llama_cpp.llama_supports_gpu_offload())
"""
### Llama CPP model loading and inference functions

def load_model():
    return Llama(
        model_path=model_path,
        n_ctx=4096,          # context size
        n_threads=16,         # CPU threads
        n_gpu_layers=21,     # -1 = offload as many layers as possible to GPU
        verbose=False,
        chat_format="none"
    )

def run_inference(prompt: str, llm: Llama):
    return llm(
        prompt,
        max_tokens=256,
        temperature=0.7,
        stop=["</s>"],
    )

"""

## full fp16 model

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = model_path


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float16",
        device_map="auto",
    )

    # Move all parameters to GPU
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Model loaded to CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")


    return model, tokenizer

def run_inference(prompt: str, model, tokenizer):

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,  #256
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response
