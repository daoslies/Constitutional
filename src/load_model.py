#from llama_cpp import Llama, llama_cpp
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

"""
src/load_model.py

"""
#jinx-gpt-oss-20b-Q4_K_M.gguf  | pydevmini1-q8_0.gguf | Qwen3-4B-Instruct-2507-Q8_0.gguf | pydevmini_full


# https://huggingface.co/papers/2508.08243 - Jinx: Unlimited LLMs for Probing Alignment Failures


#model_path = "models/pydevmini_full"  


## full fp16 model

def load_model(model_path: str):
    # Load the tokenizer
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="float16",
    )

    # If there’s a LoRA checkpoint, wrap the model
    try:
        model = PeftModel.from_pretrained(model, model_path)
        print("Loaded LoRA weights into base model.")
    except Exception as e:
        print("No LoRA weights found or error loading LoRA -- This is fine if it's the evaluation/judgement step:", e)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
        print(f"Model loaded to CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU")

    return model, tokenizer


def unload_model(model, tokenizer=None):
    model.to("cpu")
    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # optional


def unload_trainer(trainer_wrapper):
    trainer_wrapper.to_cpu()
    del trainer_wrapper.model
    del trainer_wrapper.tokenizer
    if hasattr(trainer_wrapper, "dataset"):
        del trainer_wrapper.dataset
    del trainer_wrapper
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()  # optional


def run_inference(prompt: str, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response





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