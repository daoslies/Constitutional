from llama_cpp import Llama, llama_cpp


"""
src/load_model.py
"""
#jinx-gpt-oss-20b-Q4_K_M.gguf  | pydevmini1-q8_0.gguf | Qwen3-4B-Instruct-2507-Q8_0.gguf

model_path = "models/pydevmini1-q8_0.gguf"  # https://huggingface.co/papers/2508.08243 - Jinx: Unlimited LLMs for Probing Alignment Failures


# print(llama_cpp.llama_supports_gpu_offload())


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
