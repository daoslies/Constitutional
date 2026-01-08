from pathlib import Path
from datetime import datetime
import csv
from .utils import extract_question, extract_response
from .load_model import run_inference

def generate_responses(prompt: str, llm, tokenizer, samples_per_prompt=3, out_csv="outputs/responses.csv"):
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    question_text = extract_question(prompt)

    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "sample_index", "question", "response"])
        writer.writeheader()

        for i in range(samples_per_prompt):
            try:
                result = run_inference(prompt, llm, tokenizer)
                text = result.get("choices", [{}])[0].get("text", str(result)) if isinstance(result, dict) else str(result)
                text = extract_response(prompt, text)
            except Exception as e:
                text = f"<ERROR: {e}>"

            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "sample_index": i,
                "question": question_text.replace("\n", " "),
                "response": text.replace("\n", " "),
            })
