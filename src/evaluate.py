from pathlib import Path
from datetime import datetime
import csv
import re
from .utils import extract_question, extract_response, extract_question_from_evaluation_prompt, extract_final_structured_block
from .load_model import run_inference

import re


## Parse Evaluation

### The below functions feel a little convoluted, but they're to deal with messy LLM outputs.
### sometimes nlp just is a bit convoluted.

def strip_conversation_blocks(text: str):
    
    """
    Remove everything above the last conversation block of the form:
        -----
        Prompt: ...
        Response: ...
        -----
    Returns the remaining text (where the actual evaluation starts).
    """
    # Find all '-----' blocks that contain 'Prompt:' and 'Response:'
    blocks = list(re.finditer(r"-----\s*Prompt:.*?Response:.*?-----", text, flags=re.DOTALL))
    if blocks:
        # Keep only text after the last block
        last_block_end = blocks[-1].end()
        return text[last_block_end:].strip()
    else:
        # If no blocks found, return text as-is
        return text.strip()


def parse_evaluation(evaluation_text: str):
    """
    Robustly parse 'Score', 'Critique', and 'Revised Response' from LLM evaluation output.
    Handles:
    - Drunken LLM output with extra line breaks
    - Multiple 'Revised Response (continued):' sections
    """

    text = strip_conversation_blocks(evaluation_text)

    print('================= PARSING EVALUATION TEXT =================')
    print(text[:1000] + ('...' if len(text) > 1000 else ''))
    print('===========================================================')

    # ---------- SCORE ----------
    score_match = re.search(r"Score:\s*(\d+)", text, flags=re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None

    # ---------- CRITIQUE ----------
    critique_match = re.search(r"Critique:\s*(.*)", text, flags=re.IGNORECASE)
    critique = critique_match.group(1).strip() if critique_match else ""

    # ---------- REVISED RESPONSE ----------
    revised_match = re.search(r"Revised Response:\s*(.*)", text, flags=re.IGNORECASE)
    revised = revised_match.group(1).strip() if revised_match else ""

    # Optional: truncate for CSV
    max_sample_length = 1000
    critique = (critique[:max_sample_length] + "...") if len(critique) > max_sample_length else critique
    revised = (revised[:max_sample_length] + "...") if len(revised) > max_sample_length else revised

    return score, critique, revised


## Saving Evaluation


def write_evaluation_csv(out_csv, sample_index, question, response, score, critique, revised, error=None):
    """
    Clean evaluation CSV with columns:
    timestamp, sample_index, question, score, response, critique, revised_response, error
    """
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    question = extract_question_from_evaluation_prompt(question) # filters out the system prompt

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "timestamp", "sample_index", "score", "question", 
            "response", "critique", "revised_response", "error"
        ])
        if out_path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.utcnow().isoformat(),
            "sample_index": sample_index,
            "score": score,
            "question": question.replace("\n", " "),
            "response": extract_response(question, response).replace("\n", " "),
            "critique": critique.replace("\n", " "),
            "revised_response": revised.replace("\n", " "),
            "error": error
        })

## Run Evaluation

def evaluate_responses(prompt, response_text, llm, tokenizer, samples_per_prompt=1, evaluations_per_sample=3, out_csv="outputs/evaluations.csv"):
    question_text = extract_question(prompt)



    for sample_idx in range(samples_per_prompt):
        # use the same response for multiple evaluations
        try:
            error = None
        except Exception as e:
            response_text = f"<ERROR: {e}>"
            error = str(e)

        scores = []
        for eval_idx in range(evaluations_per_sample):
            if error:
                score, critique, revised = 1, f"System error: {error}", ""
            else:
                eval_prompt = f"Evaluate this response:\nPrompt: {question_text}"
                eval_output = run_inference(eval_prompt, llm, tokenizer)
                eval_text = eval_output.get("choices", [{}])[0].get("text", str(eval_output)) if isinstance(eval_output, dict) else str(eval_output)
                score, critique, revised = parse_evaluation(eval_text)
                if score is None:
                    score = 1
            scores.append(score)
            write_evaluation_csv(
                out_csv,
                sample_index=f"{sample_idx}_{eval_idx}",
                question=question_text,
                response=response_text,
                score=score,
                critique=critique,
                revised=revised,
                error=error
            )

        avg_score = sum(scores) / len(scores)
        print(f"Sample {question_text}: average score = {avg_score:.2f}")
        break
