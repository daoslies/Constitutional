from pathlib import Path
import csv
import yaml
import re
import pandas as pd


PROMPTS = "configs/prompts.yaml"
questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses/"
run_version = "py_dev_mini_1"

def load_prompts(path=PROMPTS):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_questions(path=questions_csv):
    return get_column_from_csv(path, "question")

def get_responses(path=responses_csv, run_version=run_version):
    path = Path(path + run_version + "/combined_responses.csv")
    return get_column_from_csv(path, "response")


def get_question_response_pairs(path=responses_csv, run_version=run_version):
    path = Path(path + run_version + "/combined_responses.csv")
    data = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", None)
            response = row.get("response", None)
            if question and response:
                data.append((question, response))
    return data


def get_column_from_csv(path, column_name):
    data = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get(column_name, None)
            if val:
                data.append(val)
    return data

def extract_question(prompt: str) -> str:
    if "Question:\n" in prompt:
        return prompt.split("Question:\n")[-1].strip()
    return prompt.strip()

def extract_response(prompt: str, full_text: str) -> str:
    question_text = extract_question(prompt)
    idx = full_text.rfind(question_text)
    if idx != -1:
        return full_text[idx + len(question_text):].strip()
    return full_text.strip()
    
def extract_question_from_evaluation_prompt(evaluation_prompt: str) -> str:
    match = re.search(
        r"-----\s*Prompt:\s*(.*?)\s*Response:",
        evaluation_prompt,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return "<UNKNOWN QUESTION>"


### dealing with cleaning up the evaluation output

STRUCTURED_FIELDS = ["Critique:", "Score:", "Revised Response:"]

def extract_final_structured_block(text: str) -> str:
    """
    Extract the final structured evaluation block, regardless of field order.
    Anchors on the last occurrence of any known field header.
    """
    last_idx = -1
    for field in STRUCTURED_FIELDS:
        idx = text.rfind(field)
        if idx > last_idx:
            last_idx = idx

    if last_idx == -1:
        return ""

    return text[last_idx:]

def combine_csv_files(folder_path, output_file, pattern="*.csv"):
    folder = Path(folder_path)
    csv_files = list(folder.glob(pattern))
    if not csv_files:
        print(f"No CSV files found in {folder_path} to combine.")
        return

    combined_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")
