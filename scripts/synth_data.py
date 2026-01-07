from pathlib import Path
import sys
from src.load_model import Llama, load_model, run_inference

import yaml
import csv
import argparse
import time
from datetime import datetime

PROMPTS = "configs/prompts.yaml"

questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses/"


def load_prompts(path=PROMPTS):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_prompt(prompts, category, name):
    return prompts["prompts"][category][name]

#

def get_column_from_csv(path, column_name):

    data = []

    with Path(path).open(newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:
            q = row.get(column_name, None)
            if q is None:
                continue
            else:
                data.append(q)

    return data

def get_questions(path=questions_csv):

    questions = get_column_from_csv(path, "question")

    return questions

def get_responses(path=responses_csv, epoch = 0):

    epoch = 0

    path = Path(path) / str(epoch)

    response_paths = list(Path(path).glob("responses_*.csv"))
    sorted_paths = sorted(response_paths, key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
    
    responses = []
    for idx, path in enumerate(sorted_paths):

        responses.extend(get_column_from_csv(path, "response_text"))

    return responses

def get_responses_original_prompts(path=questions_csv):

    prompts = get_column_from_csv(path, "original_prompt")

    return prompts

#


def generate_responses(prompt, llm, samples_per_prompt=10, out_csv="outputs/responses.csv"):
    """Generate responses for a single prompt and save to CSV."""
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["timestamp", "sample_index", "response_text", "original_prompt"]) 
        writer.writeheader()
        
        for i in range(samples_per_prompt):
            try:
                result = run_inference(prompt, llm)
                text = result.get("choices", [{}])[0].get("text", str(result)) if isinstance(result, dict) else str(result)
            except Exception as e:
                text = f"<ERROR: {e}>"
            
            writer.writerow({
                "timestamp": datetime.utcnow().isoformat(),
                "sample_index": i,
                "response_text": text.replace("\n", "\\n"),
                "original_prompt": prompt.replace("\n", "\\n"),
            })



def run():

    task = "label_responses"   # "generate_questions | respond_to_questions | label_responses"
    epoch = "codebug"  # e.g., "0", "1", "2", ..., "codebug"

    llm = load_model()
    
    prompts = load_prompts(Path(PROMPTS))['role']


    if task == "generate_questions":
        question_generator_prompt = prompts['question_generator']['prompt']
        generate_responses(question_generator_prompt, llm, samples_per_prompt=10, out_csv="outputs/questions.csv")

    elif task == "respond_to_questions":
        question_answerer_prompt = prompts['question_answerer']['prompt']

        for idx, question in enumerate(get_questions()):
            question_prompt = question_answerer_prompt.replace("{question}", question)
            generate_responses(question_prompt, llm, samples_per_prompt=3, out_csv=f"outputs/responses/{epoch}/responses_{idx}.csv")

    elif task == "label_responses":
        judge_prompt = prompts['judge']['prompt']

        original_questions = get_questions(path=questions_csv)

        for idx, response in enumerate(get_responses()):
            judge_prompt = judge_prompt.replace("{judge_prompt}", "Prompt: " + original_questions[idx] + "\n Response: " + response)

            print(judge_prompt)
            generate_responses(judge_prompt, llm, samples_per_prompt=3, out_csv=f"outputs/judging/{epoch}/responses_{idx}.csv")
            print(f'Response generated for judging. {idx}')
