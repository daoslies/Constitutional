from src.load_model import load_model, run_inference
from src.generate import generate_responses
from src.evaluate import evaluate_responses
from src.utils import get_column_from_csv, extract_question, load_prompts, get_questions, get_responses, get_question_response_pairs
from pathlib import Path
import glob




run_version = "py_dev_mini_1"

task = "respond_to_questions"  # ( generate_questions | respond_to_questions | label_responses )




##### Paths

PROMPTS = "configs/prompts.yaml"
questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses/"

######

def run(task=task, run_version=run_version): 
    llm, tokenizer = load_model()
    prompts = load_prompts(PROMPTS)['role']

    if task == "generate_questions":
        question_prompt = prompts['question_generator']['prompt']
        generate_responses(question_prompt, llm, tokenizer, samples_per_prompt=10, out_csv=questions_csv)

    elif task == "respond_to_questions":
        answer_prompt = prompts['question_answerer']['prompt']
        for idx, question in enumerate(get_questions()):
            prompt_text = answer_prompt.replace("{question}", question)
            generate_responses(prompt_text, llm, tokenizer, samples_per_prompt=3,
                               out_csv=f"outputs/responses/{run_version}/responses_{idx}.csv")

    elif task == "label_responses":
        judge_prompt_template = prompts['judge']['prompt']

        # iterate over responses CSVs -> for when response csv spreads over multiple files, get rid of when fully automated.
        # You can combine csvs via scripts/auxiliary.py
        #response_paths = sorted((Path(responses_csv)/run_version).glob("responses_*.csv"), key=lambda x: int(x.stem.split('_')[-1]))

        for idx, (question, response_text) in enumerate(get_question_response_pairs(run_version=run_version)):
            question_idx = idx // 3
            response_idx = idx % 3
            print(f"Evaluating sample {idx} question {question_idx} response {response_idx}")

            judge_prompt = judge_prompt_template.replace("{judge_prompt}", f"Prompt: {question}\nResponse: {response_text}")

            evaluate_responses(judge_prompt, response_text, llm, tokenizer,
                                samples_per_prompt=1, evaluations_per_sample=3,
                                out_csv=f"outputs/evaluations/{run_version}/evaluation_sample_{idx}_question_{question_idx}_repeat_{response_idx}.csv")


