from src.load_model import load_model, run_inference
from src.generate import generate_responses
from src.evaluate import evaluate_responses
from src.utils import get_column_from_csv, extract_question, load_prompts, get_questions, get_responses, get_question_response_pairs, combine_csv_files
from pathlib import Path
import glob
from .train import TrainerWrapper
from tqdm import tqdm




run_version = "py_dev_mini_1"

task = "respond_to_questions"  # ( generate_questions | respond_to_questions | label_responses | training )




##### Paths

PROMPTS = "configs/prompts.yaml"
questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses/"
evaluations_csv = "outputs/evaluations/"

######

def generate_questions(llm, tokenizer, prompts, out_csv):
    question_prompt = prompts['question_generator']['prompt']
    generate_responses(question_prompt, llm, tokenizer, samples_per_prompt=10, out_csv=out_csv)

def respond_to_questions(llm, tokenizer, prompts, run_version, responses_csv, epoch):
    epoch_folder = f"{responses_csv}/{run_version}/{epoch}"
    Path(epoch_folder).mkdir(parents=True, exist_ok=True)

    answer_prompt = prompts['question_answerer']['prompt']
    questions = get_questions()
    for idx, question in tqdm(enumerate(questions), total=len(questions), desc="Responding to questions"):
        response_file = f"{epoch_folder}/responses_{idx}.csv"
        if Path(response_file).exists():
            print(f"Skipping question {idx}, response file already exists.")
            continue

        prompt_text = answer_prompt.replace("{question}", question)
        generate_responses(prompt_text, llm, tokenizer, samples_per_prompt=3, out_csv=response_file)

    # Combine individual response CSVs into a single file
    combined_csv_path = f"{epoch_folder}/combined_responses.csv"
    combine_csv_files(epoch_folder, combined_csv_path, pattern="responses_*.csv")

def label_responses(llm, tokenizer, prompts, run_version, responses_csv, evaluations_csv, epoch):
    epoch_responses_file = f"{responses_csv}/{run_version}/{epoch}/combined_responses.csv"
    epoch_evaluations_folder = f"{evaluations_csv}/{run_version}/{epoch}"
    Path(epoch_evaluations_folder).mkdir(parents=True, exist_ok=True)

    judge_prompt_template = prompts['judge']['prompt']
    question_response_pairs = get_question_response_pairs(path=epoch_responses_file)

    for idx, (question, response_text) in tqdm(enumerate(question_response_pairs), total=len(question_response_pairs), desc="Labeling responses"):
        evaluation_file = f"{epoch_evaluations_folder}/evaluation_sample_{idx}_question_{idx // 3}_repeat_{idx % 3}.csv"
        if Path(evaluation_file).exists():
            print(f"Skipping evaluation for sample {idx}, file already exists.")
            continue

        judge_prompt = judge_prompt_template.replace("{judge_prompt}", f"Prompt: {question}\nResponse: {response_text}")

        evaluate_responses(judge_prompt, response_text, llm, tokenizer,
                            samples_per_prompt=1, evaluations_per_sample=3,
                            out_csv=evaluation_file)

    # Combine individual evaluation CSVs into a single file
    combined_evaluations_path = f"{epoch_evaluations_folder}/combined_evaluations.csv"
    combine_csv_files(epoch_evaluations_folder, combined_evaluations_path, pattern="evaluation_sample_*.csv")

def train_model(run_version, model_path, output_path, dataset_path, epoch):
    if epoch == 0:
        current_model_path = model_path
    else:
        current_model_path = f"{output_path}/epoch_{epoch - 1}"

    epoch_output_path = f"{output_path}/epoch_{epoch}"
    trainer = TrainerWrapper(current_model_path, epoch_output_path, dataset_path)
    trainer.prepare_dataset()
    trainer.train()

