from scripts.synth_data import generate_questions, respond_to_questions, label_responses, train_model
from src.load_model import load_model
from src.utils import load_prompts

run_version = "py_dev_mini_1_full_loop_v1"

PROMPTS = "configs/prompts.yaml"
questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses"
evaluations_csv = "outputs/evaluations"
combined_responses_csv = "outputs/combined_responses.csv"

model_path = "../models/pydevmini_full"
output_path = f"../models/lora-epistemic-humility-v1-{run_version}"
dataset_path = f"../outputs/evaluations/{run_version}/00_{run_version}_Training_data.csv"

def run():
    llm, tokenizer = load_model()
    prompts = load_prompts(PROMPTS)['role']

    epoch = 0
    while True:
        print(f"Starting pipeline loop for epoch {epoch}...")

        #print("Step 1: Generating questions...")    ### Not needed, we're only generating the questions once and then they're in outputs/questions.csv
        #generate_questions(llm, tokenizer, prompts, questions_csv)

        print("Step 2: Responding to questions...")
        respond_to_questions(llm, tokenizer, prompts, run_version, responses_csv, epoch)

        print("Step 3: Labeling responses...")
        label_responses(llm, tokenizer, prompts, run_version, combined_responses_csv, evaluations_csv, epoch)

        print("Step 4: Training model...")
        train_model(run_version, model_path, output_path, dataset_path, epoch)

        print(f"Pipeline loop for epoch {epoch} completed. Restarting...")
        epoch += 1