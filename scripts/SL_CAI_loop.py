from scripts.synth_data import generate_questions, respond_to_questions, label_responses, train_model
from src.load_model import load_model, unload_model, unload_trainer
from src.utils import load_prompts

run_version = "py_dev_mini_1_full_loop_v3_lr5e4"

PROMPTS = "configs/prompts.yaml"
questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses"
evaluations_csv = "outputs/evaluations"
combined_responses_csv = "outputs/combined_responses.csv"

model_path = "models/pydevmini_full"
output_path = f"models/lora-epistemic-humility-v1-{run_version}"
dataset_path = f"outputs/evaluations/{run_version}/00_{run_version}_Training_data.csv"

def run():
    base_model_path = "models/pydevmini_full"
    trained_model_path_template = output_path + f"/epoch_{{epoch}}"

    prompts = load_prompts(PROMPTS)['role']

    epoch = 0
    while True:
        print(f"Starting pipeline loop for epoch {epoch}...")

        # Load the base model for responding on the first epoch
        # and the trained model for subsequent epochs
        llm, tokenizer = load_model(base_model_path if epoch == 0 else trained_model_path_template.format(epoch=epoch - 1))
        print("Step 2: Responding to questions...")
        respond_to_questions(llm, tokenizer, prompts, run_version, responses_csv, epoch)

        unload_model(llm, tokenizer) ## free up vram

        # Load the base model for labeling
        llm, tokenizer = load_model(base_model_path)  ## Base model is always used as judge
        print("Step 3: Labeling responses...")
        label_responses(llm, tokenizer, prompts, run_version, responses_csv, evaluations_csv, epoch)

        unload_model(llm, tokenizer) ## free up vram
        
        """        ## Uneeded as we've moved this into train_model's trainer()
        # Load the freshly trained model for training
        if epoch == 0:
            model_path = base_model_path
        else:
            model_path = trained_model_path_template.format(epoch=epoch - 1)
        """

        # Update dataset path to use the most recent combined evaluations file for the current epoch
        dataset_path = f"outputs/evaluations/{run_version}/{epoch}/combined_evaluations.csv"
        
        print("Step 4: Training model...")
        #trainer = train_model(run_version, base_model_path, output_path, dataset_path, epoch)
        
        #unload_trainer(trainer) ## free up vram

        print(f"Pipeline loop for epoch {epoch} completed. Restarting...")
        epoch += 1