from src.load_model import load_model, run_inference
from src.generate import generate_responses
from src.evaluate import evaluate_responses
from src.utils import get_column_from_csv, extract_question, load_prompts, get_questions
from pathlib import Path
import glob


PROMPTS = "configs/prompts.yaml"
questions_csv = "outputs/questions.csv"
responses_csv = "outputs/responses/"



######

def run(task="respond_to_questions", epoch="py_dev_mini_1"):
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
                               out_csv=f"outputs/responses/{epoch}/responses_{idx}.csv")

    elif task == "label_responses":
        judge_prompt_template = prompts['judge']['prompt']
        questions = get_questions()

        print(questions[0])
        # iterate over responses CSVs
        response_paths = sorted((Path(responses_csv)/epoch).glob("responses_*.csv"), key=lambda x: int(x.stem.split('_')[-1]))
        for idx, path in enumerate(response_paths):
            responses = get_column_from_csv(path, "response")
            for resp_idx, response_text in enumerate(responses):
                judge_prompt = judge_prompt_template.replace("{judge_prompt}", f"Prompt: {questions[idx]}\nResponse: {response_text}")
                evaluate_responses(judge_prompt, response_text, llm, tokenizer,
                                   samples_per_prompt=1, evaluations_per_sample=3,
                                   out_csv=f"outputs/evaluations/{epoch}/responses_{idx}_{resp_idx}.csv")
