import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

class TrainerWrapper:
    def __init__(self, model_path, output_path, dataset_path):
        self.model_path = model_path
        self.output_path = output_path
        self.dataset_path = dataset_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            return_special_tokens_mask=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
        ).half()

        self.model = self.model.to(torch.device("cuda"))

        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def to_cpu(self):
        return self.model.to("cpu")

    def prepare_dataset(self):
        dataset = load_dataset("csv", data_files=self.dataset_path)

        def extract_training_pairs(example):
            prompt = example["question"]
            response = example["revised_response"]
            text = f"User: {prompt}\nAssistant: {response}"
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=102,
                padding=False,
            )
            return tokens

        self.dataset = dataset.map(extract_training_pairs, remove_columns=dataset["train"].column_names)

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.output_path,
            optim="paged_adamw_8bit",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-5,
            num_train_epochs=3,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            fp16=False,
            bf16=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            data_collator=data_collator,
        )

        trainer.train()
        self.model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)

# Example usage:
# trainer = TrainerWrapper(
#     model_path="../models/pydevmini_full",
#     output_path="../models/lora-epistemic-humility-v1-pydevmini",
#     dataset_path="../outputs/evaluations/py_dev_mini_1/00_py_dev_mini_1_Training_data.csv",
# )
# trainer.prepare_dataset()
# trainer.train()