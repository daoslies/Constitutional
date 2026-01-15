import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

class TrainerWrapper:
    def __init__(self, 
                 base_model_path: str,
                 output_path: str,
                 dataset_path: str,
                 run_version: str,
                 epoch: int):

        self.run_version = run_version
        self.base_model_path = base_model_path
        self.base_output_path = output_path
        self.dataset_path = dataset_path

        self.SL_CAI_epoch = epoch

        self.current_output_path = f"{output_path}/epoch_{epoch}"
        self.lora_adapter_path = f"{output_path}/epoch_{epoch - 1}" if epoch > 0 else None

        print(f"Base Model: {self.base_model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            return_special_tokens_mask=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
        ).half()

        self.model = self.model.to(torch.device("cuda"))

        if self.SL_CAI_epoch == 0:

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
            print("Initialized LoRA layers for SL-CAI training.")

                     
        if self.SL_CAI_epoch > 0 and self.lora_adapter_path is not None:
            print(f"Loading LoRA adapter from previous epoch at: {self.lora_adapter_path}")

            self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_adapter_path,
                    is_trainable=True,
                    )
            

            for name, param in self.model.named_parameters():
                if param.requires_grad and "lora" not in name:
                    raise RuntimeError(f"Non-LoRA param trainable: {name}")
                

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
            output_dir=self.current_output_path,
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
        self.model.save_pretrained(self.current_output_path)
        self.tokenizer.save_pretrained(self.current_output_path)

# Example usage:
# trainer = TrainerWrapper(
#     model_path="../models/pydevmini_full",
#     output_path="../models/lora-epistemic-humility-v1-pydevmini",
#     dataset_path="../outputs/evaluations/py_dev_mini_1/00_py_dev_mini_1_Training_data.csv",
# )
# trainer.prepare_dataset()
# trainer.train()
