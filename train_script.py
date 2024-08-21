from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import os

class TinyLlamaFinetuner:
    def __init__(self, model_name, dataset_name, output_dir, device="0"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        # Load the model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto",
                                                          trust_remote_code=False,
                                                          revision="main")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Prepare the model for k-bit training
        self.model.train()
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

    def configure_lora(self, r=8, lora_alpha=32, target_modules=["q_proj"], lora_dropout=0.05):
        # Configure LoRA
        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()

    def load_and_tokenize_dataset(self):
        # Load and tokenize dataset
        dataset = load_dataset(self.dataset_name)

        def tokenize_function(batch):
            texts = batch['prompt']
            self.tokenizer.truncation_side = "left"
            tokenized_inputs = self.tokenizer(
                texts,
                return_tensors="np",
                truncation=True,
                max_length=2048
            )
            return tokenized_inputs

        tokenized_data = dataset.map(tokenize_function, batched=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenized_data = tokenized_data

    def fine_tune_model(self, lr=2e-4, batch_size=4, num_epochs=10):
        # Fine-tune the model
        data_collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        training_args = transformers.TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            fp16=True,
            optim="paged_adamw_8bit",
        )

        trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.tokenized_data["train"],
            eval_dataset=self.tokenized_data["eval"],
            args=training_args,
            data_collator=data_collator
        )

        trainer.train()
        
    def push_to_hub(self, hf_name):
        # Push the model and trainer to Hugging Face Hub
        model_id = hf_name + "/" + self.output_dir
        self.model.push_to_hub(model_id)
        self.trainer.push_to_hub(model_id)

# Usage
finetuner = TinyLlamaFinetuner(
    model_name="TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ",
    dataset_name='vishals9711/tiny_llama_headline',
    output_dir="vishals9711_ft"
)

finetuner.configure_lora()
finetuner.load_and_tokenize_dataset()
finetuner.fine_tune_model()