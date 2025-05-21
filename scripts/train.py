#!/usr/bin/env python
# Core training script for Hugging Face Transformers models

import os
import argparse
import yaml
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import evaluate

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformers model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Path to training configuration YAML")
    parser.add_argument("--model_name", type=str, help="Hugging Face model name or path (overrides config)")
    parser.add_argument("--data_path", type=str, help="Path to training data (overrides config)")
    parser.add_argument("--output_dir", type=str, help="Path to save model outputs (overrides config)")
    return parser.parse_args()

# Load configuration from YAML
def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")

# Compute metrics for evaluation
def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1.compute(predictions=predictions, references=labels, average="weighted")["f1"],
    }

def main():
    args = parse_args()
    config = load_config(args.config)

    # Override config with CLI args if provided
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.data_path:
        config["data"]["train_file"] = args.data_path
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir

    # Set seed for reproducibility
    set_seed(config["training"]["seed"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"], num_labels=config["model"]["num_labels"]
    )

    # Load dataset
    data_files = {"train": config["data"]["train_file"]}
    if "validation_file" in config["data"] and os.path.exists(config["data"]["validation_file"]):
        data_files["validation"] = config["data"]["validation_file"]

    for split, file_path in data_files.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{split} file not found: {file_path}")

    extension = data_files["train"].split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)

    # Preprocessing
    def preprocess_function(examples):
        return tokenizer(
            examples[config["data"]["text_column"]],
            truncation=True,
            padding="max_length",
            max_length=config["data"]["max_length"]
        )

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=[col for col in raw_datasets["train"].column_names if col != config["data"]["label_column"]]
    )

    # Training arguments with dynamic extension support
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        weight_decay=config["training"]["weight_decay"],
        logging_dir=config["training"]["logging_dir"],
        logging_steps=config["training"]["logging_steps"],
        evaluation_strategy=config["training"]["evaluation_strategy"],
        save_strategy=config["training"]["save_strategy"],
        seed=config["training"]["seed"],
        load_best_model_at_end=True,
        report_to="none" if not os.environ.get("WANDB_API_KEY") else "wandb"
    )

    # Optional extras
    optional_keys = ["fp16", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "max_grad_norm"]
    for key in optional_keys:
        if key in config["training"]:
            setattr(training_args, key, config["training"][key])

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"🚀 Training model: {config['model']['name']}")
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config["training"]["output_dir"])

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    if "validation" in tokenized_datasets:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    print(f"✅ Training complete! Model saved to {config['training']['output_dir']}")

if __name__ == "__main__":
    main()
