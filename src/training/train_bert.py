import modal
import sys
import os

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "datasets", "pandas", "scikit-learn", "accelerate")
)

app = modal.App("arxiv-finetune")

# Create a Volume to persist the trained model
vol = modal.Volume.from_name("arxiv-models", create_if_missing=True)

@app.function(
    image=image, 
    gpu="A10G", 
    timeout=3600,
    volumes={"/data": vol} # Mount the volume at /data
)
def train_distilbert():
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        TrainingArguments, 
        Trainer,
        DataCollatorWithPadding
    )
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    print("Starting Training on Modal GPU...")
    
    # Load Data
    dataset = load_dataset("ccdv/arxiv-classification", "no_ref", split="train")

    # Split
    dataset = dataset.train_test_split(test_size=0.3)
    
    # Tokenization
    model_id = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=512
        )

    tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=8)
    
    # Model Setup
    num_labels = 11
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, 
        num_labels=num_labels
    )
    
    # Metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }

    # Trainer Setup
    training_args = TrainingArguments(
        output_dir="/data/distilbert-arxiv", 
        learning_rate=2e-5,
        per_device_train_batch_size=16, 
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,        
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True, 
        optim="adamw_torch", 
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()
    
    # Final Save
    print("Saving model to Volume...")
    trainer.save_model("/data/distilbert-arxiv-final")
    vol.commit() 
    print("Model saved to /data/distilbert-arxiv-final")

if __name__ == "__main__":
    with app.run():
        train_distilbert.remote()