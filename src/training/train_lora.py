import modal

# Environment - Using a specific CUDA 12.1 image because Unsloth requires it.
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")  
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "torch", 
        "torchvision",
        "transformers", 
        "datasets", 
        "trl", 
        "peft", 
        "accelerate", 
        "bitsandbytes"
    )
)

app = modal.App("arxiv-lora-finetune")
vol = modal.Volume.from_name("arxiv-models")

@app.function(
    image=image,
    gpu="A10G",  
    timeout=5400,
    volumes={"/data": vol}
)
def train_llama():
    # Imports inside the container
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments
    import torch

    print("--- Starting Llama-3 LoRA Training ---")
    
    #Configuration
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    #Load Base Model
    print("Loading 4-bit Llama-3 model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    #LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", #Attention
                      "gate_proj", "up_proj", "down_proj",], #FFN (Reasoning)
        lora_alpha=16,
        lora_dropout=0, 
        bias="none", 
        use_gradient_checkpointing="unsloth",
    )

    # Prompt Engineering
    
    #Alpaca Prompt Template. 
    alpaca_prompt = """### Instruction:
    Classify the following ArXiv paper into one of the known categories (e.g., cs.AI, cs.CV, math.ST).
    Return ONLY the class label.

    ### Input:
    {}

    ### Response:
    {}"""

    # Load the dataset
    dataset = load_dataset("ccdv/arxiv-classification", "no_ref", split="train")
    
    # EXTRACTING LABEL NAMES
    label_names = dataset.features["label"].names
    
    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompts_func(examples):
        inputs = examples["text"]
        labels = examples["label"]
        texts = []
        
        for input_text, label_idx in zip(inputs, labels):
            #Converting Integer Label -> String Label (e.g., 2 -> "cs.AI")
            label_str = label_names[label_idx]

            text = alpaca_prompt.format(input_text[:3000], label_str) + EOS_TOKEN
            
            texts.append(text)
            
        return {"text": texts}

    #Applying the formatter to the whole dataset
    dataset = dataset.map(formatting_prompts_func, batched=True)

    #Training Configuration
    print("Starting Training...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,  
            gradient_accumulation_steps=4,  
            warmup_steps=5,
            max_steps=60, 
            learning_rate=2e-4, 
            fp16=not torch.cuda.is_bf16_supported(), 
            bf16=torch.cuda.is_bf16_supported(), 
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="/tmp",
        ),
    )

    trainer_stats = trainer.train()

    print("Saving LoRA adapters to Volume...")
    # Saving only the ADAPTERS, not the full model
    model.save_pretrained("/data/llama-lora-arxiv")
    tokenizer.save_pretrained("/data/llama-lora-arxiv")

    vol.commit()
    print("SUCCESS: Model saved to /data/llama-lora-arxiv")