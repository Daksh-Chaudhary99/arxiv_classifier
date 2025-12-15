import modal

# 1. Define the Environment
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
        "bitsandbytes",
        "scikit-learn", 
        "pandas"
    )
    .add_local_python_source("src") 
)

app = modal.App("arxiv-lora-inference")
vol = modal.Volume.from_name("arxiv-models")

@app.function(
    image=image,
    gpu="A10G",
    volumes={"/data": vol},
    timeout=600
)
def run_llama_eval():
    from src.data_loader import load_data
    from src.classifiers.llm_lora import LlamaLoraClassifier
    from sklearn.metrics import classification_report
    
    print("--- Starting Llama-3 Inference ---")
    
    # Load Data
    dataset, label_names = load_data()
    test_data = dataset.select(range(20)) 
    
    # Initialize Model
    classifier = LlamaLoraClassifier(model_path="/data/llama-lora-arxiv")
    
    y_true = []
    y_pred = []
    
    print(f"Eval on {len(test_data)} docs...")
    
    for i, row in enumerate(test_data):
        text = row['text']
        true_label_id = row['label']
        true_label_name = label_names[true_label_id]
        
        # Predict
        pred_label = classifier.predict(text)
        
        print(f"Doc {i}: True='{true_label_name}' | Pred='{pred_label}'")
        
        y_true.append(true_label_name)
        y_pred.append(pred_label)
        
    print("\n--- Final Llama-3 Results ---")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    with app.run():
        run_llama_eval.remote()