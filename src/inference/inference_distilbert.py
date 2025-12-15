import modal
import sys
import os

# 1. Define the Volume
vol = modal.Volume.from_name("arxiv-models")

# 2. Define the Cloud Environment
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "datasets", "pandas", "scikit-learn")
    .add_local_python_source("src") 
)

app = modal.App("arxiv-classifier-inference")

@app.function(
    image=image, 
    gpu="A10G", 
    timeout=600,
    volumes={"/data": vol}
)
def run_evaluation_on_cloud():
    from src.data_loader import load_data
    from src.processors.chunker import SlidingWindowChunker
    from src.classifiers.distilbert import DistilBERTClassifier
    from sklearn.metrics import classification_report

    print("Running Inference on Modal GPU...")
    
    # 1. Load Data (Testing on 50 random papers)
    dataset, label_names = load_data()
    test_data = dataset.select(range(50)) 
    
    # 2. Configure the Pipeline
    saved_model_path = "/data/distilbert-arxiv-final"
    
    print(f"Loading Fine-Tuned Model from: {saved_model_path}")
    
    # The Chunker: Slices 20k-token papers into 512-token windows
    processor = SlidingWindowChunker(model_id="distilbert-base-uncased")
    
    # The Classifier: Loads weights from Volume, predicts, and votes
    classifier = DistilBERTClassifier(
        model_id=saved_model_path, 
        num_labels=len(label_names)
    )
    
    # 3. Execution Loop
    y_true = []
    y_pred = []
    
    print(f"Evaluating on {len(test_data)} documents...")
    
    for i, row in enumerate(test_data):
        text = row['text']
        true_label_id = row['label']
        true_label_name = label_names[true_label_id]
        
        # Step 1: Chunking
        chunks = processor.process_document(text)
        
        # Step 2: Prediction & Voting
        pred_label_id = classifier.predict(chunks)
        
        # Convert prediction ID back to name (e.g., "3" -> "cs.AI")
        if pred_label_id.isdigit():
            pred_label_name = label_names[int(pred_label_id)]
        else:
            pred_label_name = "Unknown"
            
        y_true.append(true_label_name)
        y_pred.append(pred_label_name)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(test_data)}...")
        
    # 4. Final Report
    print("\n--- Final Results ---")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    with app.run():
        run_evaluation_on_cloud.remote()