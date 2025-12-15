import argparse
import modal

def run_cloud_inference(model_type: str):
    if model_type == "distilbert":
        print(f"🚀 Launching DistilBERT Inference on Cloud...")
        print("Run this command: modal run src/inference/inference_distilbert.py")
        
    elif model_type == "llama":
        print(f"🦙 Launching Llama-3 Inference on Cloud...")
        print("Run this command: modal run src/inference/inference_lora.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["distilbert", "llama"])
    args = parser.parse_args()
    
    run_cloud_inference(args.model)