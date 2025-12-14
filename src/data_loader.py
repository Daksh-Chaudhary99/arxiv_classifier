from datasets import load_dataset
import pandas as pd

def load_data():
    print("Loading the dataset")
    dataset = load_dataset("ccdv/arxiv-classification", "no_ref", split="train")
    
    label_names = dataset.features['label'].names
    print(f"Found {len(label_names)} classes: {label_names}")

    df = dataset.to_pandas()
    df['label_name'] = [label_names[i] for i in df['label']]

    print(f"Dataset size: {len(df)}")
    print(df[['text', 'label_name']].head(2))

    return dataset, label_names

if __name__ == "__main__":
    load_data()