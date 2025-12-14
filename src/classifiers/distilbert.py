import torch
from typing import List
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.interfaces import BaseClassifier


class ChunkDataset(Dataset):
    """
    Wraps chunks so DataLoader can batch them.
    """
    def __init__(self, chunks: List[str]):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"text": self.chunks[idx]}


class DistilBERTClassifier(BaseClassifier):
    def __init__(
        self, 
        model_id: str = "distilbert-base-uncased", 
        num_labels: int = 11,
        batch_size: int = 16,
        device: str = None
    ):
        """
        Args:
            model_id: Can be a HF Hub ID OR a Volume path (e.g., "/data/distilbert-arxiv-final")
        """
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing DistilBERT from '{model_id}' on {self.device}...")
        
        # Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id, 
            num_labels=num_labels
        ).to(self.device)
        self.model.eval() 

    def predict(self, inputs: List[str]) -> str:
        """
        Args:
            inputs: A list of text chunks (strings) derived from one document.
        Returns:
            str: The predicted class ID (e.g., "3") based on Majority Voting.
        """
        if not inputs:
            return "Unknown"

        # Setup Data Loading
        dataset = ChunkDataset(inputs)
        
        def tokenize_batch(batch):
            texts = [item["text"] for item in batch]
            return self.tokenizer(
                texts, 
                truncation=True, 
                max_length=512, 
                padding=True,
                return_tensors="pt"
            )

        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=tokenize_batch,
            shuffle=False
        )

        all_preds = []

        # Inference Loop
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
                all_preds.extend(preds)

        # Aggregation (Majority Voting)
        if not all_preds:
            return "Unknown"
            
        vote_counts = Counter(all_preds)
        winner_id = vote_counts.most_common(1)[0][0]
        
        return str(winner_id)