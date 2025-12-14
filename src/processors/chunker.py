from typing import List, Any
from transformers import AutoTokenizer
from src.interfaces import DocumentProcessor

class SlidingWindowChunker(DocumentProcessor):
    def __init__(self, model_id: str="distilbert-base-uncased", window_size: int=512, overlap: int=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.window_size = window_size
        self.overlap = overlap
        self.stride = window_size - overlap

    def process_document(self, text: str) -> List[str]:
        # Tokenize the whole document
        tokens = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        
        chunks = []
        
        # Sliding Window Loop
        token_length = len(tokens)
        for i in range(0, token_length, self.stride):
            tokens_sub = tokens[i:i+self.window_size]
            chunk = self.tokenizer.decode(tokens_sub)
            chunks.append(chunk)

        if not chunks:
            return [text]
            
        return chunks