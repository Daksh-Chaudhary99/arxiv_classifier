from typing import Optional
from src.interfaces import BaseClassifier

class ModelFactory:
    @staticmethod
    def get_model(model_type: str, model_path: str) -> BaseClassifier:
        """
        Factory method to instantiate a classifier based on model_type.
        Lazy imports are used to avoid loading unnecessary heavy libraries.
        """
        if model_type == "distilbert":
            from src.classifiers.distilbert import DistilBERTClassifier
            return DistilBERTClassifier(model_path=model_path)
            
        elif model_type == "llama_lora":
            from src.classifiers.llm_lora import LlamaLoraClassifier
            return LlamaLoraClassifier(model_path=model_path)
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Choices: ['distilbert', 'llama_lora']")