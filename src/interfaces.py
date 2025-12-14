from abc import ABC, abstractmethod
from typing import Any, Dict

class DocumentProcessor(ABC):
    @abstractmethod
    def process_document(self, document:str) -> Any:
        """
        Processes a raw document into model-ready input.
        Returns:
            str: For summarization or zero-shot approaches.
            List[str]: For sliding window chunking approaches.
        """
        pass


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, inputs:Any)->str:
        """
        Predicts the class label for the given input.
        Args:
            inputs: Can be a single string (document or summary) or a list of strings (document chunks).
        """
        pass