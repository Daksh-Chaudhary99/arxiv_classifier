from src.interfaces import BaseClassifier
from unsloth import FastLanguageModel
import torch

class LlamaLoraClassifier(BaseClassifier):
    def __init__(self, model_path: str):
        print(f"Loading Llama-3 from {model_path}...")
        
        #Load Base Model + Adapters
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        FastLanguageModel.for_inference(self.model)
        
        #Prompt Template
        self.alpaca_prompt = """### Instruction:
        Classify the following ArXiv paper into one of the known categories (e.g., cs.AI, cs.CV, math.ST).
        Return ONLY the class label.

        ### Input:
        {}

        ### Response:
        """

    def predict(self, text: str) -> str:

        formatted_prompt = self.alpaca_prompt.format(text[:3000])

        inputs = self.tokenizer(
            [formatted_prompt], 
            return_tensors="pt"
        ).to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        #Skipping the input prompt and only looking at what the model added
        decoded_output = self.tokenizer.batch_decode(outputs)
        raw_response = decoded_output[0]
        
        #Extracting the part after "### Response:"
        response_start = raw_response.find("### Response:\n")
        if response_start != -1:
            clean_response = raw_response[response_start + len("### Response:\n"):].strip()
            clean_response = clean_response.replace("<|end_of_text|>", "").strip()
            return clean_response
        
        return "Unknown"