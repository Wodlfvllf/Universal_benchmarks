from .base import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class HuggingFaceModel(BaseModel):
    """A wrapper for Hugging Face Hub models."""

    def _load_model(self) -> Any:
        """Loads the model and tokenizer from Hugging Face Hub."""
        model_kwargs = self.config.get('model_kwargs', {})
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        return model

    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        """
        Generates text using the loaded Hugging Face model.
        """
        # Default generation parameters
        default_kwargs = {
            'max_length': self.config.get('max_length', 512),
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.95),
            'pad_token_id': self.tokenizer.pad_token_id
        }
        # Override defaults with any provided kwargs
        default_kwargs.update(kwargs)

        # Tokenize the inputs
        tokenized_inputs = self.tokenizer(inputs, return_tensors='pt', padding=True).to(self.device)

        # Generate outputs
        output_ids = self.model.generate(**tokenized_inputs, **default_kwargs)

        # Decode the outputs
        decoded_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # Clean up the output to remove the prompt
        cleaned_outputs = []
        for i, output in enumerate(decoded_outputs):
            # The generated text includes the prompt, so we remove it.
            prompt_text = inputs[i]
            if output.startswith(prompt_text):
                cleaned_outputs.append(output[len(prompt_text):].strip())
            else:
                cleaned_outputs.append(output.strip())
        
        return cleaned_outputs

    def classify(self, inputs: List[str], candidate_labels: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Performs zero-shot classification using a pipeline.
        """
        classifier = pipeline("zero-shot-classification", model=self.model, tokenizer=self.tokenizer, device=self.device)
        return classifier(inputs, candidate_labels, **kwargs)
