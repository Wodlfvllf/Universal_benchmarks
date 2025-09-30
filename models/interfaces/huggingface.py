import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    GenerationConfig
)
from typing import List, Dict, Any, Optional, Union
import numpy as np
from .base import BaseModel, ModelConfig, ModelType, ModelOutput

class HuggingFaceModel(BaseModel):
    """Interface for HuggingFace Transformers models"""
    
    def setup(self):
        """Initialize HuggingFace model"""
        from transformers import AutoConfig
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Determine model class based on task
        config = AutoConfig.from_pretrained(self.config.model_name)
        
        if self.config.model_type == ModelType.TEXT:
            if hasattr(config, 'num_labels'):
                # Classification model
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=self._get_torch_dtype(),
                    device_map="auto" if self.config.device == "cuda" else None,
                    trust_remote_code=True
                )
            else:
                # Generation model
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir,
                    torch_dtype=self._get_torch_dtype(),
                    device_map="auto" if self.config.device == "cuda" else None,
                    trust_remote_code=True
                )
                
        # Move to device if needed
        if self.config.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            
        # Set eval mode
        self.model.eval()
        
    def _get_torch_dtype(self):
        """Get torch dtype from config"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "int8": torch.int8
        }
        return dtype_map.get(self.config.precision, torch.float32)
        
    def generate(self, 
                prompts: Union[str, List[str]], 
                max_new_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.95,
                do_sample: bool = True,
                **kwargs) -> Union[str, List[str]]:
        """Generate text completions"""
        
        # Handle single string input
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]
            
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        # Generate
        with torch.no_grad():
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
            
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],  # Remove prompt
            skip_special_tokens=True
        )
        
        return generated_texts[0] if single_input else generated_texts
        
    def classify(self,
                inputs: Union[str, List[str]],
                labels: Optional[List[str]] = None,
                return_all_scores: bool = False,
                **kwargs) -> ModelOutput:
        """Perform classification"""
        
        # Handle single input
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
            
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        if self.config.device == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}
            
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**encoded)
            
        logits = outputs.logits.cpu().numpy()
        probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        predictions = np.argmax(logits, axis=-1)
        
        # Map to labels if provided
        if labels:
            predictions = [labels[pred] for pred in predictions]
            
        return [ModelOutput(
            predictions=predictions[i],
            logits=logits[i],
            probabilities=probabilities[i]
        ) for i in range(len(predictions))]
        
    def embed(self,
             inputs: Union[str, List[str]],
             pooling: str = "mean",
             **kwargs) -> np.ndarray:
        """Generate embeddings"""
        
        # Handle single input
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
            
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        )
        
        if self.config.device == "cuda":
            encoded = {k: v.cuda() for k, v in encoded.items()}
            
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded, output_hidden_states=True)
            
        # Extract embeddings from last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Apply pooling
        if pooling == "mean":
            # Mean pooling over sequence length
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * attention_mask).sum(1) / attention_mask.sum(1)
        elif pooling == "cls":
            # Use CLS token
            embeddings = hidden_states[:, 0, :]
        elif pooling == "max":
            # Max pooling
            embeddings = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        embeddings = embeddings.cpu().numpy()
        
        return embeddings[0] if single_input else embeddings
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.model_name,
            "type": "huggingface",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.model.device),
            "precision": self.config.precision,
            "max_length": self.config.max_length
        }