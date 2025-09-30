import openai
from typing import List, Dict, Any, Optional, Union
import os
import time
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import BaseModel, ModelConfig, ModelType, ModelOutput

class OpenAIModel(BaseModel):
    """Interface for OpenAI API models"""
    
    def setup(self):
        """Initialize OpenAI client"""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        openai.api_key = self.api_key
        
        # Model name mapping
        self.model_map = {
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
            "text-embedding-3-small": "text-embedding-3-small",
            "text-embedding-3-large": "text-embedding-3-large"
        }
        
        self.model_id = self.model_map.get(self.config.model_name, self.config.model_name)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _api_call(self, func, **kwargs):
        """Make API call with retry logic"""
        try:
            return func(**kwargs)
        except openai.RateLimitError:
            time.sleep(2)
            raise
        except Exception as e:
            print(f"API call failed: {e}")
            raise
            
    def generate(self, 
                prompts: Union[str, List[str]], 
                max_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.95,
                system_prompt: Optional[str] = None,
                **kwargs) -> Union[str, List[str]]:
        """Generate text completions using OpenAI API"""
        
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]
            
        responses = []
        
        for prompt in prompts:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._api_call(
                openai.ChatCompletion.create,
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            responses.append(response.choices[0].message.content)
            
        return responses[0] if single_input else responses
        
    def classify(self,
                inputs: Union[str, List[str]],
                labels: Optional[List[str]] = None,
                **kwargs) -> ModelOutput:
        """Perform classification using OpenAI"""
        
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
            
        if not labels:
            raise ValueError("Labels must be provided for OpenAI classification")
            
        # Create classification prompt
        label_str = ", ".join(labels)
        system_prompt = f"Classify the following text into one of these categories: {label_str}. Respond with only the category name."
        
        predictions = []
        
        for input_text in inputs:
            response = self.generate(
                input_text,
                system_prompt=system_prompt,
                temperature=0,  # Deterministic for classification
                max_tokens=50
            )
            
            # Find matching label
            response_lower = response.lower().strip()
            matched_label = None
            for label in labels:
                if label.lower() in response_lower:
                    matched_label = label
                    break
                    
            predictions.append(matched_label or labels[0])
            
        return ModelOutput(
            predictions=predictions[0] if single_input else predictions
        )
        
    def embed(self,
             inputs: Union[str, List[str]],
             **kwargs) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        
        single_input = isinstance(inputs, str)
        if single_input:
            inputs = [inputs]
            
        # Use embedding model
        response = self._api_call(
            openai.Embedding.create,
            model="text-embedding-3-small",
            input=inputs
        )
        
        embeddings = np.array([item['embedding'] for item in response['data']])
        
        return embeddings[0] if single_input else embeddings
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.model_name,
            "type": "openai",
            "model_id": self.model_id,
            "api_based": True,
            "max_length": self.config.max_length
        }