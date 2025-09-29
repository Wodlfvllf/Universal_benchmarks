
import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseModel, ModelConfig, ModelOutput, ModelType

class GoogleModel(BaseModel):
    """Interface for Google's Gemini models"""

    def setup(self):
        """Initialize Google's API client"""
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(self.config.model_name)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _api_call(self, func, **kwargs):
        """Make API call with retry logic"""
        try:
            return func(**kwargs)
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
        """Generate text completions using the Gemini API"""

        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]

        responses = []

        for prompt in prompts:
            # Note: The Gemini API uses a different format for system prompts.
            # It's passed during model initialization or in specific contents.
            # For simplicity, we'll prepend it to the user prompt if provided.
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            response = self._api_call(
                self.model.generate_content,
                contents=full_prompt,
                generation_config=generation_config,
                **kwargs
            )

            responses.append(response.text)

        return responses[0] if single_input else responses

    def classify(self, 
                 inputs: Union[str, List[str]], 
                 labels: Optional[List[str]] = None, 
                 **kwargs) -> ModelOutput:
        """Perform classification using Gemini"""
        raise NotImplementedError("Classification is not yet implemented for Google models.")

    def embed(self, 
              inputs: Union[str, List[str]], 
              **kwargs) -> np.ndarray:
        """Generate embeddings using Gemini API"""
        raise NotImplementedError("Embedding is not yet implemented for Google models.")

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.config.model_name,
            "type": "google",
            "api_based": True,
            "max_length": self.config.max_length
        }
