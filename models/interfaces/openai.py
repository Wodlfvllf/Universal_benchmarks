from .base import BaseModel
from typing import List

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Add actual OpenAI API call here
        return ["OpenAI model response" for _ in prompts]
