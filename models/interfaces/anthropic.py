from .base import BaseModel
from typing import List

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def generate(self, prompts: List[str], **kwargs) -> List[str]:
        # Add actual Anthropic API call here
        return ["Anthropic model response" for _ in prompts]
