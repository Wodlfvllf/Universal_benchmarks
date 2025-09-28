import re
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer

class TextProcessor:
    """Text preprocessing utilities"""

    def __init__(self, lowercase: bool = False, 
                 remove_punctuation: bool = False,
                 normalize_whitespace: bool = True,
                 max_length: Optional[int] = None):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.normalize_whitespace = normalize_whitespace
        self.max_length = max_length

    def process(self, text: str) -> str:
        """Apply preprocessing to text"""

        if self.lowercase:
            text = text.lower()

        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        if self.normalize_whitespace:
            text = ' '.join(text.split())

        if self.max_length:
            text = text[:self.max_length]

        return text

    def batch_process(self, texts: List[str]) -> List[str]:
        """Process batch of texts"""
        return [self.process(text) for text in texts]

class TokenizationProcessor:
    """Tokenization processor for transformer models"""

    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize single text"""
        return self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

    def batch_tokenize(self, texts: List[str]) -> Dict[str, Any]:
        """Tokenize batch of texts"""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
