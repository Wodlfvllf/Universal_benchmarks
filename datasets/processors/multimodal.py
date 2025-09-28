from typing import Dict, Any
import torch

from .text import TokenizationProcessor
from .image import ImageProcessor

class MultimodalProcessor:
    """Processes multimodal inputs (e.g., text and image)."""

    def __init__(self, tokenizer_name: str, image_processor_config: Dict = None):
        """
        Initializes the multimodal processor.

        Args:
            tokenizer_name: The name of the Hugging Face tokenizer to use.
            image_processor_config: Configuration for the ImageProcessor.
        """
        self.tokenizer = TokenizationProcessor(model_name=tokenizer_name)
        self.image_processor = ImageProcessor(**(image_processor_config or {}))

    def process(self, text_input: str = None, image_input: Any = None) -> Dict[str, Any]:
        """
        Processes a combination of text and image inputs.

        Args:
            text_input: The text data to process.
            image_input: The image data to process (path or PIL Image).

        Returns:
            A dictionary containing the processed text and image tensors.
        """
        processed_data = {}

        if text_input is not None:
            processed_data['text'] = self.tokenizer.tokenize(text_input)

        if image_input is not None:
            processed_data['image'] = self.image_processor.process(image_input)

        return processed_data

    def batch_process(self, text_inputs: List[str] = None, image_inputs: List[Any] = None) -> Dict[str, Any]:
        """
        Processes a batch of multimodal inputs.
        """
        batch_processed_data = {}

        if text_inputs:
            batch_processed_data['text'] = self.tokenizer.batch_tokenize(text_inputs)
        
        if image_inputs:
            batch_processed_data['image'] = self.image_processor.batch_process(image_inputs)

        return batch_processed_data
