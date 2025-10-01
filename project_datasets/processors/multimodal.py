from typing import Dict, Any
from .text import TextProcessor
from .image import ImageProcessor

class MultimodalProcessor:
    """Processes multimodal data (e.g., text and images)."""

    def __init__(self, text_processor: TextProcessor, image_processor: ImageProcessor):
        self.text_processor = text_processor
        self.image_processor = image_processor

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single multimodal data point.

        Args:
            data: A dictionary containing different modalities (e.g., 'text', 'image').

        Returns:
            A dictionary with processed data for each modality.
        """
        processed_data = {}
        if 'text' in data:
            processed_data['text'] = self.text_processor.process(data['text'])
        if 'image' in data:
            processed_data['image'] = self.image_processor.process(data['image'])
        
        return processed_data