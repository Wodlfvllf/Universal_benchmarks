from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the model.

        Args:
            model_name: The name or path of the model to load.
            config: A dictionary of configuration options for the model.
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Loads the actual model object."""
        pass

    @abstractmethod
    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        """
        Generates text based on a list of input prompts.

        Args:
            inputs: A list of prompts to generate from.
            **kwargs: Additional generation parameters (e.g., max_length, temperature).

        Returns:
            A list of generated text strings.
        """
        pass

    def __call__(self, inputs: List[str], **kwargs) -> List[str]:
        """Allows the model to be called directly."""
        return self.generate(inputs, **kwargs)
