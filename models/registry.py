from typing import Dict, Type, List, Any, Optional
from .interfaces.base import BaseModel
from .interfaces.huggingface import HuggingFaceModel

class ModelRegistry:
    """Central registry for all model implementations."""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]):
        """Register a model implementation."""
        if name in cls._models:
            print(f"Warning: Model type '{name}' is already registered. Overwriting.")
        cls._models[name] = model_class

    @classmethod
    def get_model(cls, model_type: str, model_name: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Get a model instance by type and name."""
        if model_type not in cls._models:
            raise ValueError(f"Model type '{model_type}' not registered. Available types: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        return model_class(model_name=model_name, config=config)

    @classmethod
    def list_model_types(cls) -> List[str]:
        """List all registered model types."""
        return list(cls._models.keys())

def register_all_models():
    """Register all built-in model interfaces."""
    ModelRegistry.register('huggingface', HuggingFaceModel)
    # Add more model types like 'openai', 'anthropic' here

# Auto-register models when this module is imported
register_all_models()
