from typing import Dict, Type, Optional, Any, List
from .interfaces.base import BaseModel, ModelConfig, ModelType
from .interfaces.huggingface import HuggingFaceModel
from .interfaces.openai import OpenAIModel
from .interfaces.anthropic import AnthropicModel
from .interfaces.google import GoogleModel
from .adapters.multimodal_model import MultimodalModelAdapter

class ModelRegistry:
    """Central registry for model implementations"""
    
    MODEL_IMPLEMENTATIONS: Dict[str, Type[BaseModel]] = {
        'huggingface': HuggingFaceModel,
        'openai': OpenAIModel,
        'anthropic': AnthropicModel,
        'google': GoogleModel,
        'multimodal': MultimodalModelAdapter,
    }
    
    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        # Text Models
        'gpt-4': {
            'implementation': 'openai',
            'model_type': ModelType.TEXT,
            'max_length': 8192
        },
        'gpt-3.5-turbo': {
            'implementation': 'openai',
            'model_type': ModelType.TEXT,
            'max_length': 4096
        },
        'claude-3-opus': {
            'implementation': 'anthropic',
            'model_type': ModelType.TEXT,
            'max_length': 200000
        },
        'gemini-1.5-pro': {
            'implementation': 'google',
            'model_name': 'gemini-1.5-pro-latest',
            'model_type': ModelType.TEXT,
            'max_length': 1000000
        },
        'gemini-pro': {
            'implementation': 'google',
            'model_name': 'gemini-pro',
            'model_type': ModelType.TEXT,
            'max_length': 30720
        },
        'llama-3-70b': {
            'implementation': 'huggingface',
            'model_type': ModelType.TEXT,
            'model_name': 'meta-llama/Meta-Llama-3-70B',
            'max_length': 8192
        },
        'mistral-7b': {
            'implementation': 'huggingface',
            'model_type': ModelType.TEXT,
            'model_name': 'mistralai/Mistral-7B-v0.1',
            'max_length': 32768
        },
        
        # Vision-Language Models
        'gpt-4-vision': {
            'implementation': 'openai',
            'model_type': ModelType.MULTIMODAL,
            'max_length': 4096
        },
        'claude-3-vision': {
            'implementation': 'anthropic',
            'model_type': ModelType.MULTIMODAL,
            'max_length': 200000
        },
        'llava-v1.6': {
            'implementation': 'multimodal',
            'model_type': ModelType.MULTIMODAL,
            'model_name': 'liuhaotian/llava-v1.6-34b',
            'max_length': 4096
        },
        'blip-2': {
            'implementation': 'multimodal',
            'model_type': ModelType.MULTIMODAL,
            'model_name': 'Salesforce/blip2-opt-2.7b',
            'max_length': 512
        },
        
        # Code Models
        'codegen-16B': {
            'implementation': 'huggingface',
            'model_type': ModelType.CODE,
            'model_name': 'Salesforce/codegen-16B-multi',
            'max_length': 2048
        },
        'starcoder': {
            'implementation': 'huggingface',
            'model_type': ModelType.CODE,
            'model_name': 'bigcode/starcoder',
            'max_length': 8192
        }
    }
    
    @classmethod
    def get_model(cls, 
                  name: str,
                  config_override: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Get model instance by name"""
        
        if name not in cls.MODEL_CONFIGS:
            # Try to interpret as HuggingFace model
            config = ModelConfig(
                model_name=name,
                model_type=ModelType.TEXT
            )
            return HuggingFaceModel(config)
            
        # Get pre-configured model
        model_info = cls.MODEL_CONFIGS[name].copy()
        implementation = model_info.pop('implementation')
        
        # Create config
        config_dict = {
            'model_name': model_info.get('model_name', name),
            **model_info
        }
        
        # Apply overrides
        if config_override:
            config_dict.update(config_override)
            
        config = ModelConfig(**config_dict)
        
        # Get implementation class
        if implementation not in cls.MODEL_IMPLEMENTATIONS:
            raise ValueError(f"Unknown implementation: {implementation}")
            
        model_class = cls.MODEL_IMPLEMENTATIONS[implementation]
        return model_class(config)
        
    @classmethod
    def register_model(cls, name: str, config: Dict[str, Any]):
        """Register new model configuration"""
        cls.MODEL_CONFIGS[name] = config
        
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models"""
        return list(cls.MODEL_CONFIGS.keys())

    @classmethod
    def register_implementation(cls, name: str, model_class: Type[BaseModel]):
        """Register a new model implementation"""
        cls.MODEL_IMPLEMENTATIONS[name] = model_class