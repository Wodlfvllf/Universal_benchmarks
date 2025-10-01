from typing import Dict, Any, Optional, Type, List
from .loaders.base import DatasetLoader
from .loaders.huggingface import HuggingFaceLoader
from .loaders.local import LocalFileLoader

class DatasetRegistry:
    """Central registry for datasets and loaders"""
    
    # Pre-configured dataset mappings
    DATASET_CONFIGS = {
        # LLM Benchmarks
        'glue': {
            'loader': 'huggingface',
            'identifier': 'nyu-mll/glue',
            'subtasks': ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
        },
        'superglue': {
            'loader': 'huggingface',
            'identifier': 'super_glue',
            'subtasks': ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
        },
        'mmlu': {
            'loader': 'huggingface',
            'identifier': 'cais/mmlu',
            'subtasks': None  # All 57 subjects
        },
        'hellaswag': {
            'loader': 'huggingface',
            'identifier': 'Rowan/hellaswag'
        },
        'arc': {
            'loader': 'huggingface',
            'identifier': 'allenai/ai2_arc'
        },
        'gsm8k': {
            'loader': 'huggingface',
            'identifier': 'gsm8k'
        },
        'humaneval': {
            'loader': 'huggingface',
            'identifier': 'openai_humaneval'
        },
        
        # Vision-Language Benchmarks
        'vqav2': {
            'loader': 'huggingface',
            'identifier': 'HuggingFaceM4/VQAv2'
        },
        'coco_captions': {
            'loader': 'huggingface',
            'identifier': 'HuggingFaceM4/COCO'
        },
        
        # Add more dataset configs...
    }
    
    # Loader classes
    LOADERS: Dict[str, Type[DatasetLoader]] = {
        'huggingface': HuggingFaceLoader,
        'local': LocalFileLoader,
    }
    
    @classmethod
    def get_dataset(cls, dataset_name: str, **kwargs) -> Any:
        """Load dataset by name"""
        if dataset_name not in cls.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        config = cls.DATASET_CONFIGS[dataset_name].copy()
        loader_name = config.pop('loader')
        
        if loader_name not in cls.LOADERS:
            raise ValueError(f"Unknown loader: {loader_name}")
            
        loader_class = cls.LOADERS[loader_name]
        loader = loader_class()
        
        # Merge config with kwargs
        load_kwargs = {**config, **kwargs}
        
        return loader.load(**load_kwargs)
        
    @classmethod
    def register_dataset(cls, name: str, config: Dict[str, Any]):
        """Register new dataset configuration"""
        cls.DATASET_CONFIGS[name] = config
        
    @classmethod
    def register_loader(cls, name: str, loader_class: Type[DatasetLoader]):
        """Register new loader class"""
        cls.LOADERS[name] = loader_class
        
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets"""
        return list(cls.DATASET_CONFIGS.keys())