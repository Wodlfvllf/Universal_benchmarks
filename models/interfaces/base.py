from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

class ModelType(Enum):
    """Model type enumeration"""
    TEXT = "text"
    VISION = "vision"
    MULTIMODAL = "multimodal"
    CODE = "code"
    AUDIO = "audio"

@dataclass
class ModelConfig:
    """Base configuration for models"""
    model_name: str
    model_type: ModelType
    max_length: int = 512
    batch_size: int = 32
    device: str = "cuda"
    precision: str = "float32"
    cache_dir: Optional[str] = None
    use_cache: bool = True
    num_threads: int = 1
    
@dataclass
class ModelOutput:
    """Standard model output format"""
    predictions: Union[str, List[str], np.ndarray]
    logits: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    attention_weights: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.setup()
        
    @abstractmethod
    def setup(self):
        """Initialize model and components"""
        pass
        
    @abstractmethod
    def generate(self, 
                prompts: Union[str, List[str]], 
                **kwargs) -> Union[str, List[str]]:
        """Generate text completions"""
        pass
        
    @abstractmethod
    def classify(self,
                inputs: Union[str, List[str]],
                labels: Optional[List[str]] = None,
                **kwargs) -> ModelOutput:
        """Perform classification"""
        pass
        
    @abstractmethod
    def embed(self,
             inputs: Union[str, List[str]],
             **kwargs) -> np.ndarray:
        """Generate embeddings"""
        pass
        
    def batch_process(self,
                     inputs: List[Any],
                     process_fn: callable,
                     batch_size: Optional[int] = None) -> List[Any]:
        """Process inputs in batches"""
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_results = process_fn(batch)
            results.extend(batch_results)
            
        return results
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        raise NotImplementedError("Checkpoint saving not implemented")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        raise NotImplementedError("Checkpoint loading not implemented")