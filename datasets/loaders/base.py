from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import hashlib
import json
from datetime import datetime

class DatasetLoader(ABC):
    """Abstract base class for dataset loaders"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'universal_benchmarks'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(self, identifier: str, **kwargs) -> Any:
        """Load dataset from source"""
        pass

    @abstractmethod
    def verify(self, dataset: Any) -> bool:
        """Verify dataset integrity"""
        pass

    def get_cache_key(self, identifier: str, **kwargs) -> str:
        """Generate cache key for dataset"""
        key_parts = [identifier] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def save_metadata(self, dataset: Any, identifier: str):
        """Save dataset metadata"""
        metadata = {
            'identifier': identifier,
            'loader': self.__class__.__name__,
            'timestamp': str(datetime.now()),
            'size': self.get_dataset_size(dataset)
        }

        metadata_path = self.cache_dir / f"{identifier}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    @abstractmethod
    def get_dataset_size(self, dataset: Any) -> Dict[str, int]:
        """Get dataset size information"""
        pass
