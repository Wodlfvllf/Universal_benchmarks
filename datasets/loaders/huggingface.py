from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from typing import Dict, Optional, Union, List, Any
import os
from pathlib import Path
from .base import DatasetLoader

class HuggingFaceLoader(DatasetLoader):
    """Loader for HuggingFace datasets"""

    def __init__(self, cache_dir: Optional[Path] = None, use_auth_token: Optional[str] = None):
        super().__init__(cache_dir)
        self.use_auth_token = use_auth_token or os.getenv('HF_AUTH_TOKEN')

    def load(self, 
             identifier: str, 
             name: Optional[str] = None,
             split: Optional[Union[str, List[str]]] = None,
             streaming: bool = False,
             **kwargs) -> Union[Dataset, DatasetDict]:
        """
        Load HuggingFace dataset
        
        Args:
            identifier: Dataset identifier (e.g., 'glue')
            name: Dataset configuration name (e.g., 'cola')
            split: Split(s) to load ('train', 'validation', 'test')
            streaming: Whether to use streaming mode
            **kwargs: Additional arguments for load_dataset
        """

        # Check cache first
        cache_key = self.get_cache_key(identifier, name=name, split=split)
        cache_path = self.cache_dir / 'datasets' / cache_key

        if cache_path.exists() and not streaming:
            try:
                dataset = load_from_disk(str(cache_path))
                print(f"Loaded dataset from cache: {cache_path}")
                return dataset
            except Exception as e:
                print(f"Cache load failed: {e}, downloading fresh")

        # Load from HuggingFace
        dataset = load_dataset(
            identifier,
            name=name,
            split=split,
            cache_dir=str(self.cache_dir / 'downloads'),
            use_auth_token=self.use_auth_token,
            streaming=streaming,
            **kwargs
        )

        # Save to cache if not streaming
        if not streaming:
            dataset.save_to_disk(str(cache_path))
            self.save_metadata(dataset, identifier)

        return dataset

    def verify(self, dataset: Any) -> bool:
        """Verify HuggingFace dataset"""
        if isinstance(dataset, DatasetDict):
            for split_name, split_dataset in dataset.items():
                if len(split_dataset) == 0:
                    return False
        elif hasattr(dataset, '__len__'):
            return len(dataset) > 0
        return True

    def get_dataset_size(self, dataset: Any) -> Dict[str, int]:
        """Get dataset size information"""
        if isinstance(dataset, DatasetDict):
            return {split: len(data) for split, data in dataset.items()}
        else:
            return {'size': len(dataset)}

    def list_available_datasets(self) -> List[str]:
        """List available datasets from HuggingFace"""
        from huggingface_hub import list_datasets
        datasets = list_datasets()
        return [d.id for d in datasets]
