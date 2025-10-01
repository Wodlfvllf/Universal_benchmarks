import pandas as pd
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from .base import DatasetLoader

class LocalFileLoader(DatasetLoader):
    """Loader for local files (CSV, JSON, TXT, etc.)"""

    SUPPORTED_FORMATS = {
        '.csv': 'csv',
        '.tsv': 'tsv', 
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.txt': 'text',
        '.parquet': 'parquet'
    }

    def load(self, 
             identifier: str,
             format: Optional[str] = None,
             **kwargs) -> Any:
        """
        Load dataset from local file
        
        Args:
            identifier: File path
            format: File format (auto-detected if None)
            **kwargs: Format-specific arguments
        """

        path = Path(identifier)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {identifier}")

        # Auto-detect format
        if format is None:
            suffix = path.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported file format: {suffix}")
            format = self.SUPPORTED_FORMATS[suffix]

        # Load based on format
        if format == 'csv':
            return self._load_csv(path, **kwargs)
        elif format == 'tsv':
            return self._load_csv(path, delimiter='\t', **kwargs)
        elif format == 'json':
            return self._load_json(path, **kwargs)
        elif format == 'jsonl':
            return self._load_jsonl(path, **kwargs)
        elif format == 'text':
            return self._load_text(path, **kwargs)
        elif format == 'parquet':
            return self._load_parquet(path, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        return pd.read_csv(path, **kwargs)

    def _load_json(self, path: Path, **kwargs) -> Any:
        """Load JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_jsonl(self, path: Path, **kwargs) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_text(self, path: Path, **kwargs) -> List[str]:
        """Load text file"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file"""
        return pd.read_parquet(path, **kwargs)

    def verify(self, dataset: Any) -> bool:
        """Verify local dataset"""
        if isinstance(dataset, pd.DataFrame):
            return not dataset.empty
        elif isinstance(dataset, list):
            return len(dataset) > 0
        return dataset is not None

    def get_dataset_size(self, dataset: Any) -> Dict[str, int]:
        """Get dataset size"""
        if isinstance(dataset, pd.DataFrame):
            return {'rows': len(dataset), 'columns': len(dataset.columns)}
        elif isinstance(dataset, list):
            return {'size': len(dataset)}
        return {'size': 1}

