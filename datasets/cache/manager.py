import shutil
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime, timedelta

class CacheManager:
    """Manage dataset caching."""

    def __init__(self, cache_dir: Optional[Path] = None,
                 max_size_gb: float = 50.0,
                 ttl_days: int = 30):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'universal_benchmarks' / 'datasets'
        self.max_size_gb = max_size_gb
        self.ttl_days = ttl_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_size(self) -> float:
        """Get current cache size in GB."""
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
        return total_size / (1024 ** 3)

    def clean_expired(self):
        """Remove expired cache entries."""
        if self.ttl_days <= 0:
            return
        
        cutoff_date = datetime.now() - timedelta(days=self.ttl_days)

        for metadata_file in self.cache_dir.glob('*_metadata.json'):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                timestamp = datetime.fromisoformat(metadata['timestamp'])
                if timestamp < cutoff_date:
                    # Remove dataset directory and metadata file
                    dataset_key = metadata_file.stem.replace('_metadata', '')
                    dataset_path = self.cache_dir / dataset_key
                    if dataset_path.exists() and dataset_path.is_dir():
                        shutil.rmtree(dataset_path)
                    metadata_file.unlink()
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"Error processing metadata file {metadata_file}: {e}")

    def clean_by_size(self):
        """Remove oldest entries if cache exceeds max size."""
        while self.get_cache_size() > self.max_size_gb:
            # Find the oldest item by modification time of its metadata
            try:
                oldest_meta = min(
                    self.cache_dir.glob('*_metadata.json'), 
                    key=lambda p: p.stat().st_mtime
                )
                
                dataset_key = oldest_meta.stem.replace('_metadata', '')
                dataset_path = self.cache_dir / dataset_key

                if dataset_path.exists() and dataset_path.is_dir():
                    shutil.rmtree(dataset_path)
                oldest_meta.unlink()

            except ValueError:
                # No metadata files found, break loop
                break

    def clear_all(self):
        """Clear the entire cache directory."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache cleared at {self.cache_dir}")
