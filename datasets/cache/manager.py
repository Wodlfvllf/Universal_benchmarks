import shutil
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime, timedelta

class CacheManager:
    """Manage dataset caching"""
    
    def __init__(self, cache_dir: Optional[Path] = None,
                 max_size_gb: float = 100.0,
                 ttl_days: int = 30):
        self.cache_dir = cache_dir or Path.home() / '.cache' / 'universal_benchmarks'
        self.max_size_gb = max_size_gb
        self.ttl_days = ttl_days
        
    def get_cache_size(self) -> float:
        """Get current cache size in GB"""
        total_size = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total_size += path.stat().st_size
        return total_size / (1024 ** 3)
        
    def clean_expired(self):
        """Remove expired cache entries"""
        cutoff_date = datetime.now() - timedelta(days=self.ttl_days)
        
        for metadata_file in self.cache_dir.glob('*_metadata.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                
            timestamp = datetime.fromisoformat(metadata['timestamp'])
            if timestamp < cutoff_date:
                # Remove dataset and metadata
                dataset_path = metadata_file.parent / metadata_file.stem.replace('_metadata', '')
                if dataset_path.exists():
                    shutil.rmtree(dataset_path)
                metadata_file.unlink()
                
    def clean_by_size(self):
        """Remove old entries if cache exceeds max size"""
        current_size = self.get_cache_size()
        
        if current_size > self.max_size_gb:
            # Sort by modification time
            cache_items = []
            for path in self.cache_dir.rglob('*_metadata.json'):
                cache_items.append((path.stat().st_mtime, path))
                
            cache_items.sort()
            
            # Remove oldest until under limit
            for mtime, metadata_file in cache_items:
                if current_size <= self.max_size_gb * 0.9:  # Keep 10% buffer
                    break
                    
                # Remove dataset
                dataset_path = metadata_file.parent / metadata_file.stem.replace('_metadata', '')
                if dataset_path.exists():
                    size = sum(f.stat().st_size for f in dataset_path.rglob('*')) / (1024 ** 3)
                    shutil.rmtree(dataset_path)
                    current_size -= size
                    
                metadata_file.unlink()
                
    def clear_all(self):
        """Clear entire cache"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)