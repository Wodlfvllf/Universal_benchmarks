from dataclasses import dataclass
from pathlib import Path

@dataclass
class CacheConfig:
    """Configuration for dataset caching."""
    cache_dir: Path = Path.home() / '.cache' / 'universal_benchmarks'
    max_size_gb: float = 100.0
    ttl_days: int = 30