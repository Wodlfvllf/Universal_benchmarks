import pickle
from pathlib import Path

def load_from_cache(cache_path: Path):
    """Load data from a cache file."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None

def save_to_cache(data: any, cache_path: Path):
    """Save data to a cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
