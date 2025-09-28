from typing import Dict, List, Any, Optional, Type

class BenchmarkRegistry:
    """Central registry for all benchmarks"""
    
    _benchmarks = {}
    
    @classmethod
    def register(cls, name: str, category: str, runner_class: type):
        cls._benchmarks[name] = {
            'category': category,
            'runner': runner_class
        }
        
    @classmethod
    def list_benchmarks(cls, category: Optional[str] = None) -> List[str]:
        if category:
            return [name for name, info in cls._benchmarks.items() 
                   if info['category'] == category]
        return list(cls._benchmarks.keys())
        
    @classmethod
    def get_benchmark(cls, name: str) -> type:
        if name not in cls._benchmarks:
            raise ValueError(f"Benchmark {name} not registered")
        return cls._benchmarks[name]['runner']