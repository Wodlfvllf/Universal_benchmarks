
from typing import Dict, List, Any, Optional, Type
from .base_runner import BaseBenchmarkRunner
from .llm.glue.runner import GlueBenchmarkRunner

class BenchmarkRegistry:
    """Central registry for all benchmarks"""
    
    _benchmarks = {}
    
    @classmethod
    def register(cls, name: str, category: str, runner_class: Type[BaseBenchmarkRunner]):
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
    def get_benchmark_runner(cls, name: str) -> Type[BaseBenchmarkRunner]:
        if name not in cls._benchmarks:
            raise ValueError(f"Benchmark {name} not registered")
        return cls._benchmarks[name]['runner']

def register_all_benchmarks():
    """Register all built-in benchmarks."""
    BenchmarkRegistry.register('llm/glue', 'llm', GlueBenchmarkRunner)

# Auto-register benchmarks when this module is imported
register_all_benchmarks()
