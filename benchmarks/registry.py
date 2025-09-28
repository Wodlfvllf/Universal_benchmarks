from typing import Dict, Type, List, Optional
from .base_runner import BaseBenchmarkRunner

# A default runner that can handle any benchmark with a standard config
class DefaultBenchmarkRunner(BaseBenchmarkRunner):
    def __init__(self, benchmark_name: str, model_type: str, model_name: str, model_config: Optional[Dict] = None):
        super().__init__(benchmark_name, model_type, model_name, model_config)

class BenchmarkRegistry:
    """Central registry for all benchmarks."""

    _benchmarks: Dict[str, Type[BaseBenchmarkRunner]] = {}

    @classmethod
    def register(cls, name: str, runner_class: Type[BaseBenchmarkRunner]):
        """Register a benchmark runner."""
        cls._benchmarks[name] = runner_class

    @classmethod
    def get_benchmark_runner(cls, name: str) -> Type[BaseBenchmarkRunner]:
        """Get a benchmark runner class by name."""
        # Return the default runner if a specific one isn't found
        if name not in cls._benchmarks:
            print(f"Warning: No specific runner for '{name}'. Using DefaultBenchmarkRunner.")
            return DefaultBenchmarkRunner
        return cls._benchmarks[name]

    @classmethod
    def list_benchmarks(cls) -> List[str]:
        """List all registered benchmarks."""
        # This could be improved by scanning the benchmarks directory
        return list(cls._benchmarks.keys())

# You could auto-register specific benchmark runners here if needed
# For example:
# from .llm.glue.runner import GlueRunner
# BenchmarkRegistry.register('glue', GlueRunner)
