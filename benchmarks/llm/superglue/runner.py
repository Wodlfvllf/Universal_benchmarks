from benchmarks.base_runner import BenchmarkRunner
from benchmarks.registry import BenchmarkRegistry

class SuperGlueRunner(BenchmarkRunner):
    def __init__(self, model_name: str, model_config: dict = None):
        super().__init__(
            config_path="configs/benchmark_configs/superglue.yaml",
            model_name=model_name,
            model_config=model_config
        )

BenchmarkRegistry.register(
    name="superglue",
    category="llm",
    runner_class=SuperGlueRunner
)
