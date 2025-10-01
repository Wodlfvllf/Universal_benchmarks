from benchmarks.base_runner import BenchmarkRunner
from benchmarks.registry import BenchmarkRegistry

class GlueRunner(BenchmarkRunner):
    def __init__(self, model_name: str, model_config: dict = None):
        super().__init__(
            config_path="configs/benchmark_configs/glue.yaml",
            model_name=model_name,
            model_config=model_config
        )

print("Registering glue benchmark")
BenchmarkRegistry.register(
    name="glue",
    category="llm",
    runner_class=GlueRunner
)
