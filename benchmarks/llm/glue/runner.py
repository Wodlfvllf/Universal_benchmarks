from benchmarks.base_runner import BenchmarkRunner

class GlueBenchmarkRunner(BenchmarkRunner):
    def __init__(self, model_name: str, model_config: dict = None):
        super().__init__(
            config_path="/root/benchmarks/universal-model-benchmarks/benchmarks/llm/glue/config.yaml",
            model_name=model_name,
            model_config=model_config
        )