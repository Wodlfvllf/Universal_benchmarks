from benchmarks.base_runner import BenchmarkRunner
from typing import Dict, Optional

class Vqav2Runner(BenchmarkRunner):
    def __init__(self, model_name: str, model_config: Optional[Dict] = None):
        config_path = "configs/benchmark_configs/vqav2.yaml"
        super().__init__(config_path, model_name, model_config)
