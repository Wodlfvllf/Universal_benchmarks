from benchmarks.base_runner import BenchmarkRunner
from typing import Dict, Optional

class HellaswagRunner(BenchmarkRunner):
    def __init__(self, model_name: str, model_config: Optional[Dict] = None):
        config_path = "configs/benchmark_configs/hellaswag.yaml"
        super().__init__(config_path, model_name, model_config)