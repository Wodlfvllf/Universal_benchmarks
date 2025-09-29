
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..models.registry import ModelRegistry
from ..datasets.registry import DatasetRegistry
from ..tasks.registry import TaskRegistry

class BaseBenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    def __init__(self, benchmark_name: str, model_name: str, model_type: str, model_config: Optional[Dict[str, Any]] = None):
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.model_type = model_type
        self.model_config = model_config or {}
        self.model = self._load_model()

    def _load_model(self):
        """Loads the model using the model registry."""
        return ModelRegistry.get_model(name=self.model_name, config_override=self.model_config)

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Runs the benchmark and returns the results."""
        pass
