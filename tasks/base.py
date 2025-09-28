from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class TaskInput:
    """Standard input format for tasks"""
    data: Union[str, List[str], np.ndarray, Dict]
    metadata: Optional[Dict[str, Any]] = None
    labels: Optional[Any] = None

@dataclass
class TaskOutput:
    """Standard output format for tasks"""
    predictions: Any
    logits: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseTask(ABC):
    """Abstract base class for all tasks"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.setup()

    @abstractmethod
    def setup(self):
        """Initialize task-specific components"""
        pass

    @abstractmethod
    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """Convert dataset to task inputs"""
        pass

    @abstractmethod
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format input for model consumption"""
        pass

    @abstractmethod
    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """Generate predictions using model"""
        pass

    @abstractmethod
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output into structured format"""
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute task-specific metrics"""
        pass
