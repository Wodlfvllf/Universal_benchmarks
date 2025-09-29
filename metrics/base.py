
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class MetricResult:
    """Standard metric result format"""
    score: float
    confidence_interval: Optional[Tuple[float, float]] = None
    per_sample_scores: Optional[List[float]] = None
    additional_metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

class BaseMetric(ABC):
    """Abstract base class for metrics"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        
    @abstractmethod
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               **kwargs) -> MetricResult:
        """Compute metric score"""
        pass
        
    def batch_compute(self,
                     predictions: List[List],
                     references: List[List]) -> List[MetricResult]:
        """Compute metrics for multiple sets"""
        results = []
        for preds, refs in zip(predictions, references):
            results.append(self.compute(preds, refs))
        return results
        
    @staticmethod
    def validate_inputs(predictions: Any, references: Any) -> Tuple[Any, Any]:
        """Validate and align inputs"""
        # Convert to numpy arrays if needed
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        if isinstance(references, list):
            references = np.array(references)
            
        # Check shapes
        if predictions.shape[0] != references.shape[0]:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} "
                f"vs references {references.shape}"
            )
            
        return predictions, references
        
    def bootstrap_confidence_interval(self,
                                     predictions: np.ndarray,
                                     references: np.ndarray,
                                     n_bootstrap: int = 1000,
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap"""
        scores = []
        n_samples = len(predictions)
        
        for _ in range(n_bootstrap):
            # Random sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_preds = predictions[indices]
            boot_refs = references[indices]
            
            # Compute metric on bootstrap sample
            result = self.compute(boot_preds, boot_refs)
            scores.append(result.score)
            
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower = np.percentile(scores, alpha/2 * 100)
        upper = np.percentile(scores, (1 - alpha/2) * 100)
        
        return (lower, upper)
