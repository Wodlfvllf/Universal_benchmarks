from scipy.stats import pearsonr, spearmanr
from .base import BaseMetric, MetricResult
from typing import List, Union
import numpy as np

class PearsonCorrelationMetric(BaseMetric):
    """Pearson correlation metric for regression"""
    
    def __init__(self, **kwargs):
        super().__init__("pearson", **kwargs)
        
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               **kwargs) -> MetricResult:
        """Compute Pearson correlation score"""
        
        predictions, references = self.validate_inputs(predictions, references)
        
        score, _ = pearsonr(references, predictions)
        
        return MetricResult(score=score)

class SpearmanCorrelationMetric(BaseMetric):
    """Spearman correlation metric for regression"""
    
    def __init__(self, **kwargs):
        super().__init__("spearman", **kwargs)
        
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               **kwargs) -> MetricResult:
        """Compute Spearman correlation score"""
        
        predictions, references = self.validate_inputs(predictions, references)
        
        score, _ = spearmanr(references, predictions)
        
        return MetricResult(score=score)