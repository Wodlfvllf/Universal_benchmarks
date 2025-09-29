
from sklearn import metrics as sklearn_metrics
import numpy as np
from typing import List, Optional, Union
from .base import BaseMetric, MetricResult

class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification"""
    
    def __init__(self, **kwargs):
        super().__init__("accuracy", **kwargs)
        
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               normalize: bool = True,
               sample_weight: Optional[np.ndarray] = None,
               **kwargs) -> MetricResult:
        """Compute accuracy score"""
        
        predictions, references = self.validate_inputs(predictions, references)
        
        # Calculate accuracy
        score = sklearn_metrics.accuracy_score(
            references,
            predictions,
            normalize=normalize,
            sample_weight=sample_weight
        )
        
        # Per-sample accuracy
        per_sample = (predictions == references).astype(float)
        
        # Calculate confidence interval if requested
        confidence_interval = None
        if kwargs.get('compute_confidence', False):
            confidence_interval = self.bootstrap_confidence_interval(
                predictions, references
            )
            
        return MetricResult(
            score=score,
            confidence_interval=confidence_interval,
            per_sample_scores=per_sample.tolist()
        )

class F1Metric(BaseMetric):
    """F1 score for classification"""
    
    def __init__(self, average: str = 'binary', **kwargs):
        super().__init__("f1", **kwargs)
        self.average = average
        
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               **kwargs) -> MetricResult:
        """Compute F1 score"""
        
        predictions, references = self.validate_inputs(predictions, references)
        
        # Calculate F1
        f1 = sklearn_metrics.f1_score(
            references,
            predictions,
            average=self.average,
            zero_division=0
        )
        
        # Also calculate precision and recall
        precision = sklearn_metrics.precision_score(
            references,
            predictions,
            average=self.average,
            zero_division=0
        )
        
        recall = sklearn_metrics.recall_score(
            references,
            predictions,
            average=self.average,
            zero_division=0
        )
        
        return MetricResult(
            score=f1,
            additional_metrics={
                'precision': precision,
                'recall': recall
            }
        )

class MatthewsCorrelationMetric(BaseMetric):
    """Matthews Correlation Coefficient"""
    
    def __init__(self, **kwargs):
        super().__init__("matthews_correlation", **kwargs)
        
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               **kwargs) -> MetricResult:
        """Compute MCC"""
        
        predictions, references = self.validate_inputs(predictions, references)
        
        mcc = sklearn_metrics.matthews_corrcoef(references, predictions)
        
        return MetricResult(score=mcc)

class ConfusionMatrixMetric(BaseMetric):
    """Confusion matrix and derived metrics"""
    
    def __init__(self, **kwargs):
        super().__init__("confusion_matrix", **kwargs)
        
    def compute(self,
               predictions: Union[List, np.ndarray],
               references: Union[List, np.ndarray],
               labels: Optional[List] = None,
               **kwargs) -> MetricResult:
        """Compute confusion matrix"""
        
        predictions, references = self.validate_inputs(predictions, references)
        
        # Calculate confusion matrix
        cm = sklearn_metrics.confusion_matrix(
            references,
            predictions,
            labels=labels
        )
        
        # Calculate per-class metrics
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_precision = cm.diagonal() / cm.sum(axis=0)
        
        # Overall metrics
        accuracy = cm.diagonal().sum() / cm.sum()
        
        return MetricResult(
            score=accuracy,
            additional_metrics={
                'per_class_accuracy': per_class_accuracy.tolist(),
                'per_class_precision': per_class_precision.tolist(),
                'confusion_matrix': cm.tolist()
            }
        )
