from typing import Dict, Type, Any, List
from .base import BaseMetric
from .classification import (
    AccuracyMetric, F1Metric, MatthewsCorrelationMetric
)
from .generation import (
    ROUGEMetric, BLEUMetric, BERTScoreMetric
)
from .question_answering import (
    ExactMatchMetric, F1ScoreQAMetric
)
from .code_evaluation import (
    PassAtKMetric, SyntaxValidityMetric
)
from .regression import (
    PearsonCorrelationMetric, SpearmanCorrelationMetric
)

class MetricRegistry:
    """Central registry for metrics"""
    
    METRICS = {
        # Classification metrics
        'accuracy': AccuracyMetric,
        'f1': F1Metric,
        'f1_binary': lambda: F1Metric(average='binary'),
        'f1_macro': lambda: F1Metric(average='macro'),
        'f1_weighted': lambda: F1Metric(average='weighted'),
        'matthews_correlation': MatthewsCorrelationMetric,
        
        # Generation metrics
        'rouge': ROUGEMetric,
        'rouge1': lambda: ROUGEMetric(rouge_types=['rouge1']),
        'rouge2': lambda: ROUGEMetric(rouge_types=['rouge2']),
        'rougeL': lambda: ROUGEMetric(rouge_types=['rougeL']),
        'bleu': BLEUMetric,
        'bertscore': BERTScoreMetric,
        
        # QA metrics
        'exact_match': ExactMatchMetric,
        'f1_qa': F1ScoreQAMetric,
        
        # Code metrics
        'pass@1': lambda: PassAtKMetric(k_values=[1]),
        'pass@10': lambda: PassAtKMetric(k_values=[10]),
        'pass@100': lambda: PassAtKMetric(k_values=[100]),
        'syntax_validity': SyntaxValidityMetric,

        # Regression metrics
        'pearson': PearsonCorrelationMetric,
        'spearman': SpearmanCorrelationMetric,
    }
    
    @classmethod
    def get_metric(cls, name: str, **kwargs) -> BaseMetric:
        """Get metric instance by name"""
        
        if name not in cls.METRICS:
            raise ValueError(f"Unknown metric: {name}")
            
        metric_class = cls.METRICS[name]
        
        # Handle lambda functions
        if callable(metric_class) and not isinstance(metric_class, type):
            return metric_class()
        else:
            return metric_class(**kwargs)
            
    @classmethod
    def register_metric(cls, name: str, metric_class: Type[BaseMetric]):
        """Register new metric"""
        cls.METRICS[name] = metric_class
        
    @classmethod
    def list_metrics(cls) -> List[str]:
        """List all available metrics"""
        return list(cls.METRICS.keys())