
from typing import List, Dict, Any
import numpy as np
from .base import MetricResult

class MetricAggregator:
    """Aggregate metrics across multiple evaluations"""
    
    @staticmethod
    def aggregate_scores(results: List[MetricResult],
                        aggregation: str = "mean") -> Dict[str, float]:
        """Aggregate metric results"""
        
        if not results:
            return {}
            
        # Collect all scores
        primary_scores = [r.score for r in results]
        
        # Aggregate primary score
        if aggregation == "mean":
            agg_score = np.mean(primary_scores)
        elif aggregation == "median":
            agg_score = np.median(primary_scores)
        elif aggregation == "max":
            agg_score = np.max(primary_scores)
        elif aggregation == "min":
            agg_score = np.min(primary_scores)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
            
        aggregated = {"score": agg_score}
        
        # Aggregate additional metrics if present
        if results[0].additional_metrics:
            additional = {}
            for key in results[0].additional_metrics.keys():
                values = [r.additional_metrics.get(key, 0) for r in results]
                if aggregation == "mean":
                    additional[key] = np.mean(values)
                elif aggregation == "median":
                    additional[key] = np.median(values)
                    
            aggregated["additional_metrics"] = additional
            
        return aggregated
        
    @staticmethod
    def weighted_average(results: List[MetricResult],
                        weights: List[float]) -> float:
        """Calculate weighted average of scores"""
        
        if len(results) != len(weights):
            raise ValueError("Results and weights must have same length")
            
        scores = [r.score for r in results]
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
