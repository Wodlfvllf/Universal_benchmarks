
import re
import string
from collections import Counter
from typing import List
from .base import BaseMetric, MetricResult
import numpy as np

class ExactMatchMetric(BaseMetric):
    """Exact match metric for QA"""
    
    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__("exact_match", **kwargs)
        self.normalize_text = normalize
        
    def normalize_answer(self, text: str) -> str:
        """Normalize answer text"""
        if not self.normalize_text:
            return text
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join(ch for ch in text if ch not in string.punctuation)
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
        
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> MetricResult:
        """Compute exact match score"""
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
            
        exact_matches = []
        
        for pred, ref in zip(predictions, references):
            pred_norm = self.normalize_answer(pred)
            ref_norm = self.normalize_answer(ref)
            exact_matches.append(float(pred_norm == ref_norm))
            
        score = np.mean(exact_matches)
        
        return MetricResult(
            score=score,
            per_sample_scores=exact_matches
        )

class F1ScoreQAMetric(BaseMetric):
    """Token-level F1 score for QA"""
    
    def __init__(self, **kwargs):
        super().__init__("f1_qa", **kwargs)
        
    def compute_f1(self, prediction: str, reference: str) -> float:
        """Compute F1 score between two strings"""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
            
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
        
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> MetricResult:
        """Compute F1 scores"""
        
        f1_scores = []
        
        for pred, ref in zip(predictions, references):
            f1 = self.compute_f1(pred, ref)
            f1_scores.append(f1)
            
        return MetricResult(
            score=np.mean(f1_scores),
            per_sample_scores=f1_scores
        )
