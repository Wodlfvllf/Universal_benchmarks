
import sacrebleu
import numpy as np
from typing import List, Dict, Optional
from .base import BaseMetric, MetricResult

class CHRFMetric(BaseMetric):
    """CHRF score for text generation"""
    
    def __init__(self, word_order: int = 2, **kwargs):
        super().__init__("chrf", **kwargs)
        self.word_order = word_order
        
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> MetricResult:
        """Compute CHRF score"""
        
        references = [[ref] for ref in references]

        score = sacrebleu.corpus_chrf(predictions, references, word_order=self.word_order)
            
        return MetricResult(
            score=score.score,
        )
