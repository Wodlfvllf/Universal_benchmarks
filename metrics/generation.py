
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import bert_score
import numpy as np
from typing import List, Dict
from .base import BaseMetric, MetricResult

class ROUGEMetric(BaseMetric):
    """ROUGE metrics for text generation"""
    
    def __init__(self, rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL'], **kwargs):
        super().__init__("rouge", **kwargs)
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
        
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> MetricResult:
        """Compute ROUGE scores"""
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
            
        # Calculate scores for each sample
        all_scores = {rouge_type: [] for rouge_type in self.rouge_types}
        
        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                all_scores[rouge_type].append(scores[rouge_type].fmeasure)
                
        # Average scores
        avg_scores = {
            rouge_type: np.mean(scores_list)
            for rouge_type, scores_list in all_scores.items()
        }
        
        # Use ROUGE-L as primary score
        primary_score = avg_scores.get('rougeL', list(avg_scores.values())[0])
        
        return MetricResult(
            score=primary_score,
            additional_metrics=avg_scores,
            per_sample_scores=all_scores.get('rougeL', [])
        )

class BLEUMetric(BaseMetric):
    """BLEU score for text generation"""
    
    def __init__(self, n_gram: int = 4, smooth: bool = True, **kwargs):
        super().__init__("bleu", **kwargs)
        self.n_gram = n_gram
        self.smooth = smooth
        
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> MetricResult:
        """Compute BLEU score"""
        
        # Tokenize
        pred_tokens = [pred.split() for pred in predictions]
        ref_tokens = [[ref.split()] for ref in references]
        
        # Calculate corpus BLEU
        weights = tuple([1.0/self.n_gram] * self.n_gram)
        
        if len(predictions) > 1:
            # Corpus-level BLEU
            score = corpus_bleu(ref_tokens, pred_tokens, weights=weights)
        else:
            # Sentence-level BLEU
            score = sentence_bleu(ref_tokens[0], pred_tokens[0], weights=weights)
            
        # Per-sample scores
        per_sample = []
        for pred, ref in zip(pred_tokens, ref_tokens):
            sample_score = sentence_bleu(ref, pred, weights=weights)
            per_sample.append(sample_score)
            
        return MetricResult(
            score=score,
            per_sample_scores=per_sample
        )

class BERTScoreMetric(BaseMetric):
    """BERTScore for semantic similarity"""
    
    def __init__(self, model_type: str = "bert-base-uncased", **kwargs):
        super().__init__("bertscore", **kwargs)
        self.model_type = model_type
        
    def compute(self,
               predictions: List[str],
               references: List[str],
               **kwargs) -> MetricResult:
        """Compute BERTScore"""
        
        P, R, F1 = bert_score.score(
            predictions,
            references,
            model_type=self.model_type,
            verbose=False
        )
        
        return MetricResult(
            score=F1.mean().item(),
            additional_metrics={
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            },
            per_sample_scores=F1.tolist()
        )
