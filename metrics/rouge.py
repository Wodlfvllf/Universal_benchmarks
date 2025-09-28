from rouge_score import rouge_scorer
from typing import List, Dict
import numpy as np

def calculate_rouge(predictions: List[str], references: List[str], rouge_types: List[str] = None) -> Dict[str, float]:
    """
    Calculates ROUGE scores.

    Args:
        predictions: A list of predicted sentences.
        references: A list of reference sentences.
        rouge_types: A list of ROUGE types to calculate (e.g., ['rouge1', 'rougeL']).
                     If None, defaults to ['rouge1', 'rouge2', 'rougeL'].

    Returns:
        A dictionary of ROUGE scores (f-measure) for each type.
    """
    if len(predictions) != len(references):
        raise ValueError("The number of predictions and references must be the same.")

    if rouge_types is None:
        rouge_types = ['rouge1', 'rouge2', 'rougeL']

    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    scores = {rouge_type: [] for rouge_type in rouge_types}

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for rouge_type in rouge_types:
            scores[rouge_type].append(score[rouge_type].fmeasure)

    # Average the scores
    avg_scores = {rouge_type: np.mean(scores[rouge_type]) for rouge_type in rouge_types}
    
    return avg_scores
