from typing import List

def exact_match_score(predictions: List[str], references: List[str]) -> float:
    """Calculates the exact match score."""
    score = 0
    for pred, ref in zip(predictions, references):
        if pred == ref:
            score += 1
    return score / len(predictions) if predictions else 0
