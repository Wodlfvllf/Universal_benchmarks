from sklearn.metrics import accuracy_score
from typing import List, Any, Dict

def compute_multiple_choice_metrics(predictions: List[Any], references: List[Any]) -> Dict[str, float]:
    """
    Computes and returns a dictionary of multiple-choice metrics.

    Args:
        predictions: A list of predicted choice indices from the model.
        references: A list of ground truth choice indices.

    Returns:
        A dictionary of metrics including accuracy.
    """
    return {
        'accuracy': accuracy_score(references, predictions)
    }
