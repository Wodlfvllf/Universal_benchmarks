from sklearn.metrics import accuracy_score
from typing import List, Any

def calculate_accuracy(predictions: List[Any], references: List[Any]) -> float:
    """
    Calculates the accuracy score.

    Args:
        predictions: A list of predictions from the model.
        references: A list of ground truth labels.

    Returns:
        The accuracy score.
    """
    if len(predictions) != len(references):
        raise ValueError("The length of predictions and references must be the same.")

    return accuracy_score(references, predictions)
