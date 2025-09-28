from sklearn.metrics import f1_score
from typing import List, Any, Literal

AverageMethod = Literal["binary", "micro", "macro", "weighted", "samples"]

def calculate_f1(predictions: List[Any], references: List[Any], average: AverageMethod = 'macro') -> float:
    """
    Calculates the F1 score.

    Args:
        predictions: A list of predictions from the model.
        references: A list of ground truth labels.
        average: The averaging method to use for multiclass targets.
                 Options: 'binary', 'micro', 'macro', 'weighted', 'samples'.

    Returns:
        The F1 score.
    """
    if len(predictions) != len(references):
        raise ValueError("The length of predictions and references must be the same.")

    return f1_score(references, predictions, average=average)
