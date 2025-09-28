from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from typing import List, Dict, Any

def compute_classification_metrics(predictions: List[Any], references: List[Any], num_labels: int) -> Dict[str, float]:
    """
    Computes and returns a dictionary of classification metrics.

    Args:
        predictions: A list of predictions from the model.
        references: A list of ground truth labels.
        num_labels: The number of unique labels in the dataset.

    Returns:
        A dictionary of metrics including accuracy, F1, precision, and recall.
    """
    metrics = {
        'accuracy': accuracy_score(references, predictions)
    }

    if num_labels == 2:
        # Binary classification
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average='binary', zero_division=0
        )
        metrics['f1'] = f1
        metrics['precision'] = precision
        metrics['recall'] = recall
    else:
        # Multi-class
        metrics['f1_macro'] = f1_score(references, predictions, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(references, predictions, average='weighted', zero_division=0)

    return metrics
