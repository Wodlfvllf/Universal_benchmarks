from nltk.translate.bleu_score import corpus_bleu
from typing import List

def calculate_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculates the corpus-level BLEU score.

    Args:
        predictions: A list of predicted sentences (strings).
        references: A list of lists of reference sentences. Each inner list
                    contains one or more reference translations for a single source.

    Returns:
        The corpus-level BLEU score.
    """
    if len(predictions) != len(references):
        raise ValueError("The number of predictions and reference sets must be the same.")

    # Tokenize the sentences
    tokenized_predictions = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split() for ref in ref_set] for ref_set in references]

    return corpus_bleu(tokenized_references, tokenized_predictions)
