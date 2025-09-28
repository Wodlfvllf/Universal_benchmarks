from typing import List, Dict
import numpy as np

def compute_generation_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute generation metrics like ROUGE and BLEU."""
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        raise ImportError("Please install rouge_score and nltk: pip install rouge_score nltk")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    bleu_scores = []

    for pred, ref in zip(predictions, references):
        if ref is None:
            continue

        # ROUGE scores
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

        # BLEU score
        ref_tokens = [ref.split()]
        pred_tokens = pred.split()
        bleu = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(bleu)

    return {
        'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
        'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
        'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0,
        'bleu': np.mean(bleu_scores) if bleu_scores else 0.0
    }
