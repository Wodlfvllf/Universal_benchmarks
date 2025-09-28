import numpy as np
from typing import List

def combinations(n: int, k: int) -> int:
    """Calculates the number of combinations (n choose k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res

def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculates the pass@k metric.

    Args:
        n: The total number of generated samples per problem.
        c: The number of correct samples that pass the tests.
        k: The k in pass@k.

    Returns:
        The pass@k score.
    """
    if n < k:
        # If we don't have enough samples to even attempt k correct ones, 
        # we can't calculate pass@k in its standard form. 
        # The score is effectively 0 unless c >= k, which is impossible here.
        return 0.0
    
    if c < k:
        # If the number of correct samples is less than k, we use the formula.
        # This calculates the probability that at least k of n samples are correct,
        # given that we observed c correct samples out of n.
        return 1.0 - combinations(n - c, k) / combinations(n, k)
    else:
        # If we have c >= k correct samples, the probability of picking k correct ones is 1.
        return 1.0

def estimate_pass_at_k(results: List[List[bool]], k: int) -> float:
    """
    Estimates pass@k from a list of test results for multiple problems.

    Args:
        results: A list of lists, where each inner list contains boolean
                 results (True for pass, False for fail) for one problem.
        k: The k in pass@k.

    Returns:
        The estimated pass@k score for the entire dataset.
    """
    pass_at_k_scores = []
    for problem_results in results:
        n = len(problem_results)
        c = sum(problem_results)
        score = calculate_pass_at_k(n, c, k)
        pass_at_k_scores.append(score)
    
    return np.mean(pass_at_k_scores) if pass_at_k_scores else 0.0
