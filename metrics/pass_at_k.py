import numpy as np

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

def estimate_pass_at_k(problem_results: list, k: int) -> float:
    """Calculates the pass@k metric."""
    
    def get_n_c(res):
        n = len(res)
        c = sum(res)
        return n, c

    n, c = get_n_c(problem_results)

    if n < k:
        return 0.0
    
    if c < k:
        return 1.0 - combinations(n - c, k) / combinations(n, k)
    else:
        return 1.0