
import ast
import subprocess
import tempfile
from typing import List, Dict
import numpy as np
from .base import BaseMetric, MetricResult

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
        return 0.0
    
    if c < k:
        return 1.0 - combinations(n - c, k) / combinations(n, k)
    else:
        return 1.0

class PassAtKMetric(BaseMetric):
    """Pass@k metric for code generation"""
    
    def __init__(self, k_values: List[int] = [1, 10, 100], **kwargs):
        super().__init__("pass_at_k", **kwargs)
        self.k_values = k_values
        
    def execute_code(self, code: str, test_cases: List[Dict]) -> bool:
        """Execute code against test cases"""
        for test in test_cases:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.write('\n\n')
                    f.write(test['test_code'])
                    temp_file = f.name
                    
                # Execute with timeout
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode != 0:
                    return False
                    
            except (subprocess.TimeoutExpired, Exception):
                return False
                
        return True
        
    def compute(self,
               predictions: List[List[str]],  # List of k predictions per problem
               references: List[Dict],  # Test cases for each problem
               **kwargs) -> MetricResult:
        """Compute pass@k scores"""
        
        pass_at_k = {k: [] for k in self.k_values}
        
        for problem_predictions, test_cases in zip(predictions, references):
            # Test each prediction
            results = []
            for code in problem_predictions:
                passed = self.execute_code(code, test_cases)
                results.append(passed)
                
            # Calculate pass@k for this problem
            for k in self.k_values:
                n = len(results)
                c = sum(results)
                score = calculate_pass_at_k(n, c, k)
                pass_at_k[k].append(score)
                
        # Average across problems
        scores = {f"pass@{k}": np.mean(pass_at_k[k]) for k in self.k_values}
        
        return MetricResult(
            score=scores.get("pass@1", 0.0),
            additional_metrics=scores
        )

class SyntaxValidityMetric(BaseMetric):
    """Check syntax validity of generated code"""
    
    def __init__(self, language: str = "python", **kwargs):
        super().__init__("syntax_validity", **kwargs)
        self.language = language
        
    def check_syntax(self, code: str) -> bool:
        """Check if code is syntactically valid"""
        if self.language == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        else:
            raise NotImplementedError(f"Language {self.language} not supported")
            
    def compute(self,
               predictions: List[str],
               references: List[Any] = None,
               **kwargs) -> MetricResult:
        """Compute syntax validity rate"""
        
        valid_counts = []
        
        for code in predictions:
            valid = self.check_syntax(code)
            valid_counts.append(float(valid))
            
        return MetricResult(
            score=np.mean(valid_counts),
            per_sample_scores=valid_counts
        )
