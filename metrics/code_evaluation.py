
import ast
import subprocess
import tempfile
from typing import List, Dict
import numpy as np
from .base import BaseMetric, MetricResult
from .pass_at_k import estimate_pass_at_k

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
                score = estimate_pass_at_k(results, k)
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
