import subprocess
import tempfile
import ast
import numpy as np
from typing import List, Dict, Any

from metrics.pass_at_k import estimate_pass_at_k

def execute_code(code: str, test_code: str, timeout: float) -> Dict[str, Any]:
    """Execute generated code against a test case in a sandboxed environment."""
    full_code = code + "\n" + test_code
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                ['python', f.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
        if result.returncode == 0:
            return {'passed': True, 'error': None}
        else:
            return {'passed': False, 'error': result.stderr}
    except subprocess.TimeoutExpired:
        return {'passed': False, 'error': 'Timeout expired'}
    except Exception as e:
        return {'passed': False, 'error': str(e)}

def check_syntax_validity(predictions: List[str]) -> float:
    """Check if generated code is syntactically valid."""
    valid_count = 0
    if not predictions:
        return 0.0

    for pred in predictions:
        try:
            ast.parse(pred)
            valid_count += 1
        except SyntaxError:
            pass

    return valid_count / len(predictions)

def compute_code_generation_metrics(predictions: List[str], references: List[Dict[str, Any]], timeout: float = 10.0) -> Dict[str, float]:
    """Compute metrics for code generation, including pass@k and syntax validity."""
    problem_results = []

    for i, ref in enumerate(references):
        # Assuming predictions is a flat list and we have n_samples per problem
        # This logic needs to be aligned with how predictions are generated.
        # For simplicity, we assume one prediction per problem here.
        pred = predictions[i]
        test_code = ref.get('test')
        if not test_code:
            continue

        prompt = ref.get('prompt', '')
        full_code = f"{prompt}{pred}"
        
        execution_result = execute_code(full_code, test_code, timeout)
        problem_results.append([execution_result['passed']])

    pass_at_1 = estimate_pass_at_k(problem_results, 1)
    
    return {
        'pass@1': pass_at_1,
        'syntax_validity': check_syntax_validity(predictions)
    }
