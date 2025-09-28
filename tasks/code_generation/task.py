import subprocess
import tempfile
import ast
from typing import List, Dict, Any
from tasks.base import BaseTask, TaskInput, TaskOutput
import numpy as np

class CodeGenerationTask(BaseTask):
    """Implementation for code generation tasks"""
    
    def setup(self):
        self.language = self.config.get('language', 'python')
        self.timeout = self.config.get('timeout', 5)
        self.sandbox = self.config.get('sandbox', True)
        
    def format_prompt(self, input_data: TaskInput) -> str:
        """Format code generation prompt"""
        data = input_data.data
        
        if isinstance(data, dict):
            # HumanEval format
            prompt = data.get('prompt', '')
            if 'signature' in data:
                prompt = f"{data['signature']}\n    {data['docstring']}\n"
        else:
            prompt = data
            
        return prompt
    
    def execute_code(self, code: str, test_cases: List[Dict]) -> Dict:
        """Execute generated code against test cases"""
        results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        for test in test_cases:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    # Write code and test
                    f.write(code)
                    f.write('\n\n')
                    f.write(test['test_code'])
                    temp_file = f.name
                
                # Execute
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode == 0:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(result.stderr)
                    
            except subprocess.TimeoutExpired:
                results['failed'] += 1
                results['errors'].append("Timeout exceeded")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(str(e))
                
        return results
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute pass@k metrics"""
        pass_at_1 = []
        pass_at_10 = []
        
        for pred, ref in zip(predictions, references):
            # Execute against test cases
            test_results = self.execute_code(pred.predictions, ref['test_cases'])
            
            pass_rate = test_results['passed'] / (test_results['passed'] + test_results['failed'])
            pass_at_1.append(1.0 if pass_rate == 1.0 else 0.0)
            
        return {
            'pass@1': np.mean(pass_at_1),
            'syntax_valid': self.check_syntax_validity(predictions)
        }
    
    def check_syntax_validity(self, predictions: List[TaskOutput]) -> float:
        """Check if generated code is syntactically valid"""
        valid_count = 0
        
        for pred in predictions:
            try:
                ast.parse(pred.predictions)
                valid_count += 1
            except SyntaxError:
                pass
                
        return valid_count / len(predictions)

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """Convert dataset to task inputs"""
        pass

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """Generate predictions using model"""
        pass

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output into structured format"""
        pass