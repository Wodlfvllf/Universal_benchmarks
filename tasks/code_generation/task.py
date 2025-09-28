import subprocess
import tempfile
import ast
from typing import List, Dict, Any
import numpy as np
from ..base import BaseTask, TaskInput, TaskOutput

class CodeGenerationTask(BaseTask):
    """Implementation for code generation tasks"""

    def setup(self):
        self.language = self.config.get('language', 'python')
        self.timeout = self.config.get('timeout', 10.0)
        self.sandbox = self.config.get('sandbox', True)

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        inputs = []
        for example in dataset:
            inputs.append(TaskInput(
                data=example,  # Keep the whole example for context
                labels=example.get('test_cases')
            ))
        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Format code generation prompt"""
        data = input_data.data
        # HumanEval-like format
        return data.get('prompt', '')

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """Generate code completions"""
        prompts = [self.format_prompt(inp) for inp in inputs]
        generations = model.generate(prompts, **kwargs)
        return [self.parse_output(gen) for gen in generations]

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output to extract code"""
        # Basic parsing, can be improved with stop strings
        return TaskOutput(predictions=raw_output)

    def execute_code(self, code: str, test_code: str) -> Dict:
        """Execute generated code against a test case"""
        # This is a simplified and potentially insecure execution method.
        # A real implementation should use a secure sandbox.
        full_code = code + "\n" + test_code
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=True) as f:
                f.write(full_code)
                f.flush()
                result = subprocess.run(
                    ['python', f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
            if result.returncode == 0:
                return {'passed': True, 'error': None}
            else:
                return {'passed': False, 'error': result.stderr}
        except subprocess.TimeoutExpired:
            return {'passed': False, 'error': 'Timeout expired'}
        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute pass@k metrics"""
        pass_at_1_scores = []
        total_tests = 0
        passed_tests = 0

        refs = [r.data for r in references] # The full example is the reference

        for pred, ref in zip(predictions, refs):
            generated_code = pred.predictions
            test_code = ref.get('test')
            if not test_code:
                continue

            # Combine the function signature from prompt with the generated body
            entry_point = ref.get('entry_point')
            prompt = ref.get('prompt')
            full_code = f"{prompt}{generated_code}"

            # This is a simplified check. HumanEval uses a more robust check.
            if entry_point:
                 check_program = f"\n\ncheck({entry_point})"
            else:
                check_program = ''

            execution_result = self.execute_code(full_code, test_code + check_program)

            total_tests += 1
            if execution_result['passed']:
                passed_tests += 1
                pass_at_1_scores.append(1.0)
            else:
                pass_at_1_scores.append(0.0)

        return {
            'pass@1': np.mean(pass_at_1_scores) if pass_at_1_scores else 0.0,
            'syntax_validity': self.check_syntax_validity(predictions)
        }

    def check_syntax_validity(self, predictions: List[TaskOutput]) -> float:
        """Check if generated code is syntactically valid"""
        valid_count = 0
        if not predictions:
            return 0.0

        for pred in predictions:
            try:
                ast.parse(pred.predictions)
                valid_count += 1
            except SyntaxError:
                pass

        return valid_count / len(predictions)
