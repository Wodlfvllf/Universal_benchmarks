import subprocess
import tempfile
import ast
from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics.registry import MetricRegistry
import numpy as np

class CodeGenerationTask(BaseTask):
    """Implementation for code generation tasks"""
    
    def setup(self):
        self.language = self.config.get('language', 'python')
        self.timeout = self.config.get('timeout', 5)
        self.pass_at_k_metric = MetricRegistry.get_metric('pass@1') # pass@1 is the default

    def prepare_inputs(self, dataset: Any, prompt_column: str, test_column: str) -> List[TaskInput]:
        """Prepare code generation inputs"""
        inputs = []
        
        prompts = dataset[prompt_column]
        tests = dataset[test_column]
        
        for i, prompt in enumerate(prompts):
            inputs.append(TaskInput(
                data=prompt,
                labels=tests[i],
                metadata={'index': i}
            ))
            
        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Format code generation prompt"""
        return input_data.data
    
    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate code generation predictions"""
        outputs = []
        
        prompts = [self.format_prompt(inp) for inp in inputs]
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.config.get('max_new_tokens', 512)
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation))
            
        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse code generation output"""
        return TaskOutput(predictions=raw_output)

    def execute_code(self, code: str, test_cases: List[str]) -> bool:
        """Execute generated code against test cases"""
        for test in test_cases:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(code)
                    f.write('\n\n')
                    f.write(test)
                    temp_file = f.name
                
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                
                if result.returncode != 0:
                    return False
                    
            except (subprocess.TimeoutExpired, Exception):
                return False
                
        return True

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute code generation metrics"""
        
        pass_at_1_scores = []
        for pred, ref in zip(predictions, references):
            passed = self.execute_code(pred.predictions, ref)
            pass_at_1_scores.append(float(passed))

        return {
            'pass@1': np.mean(pass_at_1_scores)
        }
