from ..base import BaseTask, TaskInput, TaskOutput
from .metrics import compute_code_generation_metrics

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

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[TaskInput]) -> Dict[str, float]:
        """Compute code generation metrics using the dedicated metrics module."""
        preds = [p.predictions for p in predictions]
        refs = [r.data for r in references]
        return compute_code_generation_metrics(preds, refs, timeout=self.timeout)
