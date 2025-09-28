from typing import List, Dict, Any
import numpy as np
from ..base import BaseTask, TaskInput, TaskOutput
from .metrics import compute_generation_metrics

class TextGenerationTask(BaseTask):
    """Implementation for text generation tasks"""

    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.95)

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """Prepare generation inputs"""
        inputs = []

        for example in dataset:
            prompt = example.get('prompt', example.get('input', ''))
            reference = example.get('completion', example.get('output', None))

            inputs.append(TaskInput(
                data=prompt,
                labels=reference
            ))

        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Format generation prompt"""
        return input_data.data  # Direct pass-through for generation

    def predict(self, model: Any, inputs: List[TaskInput], 
               batch_size: int = 1) -> List[TaskOutput]:
        """Generate text completions"""
        outputs = []
        prompts = [self.format_prompt(inp) for inp in inputs]

        # Assuming model.generate can handle a list of prompts
        generations = model.generate(
            prompts,
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=self.top_p
        )

        for i, generation in enumerate(generations):
            outputs.append(TaskOutput(
                predictions=generation,
                metadata={'prompt_length': len(prompts[i])}
            ))

        return outputs

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output into structured format"""
        # For generation, the raw output is usually the desired prediction
        return TaskOutput(predictions=raw_output.strip())

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[TaskInput]) -> Dict[str, float]:
        """Compute generation metrics using the dedicated metrics module."""
        preds = [p.predictions for p in predictions]
        refs = [r.labels for r in references if r.labels is not None]
        # Ensure predictions and references are aligned after filtering
        valid_predictions = [p.predictions for i, p in enumerate(predictions) if references[i].labels is not None]

        if not valid_predictions:
            return {}

        return compute_generation_metrics(valid_predictions, refs)
