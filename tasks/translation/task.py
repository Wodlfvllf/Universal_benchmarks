from typing import Dict, List, Any, Optional

from ..base import BaseTask, TaskInput, TaskOutput
from ...metrics import BLEUMetric, CHRFMetric

class TranslationTask(BaseTask):
    """A task for machine translation."""

    def setup(self):
        """Initialize task-specific components."""
        self.metrics = {
            "bleu": BLEUMetric(),
            "chrf": CHRFMetric(),
        }

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """Convert dataset to task inputs."""
        inputs = []
        for item in dataset:
            inputs.append(
                TaskInput(
                    data=item["translation"]["source"],
                    labels=[item["translation"]["target"]],
                )
            )
        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Format input for model consumption."""
        return f"Translate the following text to {self.config.get('target_lang', 'English')}:\n{input_data.data}"

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """Generate predictions using model."""
        prompts = [self.format_prompt(input_data) for input_data in inputs]
        predictions = model.generate(prompts, **kwargs)
        return [TaskOutput(predictions=pred) for pred in predictions]

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output into structured format."""
        return TaskOutput(predictions=raw_output.strip())

    def compute_metrics(self, predictions: List[TaskOutput], references: List[Any]) -> Dict[str, float]:
        """Compute task-specific metrics."""
        predictions = [pred.predictions for pred in predictions]
        references = [ref[0] for ref in references]

        results = {}
        for name, metric in self.metrics.items():
            metric_result = metric.compute(predictions, references)
            results[name] = metric_result.score

        return results