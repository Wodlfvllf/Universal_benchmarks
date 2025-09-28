from tasks.base import BaseTask, TaskInput, TaskOutput
from typing import List, Dict, Any

class TranslationTask(BaseTask):
    def setup(self):
        pass
    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        pass
    def format_prompt(self, input_data: TaskInput) -> str:
        pass
    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        pass
    def parse_output(self, raw_output: str) -> TaskOutput:
        pass
    def compute_metrics(self, predictions: List[TaskOutput], references: List[Any]) -> Dict[str, float]:
        pass