from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput

class VisualQATask(BaseTask):
    """
    Implementation for Visual Question Answering (VQA) tasks.
    """

    def setup(self):
        # VQA-specific setup, e.g., loading answer mappings
        self.answer_type = self.config.get('answer_type', 'open_ended')

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """
        Prepares inputs for VQA.
        Assumes dataset has 'image', 'question', and 'answers' columns.
        """
        inputs = []
        for example in dataset:
            inputs.append(TaskInput(
                data={
                    'image': example['image'],
                    'question': example['question']
                },
                labels=example.get('answers') # VQA often has multiple valid answers
            ))
        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Formats the prompt for VQA. The actual image data is handled separately."""
        return f"Question: {input_data.data['question']}\nAnswer:"

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """
        Generates answers for VQA. Assumes a multimodal model.
        """
        # This is a simplified representation. A real implementation would
        # require a model interface that can handle both image and text inputs.
        prompts = [self.format_prompt(inp) for inp in inputs]
        image_data = [inp.data['image'] for inp in inputs]
        
        # Assuming the model has a multimodal generate method
        if hasattr(model, 'multimodal_generate'):
            generations = model.multimodal_generate(images=image_data, texts=prompts, **kwargs)
        else:
            # Fallback for simple text-based generation if model can't handle images
            print("Warning: Model does not have 'multimodal_generate'. Falling back to text-only generation.")
            generations = model.generate(prompts, **kwargs)

        return [self.parse_output(gen) for gen in generations]

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parses the raw text output from the model."""
        return TaskOutput(predictions=raw_output.strip())

    def compute_metrics(self, predictions: List[TaskOutput], references: List[TaskInput]) -> Dict[str, float]:
        """
        Computes VQA-specific metrics.
        The standard VQA metric is complex. This is a simplified accuracy.
        """
        score = 0
        total = len(predictions)

        for pred, ref_input in zip(predictions, references):
            predicted_answer = pred.predictions.lower()
            reference_answers = [ans.lower() for ans in ref_input.labels]
            
            # Simple accuracy: score is 1 if the prediction is in the list of valid answers.
            if predicted_answer in reference_answers:
                score += 1

        return {
            'vqa_accuracy': score / total if total > 0 else 0.0
        }
