from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
import collections
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class ExtractiveQATask(BaseTask):
    """Implementation for extractive question answering tasks."""

    def setup(self):
        pass

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """Prepares inputs for extractive QA."""
        inputs = []
        for example in dataset:
            inputs.append(TaskInput(
                data={
                    'question': example['question'],
                    'context': example['context']
                },
                labels=example.get('answers', {'text': [], 'answer_start': []})
            ))
        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Formats the prompt for extractive QA."""
        return f"Context: {input_data.data['context']}\n\nQuestion: {input_data.data['question']}\n\nAnswer:"

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """Generates answers for extractive QA."""
        # This task is often handled by specialized models, but we can frame it as generation.
        prompts = [self.format_prompt(inp) for inp in inputs]
        generations = model.generate(prompts, **kwargs)
        return [self.parse_output(gen) for gen in generations]

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parses the raw text output from the model."""
        return TaskOutput(predictions=raw_output.strip())

    def compute_metrics(self, predictions: List[TaskOutput], references: List[TaskInput]) -> Dict[str, float]:
        """Computes F1 and Exact Match for extractive QA."""
        f1 = exact_match = 0
        total = len(predictions)

        for pred, ref_input in zip(predictions, references):
            predicted_answer = pred.predictions
            reference_answers = ref_input.labels['text']
            if not reference_answers:
                continue

            # Take the max F1 and EM over all possible reference answers
            f1 += max(f1_score(predicted_answer, ref) for ref in reference_answers)
            exact_match += max(normalize_answer(predicted_answer) == normalize_answer(ref) for ref in reference_answers)

        return {
            'f1': f1 / total if total > 0 else 0.0,
            'exact_match': exact_match / total if total > 0 else 0.0
        }
