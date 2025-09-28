from typing import List, Dict, Any
from ..base import BaseTask, TaskInput, TaskOutput
import numpy as np

class ImageCaptioningTask(BaseTask):
    """
    Implementation for Image Captioning tasks.
    """

    def setup(self):
        pass

    def prepare_inputs(self, dataset: Any, **kwargs) -> List[TaskInput]:
        """
        Prepares inputs for image captioning.
        Assumes dataset has 'image' and 'captions' columns.
        """
        inputs = []
        for example in dataset:
            inputs.append(TaskInput(
                data={'image': example['image']},
                labels=example.get('captions') # COCO has multiple reference captions
            ))
        return inputs

    def format_prompt(self, input_data: TaskInput) -> str:
        """Formats a prompt for image captioning. Can be empty for some models."""
        return self.config.get('prompt', "Generate a caption for the image.")

    def predict(self, model: Any, inputs: List[TaskInput], **kwargs) -> List[TaskOutput]:
        """
        Generates captions for images.
        """
        prompts = [self.format_prompt(inp) for inp in inputs]
        image_data = [inp.data['image'] for inp in inputs]

        if hasattr(model, 'multimodal_generate'):
            generations = model.multimodal_generate(images=image_data, texts=prompts, **kwargs)
        else:
            print("Warning: Model does not have 'multimodal_generate'. This task requires a multimodal model.")
            generations = ["" for _ in inputs]

        return [self.parse_output(gen) for gen in generations]

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parses the raw text output from the model."""
        return TaskOutput(predictions=raw_output.strip())

    def compute_metrics(self, predictions: List[TaskOutput], references: List[TaskInput]) -> Dict[str, float]:
        """
        Computes metrics for image captioning (e.g., BLEU, ROUGE, CIDEr).
        This is a simplified version using BLEU and ROUGE.
        """
        try:
            from rouge_score import rouge_scorer
            from nltk.translate.bleu_score import corpus_bleu
        except ImportError:
            print("Please install rouge_score and nltk: pip install rouge_score nltk")
            return {}

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = []
        
        list_of_references = []
        hypotheses = []

        for pred, ref_input in zip(predictions, references):
            hypotheses.append(pred.predictions.split())
            # NLTK corpus_bleu expects a list of reference lists
            list_of_references.append([ref.split() for ref in ref_input.labels])
            
            # ROUGE score - we take the max score against all references
            max_rouge = 0
            for ref_caption in ref_input.labels:
                score = scorer.score(ref_caption, pred.predictions)['rougeL'].fmeasure
                if score > max_rouge:
                    max_rouge = score
            rouge_scores.append(max_rouge)

        bleu_score = corpus_bleu(list_of_references, hypotheses)

        return {
            'bleu': bleu_score,
            'rougeL': np.mean(rouge_scores) if rouge_scores else 0.0
        }
