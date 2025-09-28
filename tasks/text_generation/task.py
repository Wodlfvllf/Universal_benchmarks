from typing import List, Dict, Any
import numpy as np
from ..base import BaseTask, TaskInput, TaskOutput

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
                       references: List[Any]) -> Dict[str, float]:
        """Compute generation metrics"""
        try:
            from rouge_score import rouge_scorer
            from nltk.translate.bleu_score import sentence_bleu
        except ImportError:
            print("Please install rouge_score and nltk: pip install rouge_score nltk")
            return {}

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []

        refs = [r.labels for r in references]

        for pred, ref in zip(predictions, refs):
            if ref is None:
                continue

            pred_text = pred.predictions
            ref_text = ref

            # ROUGE scores
            scores = scorer.score(ref_text, pred_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            # BLEU score
            ref_tokens = [ref_text.split()]
            pred_tokens = pred_text.split()
            bleu = sentence_bleu(ref_tokens, pred_tokens)
            bleu_scores.append(bleu)

        return {
            'rouge1': np.mean(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': np.mean(rouge2_scores) if rouge2_scores else 0.0,
            'rougeL': np.mean(rougeL_scores) if rougeL_scores else 0.0,
            'bleu': np.mean(bleu_scores) if bleu_scores else 0.0
        }
