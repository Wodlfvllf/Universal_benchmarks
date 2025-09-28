from tasks.base import BaseTask, TaskInput, TaskOutput
from typing import List, Dict, Any
import numpy as np

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
        
        for input_data in inputs:
            prompt = self.format_prompt(input_data)
            
            generation = model.generate(
                prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            outputs.append(TaskOutput(
                predictions=generation,
                metadata={'prompt_length': len(prompt)}
            ))
            
        return outputs
    
    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute generation metrics"""
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_text = pred.predictions
            ref_text = ref
            
            # ROUGE scores
            scores = scorer.score(ref_text, pred_text)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # BLEU score
            bleu = sentence_bleu([ref_text.split()], pred_text.split())
            bleu_scores.append(bleu)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
            'bleu': np.mean(bleu_scores)
        }

    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output into structured format"""
        pass