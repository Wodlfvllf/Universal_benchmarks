
from typing import List, Dict, Any
import numpy as np
from ..base import BaseTask, TaskInput, TaskOutput
from metrics.registry import MetricRegistry

class TextGenerationTask(BaseTask):
    """Implementation for text generation tasks"""
    
    def setup(self):
        self.max_length = self.config.get('max_length', 512)
        self.temperature = self.config.get('temperature', 0.7)
        self.top_p = self.config.get('top_p', 0.95)
        self.rouge_metric = MetricRegistry.get_metric('rouge')
        self.bleu_metric = MetricRegistry.get_metric('bleu')
        
    def prepare_inputs(self, dataset: Any, input_column: str, label_column: str) -> List[TaskInput]:
        """Prepare generation inputs"""
        inputs = []
        
        prompts = dataset[input_column]
        completions = dataset[label_column]
        
        for i, prompt in enumerate(prompts):
            inputs.append(TaskInput(
                data=prompt,
                labels=completions[i],
                metadata={'index': i}
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

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]

            generations = model.generate(
                batch_prompts,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            for j, generation in enumerate(generations):
                outputs.append(self.parse_output(generation))
            
        return outputs
    
    def parse_output(self, raw_output: str) -> TaskOutput:
        """Parse model output into structured format"""
        return TaskOutput(predictions=raw_output.strip())

    def compute_metrics(self, predictions: List[TaskOutput], 
                       references: List[Any]) -> Dict[str, float]:
        """Compute generation metrics"""
        
        preds = [p.predictions for p in predictions]
        
        rouge_result = self.rouge_metric.compute(preds, references)
        bleu_result = self.bleu_metric.compute(preds, references)
        
        return {
            'rougeL': rouge_result.score,
            'bleu': bleu_result.score,
        }
